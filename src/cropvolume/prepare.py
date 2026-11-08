from pathlib import Path
import numpy as np
from PIL import Image

from .cameras import import_colmap, resize_calibration, validate_c2w
from .io import atomic_json, child, identity, read_json, sha256
from .profiling import profiled, span


@profiled('capture_sealing', root_arg='sample')
def seal_capture(sample, backend="unreal"):
    """Producer calls only after all synchronous image writes have finished."""
    sample = Path(sample)
    if (sample / "CAPTURE_COMPLETE.json").exists():
        raise ValueError("Capture is already sealed")
    scene = read_json(sample / "scene.json")
    if backend not in ("unreal", "cpu_fixture") or (backend == "cpu_fixture" and not scene.get("test_fixture")):
        raise ValueError("Unsupported capture backend or unlabeled test fixture")
    capture = read_json(sample / "capture.json")
    files = ["scene.json", "labels/volume.json", "capture.json", "sparse/0/cameras.txt", "sparse/0/images.txt", "transforms.json"]
    if scene.get('schema') == 'cropvolume-authored-solid-scene-v1':
        from .authored_capture import verify_authored_capture
        verify_authored_capture(sample)
        files.extend(['sample.json', 'solid_scene.json', 'SOLID_RENDER_FINISHED.json', 'VOLUME_COMPLETE.json'])
    if backend == "unreal":
        finished = read_json(sample / "RENDER_FINISHED.json")
        if scene.get("field", {}).get("backend") == "unreal_game":
            if (finished.get("backend") != "unreal_game" or finished.get("all_plant_transforms_verified") is not True
                    or finished.get("minimum_plant_clearance_m", -1) < -1e-5
                    or not 0 <= finished.get("maximum_plant_transform_error", 1) <= 2e-5):
                raise ValueError("Native game plant placement did not pass validation")
        if scene.get('field', {}).get('backend') == 'unreal_basesim_editor':
            if (finished.get('backend') != 'unreal_basesim_random'
                    or finished.get('source_map') != scene['field']['source_map']
                    or finished.get('appearance_policy') != scene['field']['appearance_policy']
                    or not finished.get('environment_before_sha256')
                    or finished.get('environment_before_sha256') != finished.get('environment_after_sha256')
                    or finished.get('all_plant_transforms_verified') is not True
                    or not 0 <= finished.get('maximum_plant_transform_error', 1) <= 2e-5
                    or finished.get('target_plant_count') != len(scene['instances'])):
                raise ValueError('BaseSim environment or generated plant placement did not pass validation')
            if scene['field'].get('context_policy') == 'bounded_field_black_exterior_v1':
                if (finished.get('context_policy') != scene['field']['context_policy']
                        or finished.get('exterior') != 'black'
                        or finished.get('primitive_render_mode') != 'show_only_components'
                        or finished.get('clipped_ground_components') != 2
                        or finished.get('shown_components') != len(scene['instances'])+2
                        or finished.get('excluded_original_plant_instances',0) <= 0
                        or finished.get('region_center_enu_m') != scene['center_enu_m']
                        or finished.get('region_size_m') != scene['patch_m']):
                    raise ValueError('BaseSim bounded field rendering did not pass validation')
        if finished["scene_id"] != scene["scene_id"]:
            raise ValueError("Render completion belongs to another scene")
        files.append("RENDER_FINISHED.json")
    for frame in capture["frames"]:
        files.append(frame["image"])
        if frame.get("depth") and child(sample, frame["depth"]).exists():
            files.append(frame["depth"])
    marker = {"schema": "cropvolume-capture-complete-v1", "scene_id": scene["scene_id"],
              "backend": backend, "files": {f: sha256(child(sample, f)) for f in files}}
    # Validate with a provisional marker, then publish it last. It is never
    # acceptable for a consumer to discover an unvalidated completion marker.
    _verify(sample, marker)
    atomic_json(sample / "CAPTURE_COMPLETE.json", marker)
    return marker


@profiled('capture_validation', root_arg='sample')
def verify_capture(sample):
    sample = Path(sample)
    marker = read_json(sample / "CAPTURE_COMPLETE.json")
    return _verify(sample, marker)


def _verify(sample, marker):
    scene = read_json(sample / "scene.json")
    scene_hash = scene.pop("scene_id")
    if identity(scene) != scene_hash or marker["scene_id"] != scene_hash:
        raise ValueError("Capture/scene identity mismatch")
    required = {"scene.json", "labels/volume.json", "capture.json", "sparse/0/cameras.txt", "sparse/0/images.txt", "transforms.json"}
    if scene.get('schema') == 'cropvolume-authored-solid-scene-v1':
        from .authored_capture import verify_authored_capture
        verify_authored_capture(sample)
        required.update(['sample.json', 'solid_scene.json', 'SOLID_RENDER_FINISHED.json', 'VOLUME_COMPLETE.json'])
    if not required.issubset(marker["files"]):
        raise ValueError("Incomplete capture checksum manifest")
    for relative, digest in marker["files"].items():
        if sha256(child(sample, relative)) != digest:
            raise ValueError("Capture checksum mismatch: " + relative)
    frames = import_colmap(sample)
    capture = read_json(sample / "capture.json")
    if capture["scene_id"] != scene_hash or read_json(sample / "labels/volume.json")["scene_id"] != scene_hash:
        raise ValueError("Capture and label do not reference this scene")
    expected = {f["name"]: f for f in capture["frames"]}
    if set(expected) != {f["name"] for f in frames}:
        raise ValueError("Missing or extra camera views")
    for frame in frames:
        f = expected[frame["name"]]
        for key in ("c2w_cv", "K", "width", "height"):
            if not np.allclose(frame[key], f[key], atol=2e-5):
                raise ValueError("Capture camera changed: " + key)
        relative = frame["image"]
        if relative not in marker["files"]:
            raise ValueError("Image missing from capture checksums")
        with Image.open(child(sample, relative)) as image:
            if image.size != (frame["width"], frame["height"]):
                raise ValueError("Image dimensions differ from calibration")
            if (marker.get("backend") == "unreal" and scene.get('resolved_config', {}).get('capture_validation') != 'structural_only'
                    and float(np.asarray(image.convert("RGB"), dtype=float).std()) < 3):
                raise ValueError("Unreal image is nearly uniform; check exposure or missing scene geometry")
        frame.update({"split": f["split"], "depth": f.get("depth")})
    if set(f["split"] for f in frames) != {"train", "validation", "test"}:
        raise ValueError("Capture must include train, validation and test views")
    return marker, frames


def depth_initialization(sample, frames, marker, max_points=30000, seed=923):
    rng = np.random.default_rng(seed)
    points, colors = [], []
    train_frames = [f for f in frames if f["split"] == "train"]
    budget = max(1, int(np.ceil(max_points / len(train_frames))))
    for frame in train_frames:
        relative = frame.get("depth")
        if not relative or relative not in marker["files"]:
            raise ValueError("Depth-assisted recipe requires checksummed training depth")
        with Image.open(child(sample, relative)) as image:
            raw = np.asarray(image)
        if raw.shape != (frame["height"], frame["width"]) or raw.dtype.kind not in "ui" or raw.dtype.itemsize < 2:
            raise ValueError("Expected 16-bit millimeter optical-axis depth")
        # 65535 is the simulator PNG saturation value, not valid geometry.
        ys, xs = np.nonzero((raw > 0) & (raw < 65535))
        if len(xs) > budget:
            choose = rng.choice(len(xs), budget, replace=False)
            ys, xs = ys[choose], xs[choose]
        z = raw[ys, xs].astype(float) * .001
        K = np.asarray(frame["K"])
        rays = np.linalg.inv(K) @ np.vstack([xs + .5, ys + .5, np.ones(len(xs))])
        c2w = validate_c2w(frame["c2w_cv"])
        points.append((rays.T * z[:, None]) @ c2w[:3, :3].T + c2w[:3, 3])
        with Image.open(child(sample, frame["image"])) as image:
            colors.append(np.asarray(image.convert("RGB"))[ys, xs])
    xyz = np.concatenate(points).astype(np.float32)
    rgb = np.concatenate(colors).astype(np.uint8)
    if len(xyz) < 4 or not np.isfinite(xyz).all():
        raise ValueError("Insufficient valid training-view depth points")
    if len(xyz) > max_points:
        chosen = rng.choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[chosen], rgb[chosen]
    return xyz, rgb


@profiled('preparation', root_arg='sample')
def prepare(sample, output, recipe="rgb_known_pose", max_width=1024, max_points=30000):
    if max_width < 16 or max_points < 4:
        raise ValueError("Preparation needs valid image width and at least four initialization points")
    sample, output = Path(sample), Path(output)
    if (output / "dataset.json").exists():
        raise ValueError("Prepared dataset exists; use its validated identity or a new attempt directory")
    marker, frames = verify_capture(sample)
    if recipe == "depth_assisted":
        xyz, rgb = depth_initialization(sample, frames, marker, max_points)
    elif recipe == "rgb_known_pose":
        from .triangulate import triangulate
        xyz, rgb = triangulate(sample, output / "triangulation", frames)
    else:
        raise ValueError("Unknown reconstruction recipe")
    output.mkdir(parents=True, exist_ok=True)
    with span('image_preparation', total=len(frames), unit='views') as timer:
        for i, frame in enumerate(frames, 1):
            source = child(sample, frame["image"])
            target = child(output, frame["image"])
            target.parent.mkdir(parents=True, exist_ok=True)
            w = min(max_width, frame["width"])
            h = round(frame["height"] * w / frame["width"])
            frame["K"] = resize_calibration(frame["K"], frame["width"], frame["height"], w, h).tolist()
            frame["width"], frame["height"] = w, h
            frame.pop("depth", None)  # Optimization receives initialization, RGB and cameras only.
            with Image.open(source) as image:
                image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS).save(target)
            if i % 8 == 0 or i == len(frames):
                timer.update(current=i)
    np.savez(output / "initialization.npz", points=xyz, colors=rgb)
    centers = np.array([f["c2w_cv"] for f in frames])[:, :3, 3]
    scene_scale = float(max(np.linalg.norm(centers - np.median(centers, axis=0), axis=1).mean(), .1))
    data = {"schema": "cropvolume-sim-known-pose-v1", "frames": frames,
            "scene_id": marker["scene_id"], "reconstruction_recipe": recipe, "pose_source": "simulator_exact_fixed",
            "coordinate_frame": "ENU_m_Z_up", "point_count": len(xyz), "initialization": "initialization.npz",
            "initialization_sha256": sha256(output / "initialization.npz"), "scene_scale_m": scene_scale,
            "holdout_scope": "Validation/test RGB and depth excluded from optimization and initialization.",
            "capture_marker_sha256": sha256(sample / "CAPTURE_COMPLETE.json")}
    data["prepared_images_sha256"] = {f["image"]: sha256(child(output, f["image"])) for f in frames}
    data["prepared_id"] = identity(data)
    atomic_json(output / "dataset.json", data)
    return data


def verify_prepared(root):
    root = Path(root)
    data = read_json(root / "dataset.json")
    expected = data.pop("prepared_id")
    if identity(data) != expected:
        raise ValueError("Prepared metadata identity mismatch")
    data["prepared_id"] = expected
    if sha256(child(root, data["initialization"])) != data["initialization_sha256"]:
        raise ValueError("Initialization checksum mismatch")
    for relative, digest in data["prepared_images_sha256"].items():
        if sha256(child(root, relative)) != digest:
            raise ValueError("Prepared RGB checksum mismatch")
    for frame in data["frames"]:
        validate_c2w(frame["c2w_cv"])
    return data
