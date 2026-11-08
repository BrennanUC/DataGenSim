"""Unmodified authored-field reference captures; these carry no volume labels."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .cameras import capture_plan, export_cameras, import_colmap
from .io import atomic_json, child, identity, read_json, sha256
from .profiling import profiled


@profiled('reference_planning', root_arg='output')
def plan_reference(output, center_enu_m=(50., 0.), patch_m=(2., 2.), views=24,
                   width=1280, height=960):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty reference output directory")
    if len(center_enu_m) != 2 or not np.isfinite(center_enu_m).all():
        raise ValueError("Reference center must have two finite ENU coordinates")
    if len(patch_m) != 2 or not np.isfinite(patch_m).all() or min(patch_m) <= 0:
        raise ValueError("Reference patch dimensions must be positive and finite")
    # The supplied basil field's UE XY bounds map to ENU (E=UE Y, N=UE X).
    low = np.asarray(center_enu_m) - np.asarray(patch_m) / 2
    high = np.asarray(center_enu_m) + np.asarray(patch_m) / 2
    if (low < [0., -24.384]).any() or (high > [100., 24.384]).any():
        raise ValueError("Reference region must lie within the supplied basil field")
    source = {"schema": "cropvolume-field-reference-v1",
              "source_map": "/Game/FlightSchool/Maps/BasilField",
              "center_enu_m": list(center_enu_m), "patch_m": list(patch_m),
              "purpose": "Original field appearance reference; no volume labels or dataset publication",
              "field_policy": "preserve_all_authored_content",
              "coordinate_frame": "ENU_m_Z_up"}
    source["scene_id"] = identity(source)
    plan = capture_plan(source, views=views, width=width, height=height, target_height_m=.3)
    for frame in plan["frames"]:
        frame.pop("depth")
        frame["c2w_cv"][0][3] += center_enu_m[0]
        frame["c2w_cv"][1][3] += center_enu_m[1]
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "reference.json", source)
    atomic_json(output / "capture.json", plan)
    export_cameras(output, plan["frames"])
    return source


@profiled('reference_validation', root_arg='output')
def verify_reference(output):
    output = Path(output)
    source = read_json(output / "reference.json")
    scene_id = source.pop("scene_id")
    if identity(source) != scene_id:
        raise ValueError("Reference identity changed")
    plan = read_json(output / "capture.json")
    finished = read_json(output / "REFERENCE_RENDER_FINISHED.json")
    if (plan["scene_id"] != scene_id or finished.get("scene_id") != scene_id
            or finished.get("source_map") != source["source_map"]):
        raise ValueError("Reference render belongs to another map or camera plan")
    if (finished.get("field_unchanged") is not True or not finished.get("field_before_sha256")
            or finished["field_before_sha256"] != finished.get("field_after_sha256")):
        raise ValueError("Authored field changed during reference capture")
    camera_error = finished.get("maximum_camera_error", 1)
    if not np.isfinite(camera_error) or not 0 <= camera_error <= 2e-5:
        raise ValueError("Reference camera does not match calibration")
    if finished.get("frames") != len(plan["frames"]):
        raise ValueError("Reference render has missing frames")
    staged = read_json(output / "source_content.json")
    if not staged.get("files") or any(staged["files"] != staged.get(key)
                                     for key in ("files_after", "staged_files", "staged_files_after")):
        raise ValueError("Source assets changed during reference capture")
    frames = import_colmap(output)
    if len(frames) != len(plan["frames"]):
        raise ValueError("Reference camera count mismatch")
    checksums = {}
    for actual, expected in zip(frames, plan["frames"]):
        if actual["name"] != expected["name"]:
            raise ValueError("Reference camera image identity mismatch")
        for key in ("c2w_cv", "K", "width", "height"):
            if not np.allclose(actual[key], expected[key], atol=2e-5):
                raise ValueError("Reference calibration mismatch: " + key)
        relative = actual["image"]
        with Image.open(child(output, relative)) as image:
            pixels = np.asarray(image.convert("RGB"))
            if image.size != (actual["width"], actual["height"]) or pixels.std() < 3:
                raise ValueError("Invalid reference image dimensions or uniform image")
        checksums[relative] = sha256(child(output, relative))
    preview = Image.new("RGB", (960, 768), "#181818")
    draw = ImageDraw.Draw(preview)
    for i, frame_index in enumerate([0, 1, 3, len(frames) // 2]):
        frame = frames[frame_index]
        with Image.open(child(output, frame["image"])) as image:
            image = image.convert("RGB")
            image.thumbnail((480, 360))
            x, y = (i % 2) * 480, (i // 2) * 384
            preview.paste(image, (x, y))
            draw.text((x + 8, y + 364), frame["name"], fill="white")
    preview.save(output / "preview.jpg", quality=92)
    for relative in ("reference.json", "capture.json", "transforms.json", "sparse/0/cameras.txt",
                     "sparse/0/images.txt", "REFERENCE_RENDER_FINISHED.json", "source_content.json", "preview.jpg"):
        checksums[relative] = sha256(output / relative)
    receipt = {"schema": "cropvolume-field-reference-complete-v1", "scene_id": scene_id,
               "source_map": source["source_map"], "frames": len(frames), "field_unchanged": True,
               "volume_labels": False, "files": checksums}
    atomic_json(output / "REFERENCE_COMPLETE.json", receipt)
    return receipt
