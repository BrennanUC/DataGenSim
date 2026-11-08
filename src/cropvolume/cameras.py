"""All optimization cameras are OpenCV c2w in metric ENU. No scene normalization."""
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from .io import atomic_json, child, read_json

CV_TO_GL = np.diag([1., -1., -1., 1.])


def validate_c2w(matrix):
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("Invalid camera matrix")
    r = m[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=2e-5) or not np.isclose(np.linalg.det(r), 1., atol=2e-5):
        raise ValueError("Camera rotation must be right handed and orthonormal")
    return m


def look_at(center, target):
    center, target = np.asarray(center, float), np.asarray(target, float)
    forward = target - center
    if np.linalg.norm(forward) < 1e-8:
        raise ValueError("Camera and target coincide")
    forward /= np.linalg.norm(forward)
    up = np.array([0., 0., 1.]) if abs(forward[2]) < .99 else np.array([0., 1., 0.])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack([right, down, forward])
    c2w[:3, 3] = center
    return validate_c2w(c2w)


def capture_plan(scene, views=64, width=1024, height=768, target_height_m=.3, split_policy='balanced_rings', camera_height_m=None):
    if views < 24 or width < 16 or height < 16:
        raise ValueError("Use at least 24 views and valid image dimensions")
    if not np.isfinite(target_height_m) or target_height_m < 0:
        raise ValueError("Camera target height must be finite and nonnegative")
    if camera_height_m is not None and (not np.isfinite(camera_height_m) or camera_height_m <= target_height_m):
        raise ValueError('Camera height must be finite and above the target height')
    if split_policy not in ('balanced_rings', 'legacy_overhead_holdout'):
        raise ValueError('Unknown capture split policy')
    radius = max(scene["patch_m"]) * 1.2
    anchor = np.asarray(scene.get('center_enu_m', [0.,0.]) if scene.get('field', {}).get('backend') == 'unreal_basesim_editor' else [0.,0.], dtype=float)
    if anchor.shape != (2,) or not np.isfinite(anchor).all():
        raise ValueError('Invalid field camera center')
    frames = []
    for i in range(views):
        # Three oblique elevations plus overhead ring. Target is fixed across views.
        ring = i % 4
        # Stagger holdouts within each elevation ring. The legacy i%8 schedule
        # held out every overhead camera, leaving no overhead training coverage.
        split_key = ((i // 4) + 2*ring) % 8 if split_policy == 'balanced_rings' else i % 8
        angle = 2 * math.pi * (i // 4) / math.ceil(views / 4) + ring * .09
        r = radius if ring < 3 else radius * .25
        z = radius * [.4, .8, 1.2, 1.8][ring]
        center = [anchor[0] + r * math.cos(angle), anchor[1] + r * math.sin(angle), z + target_height_m if camera_height_m is None else camera_height_m]
        fx = width / (2 * math.tan(math.radians(60) / 2))
        frames.append({"name": f"IMG_{i + 1:04d}.png", "image": f"images/IMG_{i + 1:04d}.png",
                       "depth": f"depth/IMG_{i + 1:04d}.png", "image_id": i + 1,
                       "width": width, "height": height, "K": [[fx, 0., width / 2], [0., fx, height / 2], [0., 0., 1.]],
                       "c2w_cv": look_at(center, [*anchor, target_height_m]).tolist(),
                       "split": "validation" if split_key == 3 else "test" if split_key == 7 else "train"})
    return {"schema": "cropvolume-capture-plan-v1", "scene_id": scene["scene_id"], "frames": frames,
            "coordinate_frame": "ENU_m_Z_up", "pose_source": "simulator_exact", "depth_scale_m": .001,
            "split_policy": split_policy}


def resize_calibration(K, old_width, old_height, width, height):
    if min(old_width, old_height, width, height) <= 0:
        raise ValueError("Image dimensions must be positive")
    K = np.asarray(K, float).copy()
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("Invalid intrinsic matrix")
    K[0] *= width / old_width
    K[1] *= height / old_height
    return K


def export_cameras(root, frames):
    root = Path(root)
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    cameras, images, transforms = [], [], []
    for i, frame in enumerate(frames, 1):
        m = validate_c2w(frame["c2w_cv"])
        K = np.asarray(frame["K"])
        camera_id = frame.get("camera_id", i)
        cameras.append(f"{camera_id} PINHOLE {frame['width']} {frame['height']} {K[0,0]:.12g} {K[1,1]:.12g} {K[0,2]:.12g} {K[1,2]:.12g}\n")
        view = np.linalg.inv(m)
        q = Rotation.from_matrix(view[:3, :3]).as_quat()[[3, 0, 1, 2]]
        values = " ".join(f"{v:.12g}" for v in [*q, *view[:3, 3]])
        images.append(f"{frame['image_id']} {values} {camera_id} {frame['name']}\n\n")
        transforms.append({"file_path": frame["image"], "transform_matrix": (m @ CV_TO_GL).tolist(),
                           "fl_x": float(K[0, 0]), "fl_y": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                           "w": frame["width"], "h": frame["height"]})
    (sparse / "cameras.txt").write_text("# PINHOLE cameras\n" + "".join(cameras))
    (sparse / "images.txt").write_text("# Exact poses, OpenCV world-to-camera\n" + "".join(images))
    (sparse / "points3D.txt").write_text("# Empty until triangulation; never estimate poses\n")
    atomic_json(root / "transforms.json", {"camera_model": "OPENCV", "k1": 0., "k2": 0., "p1": 0., "p2": 0., "frames": transforms})


def import_colmap(root):
    root = Path(root)
    cameras = {}
    for line in (root / "sparse/0/cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if p[1] != "PINHOLE":
            raise ValueError("Simulator adapter requires distortion-free PINHOLE cameras")
        fx, fy, cx, cy = map(float, p[4:8])
        cameras[int(p[0])] = (int(p[2]), int(p[3]), [[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    lines = iter((root / "sparse/0/images.txt").read_text().splitlines())
    frames = []
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split(maxsplit=9)
        if len(p) != 10:
            raise ValueError("Malformed COLMAP image pose")
        q = np.array(list(map(float, p[1:5])))
        if not np.isclose(np.linalg.norm(q), 1., atol=1e-5):
            raise ValueError("Invalid camera quaternion")
        view = np.eye(4)
        view[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        view[:3, 3] = list(map(float, p[5:8]))
        width, height, K = cameras[int(p[8])]
        name = p[9]
        child(root / "images", name)
        frames.append({"name": name, "image": "images/" + name, "image_id": int(p[0]), "width": width,
                       "height": height, "K": K, "c2w_cv": validate_c2w(np.linalg.inv(view)).tolist()})
        try:
            next(lines)  # Observation line may be empty or contain tracked points.
        except StopIteration:
            raise ValueError("COLMAP image is missing its observation line") from None
    frames.sort(key=lambda f: f["image_id"])
    if len({f["image_id"] for f in frames}) != len(frames) or len({f["name"] for f in frames}) != len(frames):
        raise ValueError("Duplicate images")
    transforms = root / "transforms.json"
    if transforms.exists():
        by_name = {Path(f["file_path"]).name: f for f in read_json(transforms)["frames"]}
        for f in frames:
            gl = np.array(by_name[f["name"]]["transform_matrix"])
            if not np.allclose(gl @ CV_TO_GL, f["c2w_cv"], atol=2e-5):
                raise ValueError("COLMAP and OpenGL cameras disagree")
    return frames
