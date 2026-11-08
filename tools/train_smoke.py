"""A real small fit on an analytic-plane fixture, never published as plant-volume data."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

from cropvolume.cameras import look_at
from cropvolume.io import atomic_json, identity, sha256
from cropvolume.training import train


def color_at(x, y):
    return np.stack([.4 + .2 * np.sin(7 * x), .5 + .2 * np.cos(6 * y), .3 + .1 * np.cos(3 * x + 4 * y)], -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    prepared, output = args.root / "prepared", args.root / "fit"
    prepared.mkdir(parents=True, exist_ok=True)
    frames, point_chunks, color_chunks = [], [], []
    w, h = 64, 64
    K = np.array([[65., 0, w / 2], [0, 65., h / 2], [0, 0, 1.]])
    ys, xs = np.mgrid[:h, :w]
    rays = np.linalg.inv(K) @ np.vstack([(xs + .5).ravel(), (ys + .5).ravel(), np.ones(w * h)])
    for i in range(24):
        center = [((i % 6) - 2.5) * .14, ((i // 6) - 1.5) * .14, 2.4]
        m = look_at(center, [0, 0, 0])
        directions = m[:3, :3] @ rays
        xyz = m[:3, 3] + (directions * (-m[2, 3] / directions[2])).T
        rgb = (color_at(xyz[:, 0], xyz[:, 1]) * 255).astype(np.uint8)
        split = "validation" if i % 8 == 3 else "test" if i % 8 == 7 else "train"
        name = f"view_{i:03d}.png"
        relative = "images/" + name
        (prepared / "images").mkdir(exist_ok=True)
        Image.fromarray(rgb.reshape(h, w, 3)).save(prepared / relative)
        if split == "train":
            point_chunks.append(xyz[::32])
            color_chunks.append(rgb[::32])
        frames.append({"name": name, "image": relative, "image_id": i + 1, "split": split,
                       "width": w, "height": h, "K": K.tolist(), "c2w_cv": m.tolist()})
    np.savez(prepared / "initialization.npz", points=np.concatenate(point_chunks).astype(np.float32), colors=np.concatenate(color_chunks))
    data = {"schema": "cropvolume-sim-known-pose-v1", "frames": frames, "scene_id": identity(["analytic-plane-fixture", frames]),
            "reconstruction_recipe": "depth_assisted", "pose_source": "analytic_fixture_fixed", "coordinate_frame": "ENU_m_Z_up",
            "point_count": len(np.concatenate(point_chunks)), "initialization": "initialization.npz",
            "initialization_sha256": sha256(prepared / "initialization.npz"), "scene_scale_m": 2.4,
            "holdout_scope": "Analytic plane fixture; only training-view depth and colors initialize optimization.",
            "capture_marker_sha256": "test-fixture-not-a-production-capture",
            "prepared_images_sha256": {f["image"]: sha256(prepared / f["image"]) for f in frames}}
    data["prepared_id"] = identity(data)
    atomic_json(prepared / "dataset.json", data)
    result = train(prepared, output, steps=args.steps, init_points=256, max_gaussians=512)
    result["test_fixture"] = True
    atomic_json(args.root / "smoke_result.json", result)
    print("Training fixture complete", result["gaussians"], result["test"]["mean_psnr_db"])


if __name__ == "__main__":
    main()
