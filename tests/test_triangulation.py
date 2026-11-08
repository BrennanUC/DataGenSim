"""Real CPU SIFT + matching + fixed-pose COLMAP on a rendered analytic plane."""
import numpy as np
from PIL import Image
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from cropvolume.cameras import look_at
from cropvolume.io import read_json
from cropvolume.triangulate import triangulate


def test_actual_cpu_colmap_preserves_known_metric_poses(tmp_path):
    pytest.importorskip("pycolmap")
    rng = np.random.default_rng(17)
    texture = gaussian_filter(rng.uniform(0, 255, (1200, 1200)), .7)
    texture = (texture - texture.min()) / (texture.max() - texture.min()) * 255
    w, h = 320, 240
    K = np.array([[260., 0, w / 2], [0, 260., h / 2], [0, 0, 1.]])
    ys, xs = np.mgrid[:h, :w]
    rays = np.linalg.inv(K) @ np.vstack([(xs + .5).ravel(), (ys + .5).ravel(), np.ones(w * h)])
    images = tmp_path / "images"
    images.mkdir()
    frames = []
    for i in range(9):
        center = [(i % 3 - 1) * .25, (i // 3 - 1) * .25, 2.5]
        m = look_at(center, [0, 0, 0])
        directions = m[:3, :3] @ rays
        distance = -m[2, 3] / directions[2]
        points = m[:3, 3, None] + directions * distance
        color = map_coordinates(texture, [points[1] * 260 + 600, points[0] * 260 + 600], order=1).reshape(h, w)
        rgb = np.repeat(color[..., None], 3, axis=2).astype(np.uint8)
        name = f"view_{i}.png"
        Image.fromarray(rgb).save(images / name)
        frames.append({"name": name, "image": "images/" + name, "image_id": i + 1, "split": "train",
                       "width": w, "height": h, "K": K.tolist(), "c2w_cv": m.tolist()})
    xyz, colors = triangulate(tmp_path, tmp_path / "triangulation", frames)
    assert len(xyz) >= 10 and colors.shape == xyz.shape
    assert np.median(np.abs(xyz[:, 2])) < .02
    report = read_json(tmp_path / "triangulation/report.json")
    assert report["max_pose_change"] < 1e-9 and not report["heldout_used"]
