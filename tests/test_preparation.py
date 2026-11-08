import numpy as np
from PIL import Image
import pytest

from cropvolume.cameras import CV_TO_GL, capture_plan, import_colmap, look_at, resize_calibration, validate_c2w
from cropvolume.io import atomic_json, read_json
from cropvolume.prepare import prepare, verify_capture, verify_prepared


def test_pose_formats_round_trip(sample):
    planned = read_json(sample / "capture.json")["frames"]
    recovered = import_colmap(sample)
    for a, b in zip(planned, recovered):
        assert np.allclose(a["c2w_cv"], b["c2w_cv"], atol=1e-10)
    for center in ([0, 0, 5], [1, 2, 3], [0, 0, -5]):
        m = look_at(center, [0, 0, 0])
        assert np.linalg.det(m[:3, :3]) == pytest.approx(1.)


def test_capture_holdouts_cover_every_elevation_without_excluding_overhead_training():
    scene = {'patch_m': [2,2], 'scene_id': 'test-only'}
    plan = capture_plan(scene, views=96)
    for ring in range(4):
        splits = [f['split'] for f in plan['frames'][ring::4]]
        assert splits.count('train') == 18
        assert splits.count('validation') == splits.count('test') == 3
    legacy = capture_plan(scene, views=24, split_policy='legacy_overhead_holdout')
    assert all(f['split'] != 'train' for f in legacy['frames'][3::4])


def test_intrinsics_resize_preserves_rays():
    K = np.array([[500., 0, 300], [0, 450, 200], [0, 0, 1]])
    resized = resize_calibration(K, 600, 400, 300, 200)
    assert np.allclose(np.linalg.inv(K) @ [120, 80, 1], np.linalg.inv(resized) @ [60, 40, 1])
    reflected = np.eye(4)
    reflected[0, 0] = -1
    with pytest.raises(ValueError, match="right handed"):
        validate_c2w(reflected)


def test_depth_preparation_excludes_heldout_initialization(sample, tmp_path):
    output = tmp_path / "prepared"
    result = prepare(sample, output, "depth_assisted", max_width=16, max_points=100)
    assert result["point_count"] == 100 and len(result["frames"]) == 24
    assert result["frames"][0]["width"] == 16
    assert not any("depth" in f for f in result["frames"])
    xyz = np.load(output / "initialization.npz")["points"]
    assert np.isfinite(xyz).all()
    # Change held-out depth and reseal checksums: training initialization stays identical.
    capture = read_json(sample / "capture.json")
    for f in capture["frames"]:
        if f["split"] != "train":
            Image.fromarray(np.full((24, 32), 50000, np.uint16)).save(sample / f["depth"])
    (sample / "CAPTURE_COMPLETE.json").unlink()
    from cropvolume.prepare import seal_capture
    seal_capture(sample, "cpu_fixture")
    second = tmp_path / "second"
    prepare(sample, second, "depth_assisted", max_width=16, max_points=100)
    assert np.array_equal(xyz, np.load(second / "initialization.npz")["points"])


def test_depth_backprojection_uses_pixel_centers(sample, tmp_path):
    result = prepare(sample, tmp_path / "prepared", "depth_assisted", max_points=100000)
    xyz = np.load(tmp_path / "prepared/initialization.npz")["points"]
    f = read_json(sample / "capture.json")["frames"][0]
    first_camera = np.linalg.inv(np.array(f["c2w_cv"])) @ np.r_[xyz[0], 1]
    assert first_camera[2] == pytest.approx(1., abs=2e-6)
    pixel = np.array(f["K"]) @ first_camera[:3]
    assert np.allclose(pixel[:2], [.5, .5], atol=3e-5)


def test_capture_tampering_and_incomplete_writes_fail(sample):
    p = sample / "images/IMG_0001.png"
    p.write_bytes(b"interrupted-write")
    with pytest.raises(ValueError, match="checksum"):
        verify_capture(sample)


def test_prepared_camera_edits_are_detected(sample, tmp_path):
    out = tmp_path / "prepared"
    prepare(sample, out, "depth_assisted", max_points=100)
    data = read_json(out / "dataset.json")
    data["frames"][0]["c2w_cv"][0][3] += 2
    atomic_json(out / "dataset.json", data)
    with pytest.raises(ValueError, match="identity"):
        verify_prepared(out)
