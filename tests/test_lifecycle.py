import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from cropvolume.dataset import publish, rebuild_index, validate_ply
from cropvolume.io import atomic_json, child, lease, read_json, sha256
from cropvolume.storage import Storage, admission


def fixture_splat(sample):
    # Gaussian-format structural fixture; explicitly untrained and forbidden in production.
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    rows = np.zeros(4, dtype=[(n, "f4") for n in names])
    rows["rot_0"] = 1
    PlyData([PlyElement.describe(rows, "vertex")]).write(sample / "splat.ply")
    scene = read_json(sample / "scene.json")
    atomic_json(sample / "splat.manifest.json", {"scene_id": scene["scene_id"], "coordinate_frame": "ENU_m_Z_up",
        "scope": "full_scene", "sha256": sha256(sample / "splat.ply"), "capture_marker_sha256": sha256(sample / "CAPTURE_COMPLETE.json"),
        "reconstruction_recipe": "depth_assisted", "trained": False})
    atomic_json(sample / "metrics.json", {"scene_id": scene["scene_id"], "state": "complete", "reconstruction_recipe": "depth_assisted",
        "evaluation_is_withheld_from_splat_loss": True, "test": {"mean_psnr_db": 20., "mean_ssim": .8}})


def test_publication_is_idempotent_and_fixtures_are_not_production(sample):
    fixture_splat(sample)
    root = sample.parents[1]
    with pytest.raises(ValueError, match="fixtures"):
        publish(root, sample.name)
    row = publish(root, sample.name, allow_fixture=True)
    assert row == publish(root, sample.name, allow_fixture=True)
    assert len(rebuild_index(root, allow_fixture=True)) == 1
    assert rebuild_index(root) == []
    (sample / "splat.ply").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="modified"):
        rebuild_index(root, allow_fixture=True)


def test_mismatched_volume_cannot_publish(sample):
    fixture_splat(sample)
    label = read_json(sample / "labels/volume.json")
    label["scene_id"] = "another-scene"
    atomic_json(sample / "labels/volume.json", label)
    with pytest.raises(ValueError, match="checksum"):
        publish(sample.parents[1], sample.name, allow_fixture=True)


def test_retired_samples_are_removed_from_learning_index(sample):
    fixture_splat(sample)
    root=sample.parents[1]
    publish(root,sample.name,allow_fixture=True)
    atomic_json(sample/'RETIRED.json',{'reason':'rendered placement did not match plan'})
    assert rebuild_index(root,allow_fixture=True)==[]
    with pytest.raises(ValueError,match='Retired'):
        publish(root,sample.name,allow_fixture=True)


def test_quality_gates_prevent_completion(sample):
    fixture_splat(sample)
    with pytest.raises(ValueError, match="thresholds"):
        publish(sample.parents[1], sample.name, min_psnr=30, allow_fixture=True)
    assert not (sample / "COMPLETE.json").exists()


def test_disk_budget_includes_concurrent_reservations(tmp_path):
    admission(100, 1000, 50, 100, budget=300, reserve=100)
    with pytest.raises(RuntimeError, match="budget"):
        admission(100, 1000, 150, 100, budget=300, reserve=100)
    with pytest.raises(RuntimeError, match="reserve"):
        admission(100, 200, 50, 100, budget=300, reserve=100)
    storage = Storage(tmp_path, budget=1_000_000, reserve=0)
    storage.reserve_job("a", 600_000)
    with pytest.raises(RuntimeError):
        storage.reserve_job("b", 600_000)
    storage.release_job("a")
    storage.reserve_job("b", 600_000)
    assert read_json(tmp_path / "reservations.json") == {"b": 600_000}


def test_lock_and_manifest_path_safety(tmp_path):
    with lease(tmp_path / "worker"):
        with pytest.raises(RuntimeError, match="lease"):
            with lease(tmp_path / "worker"):
                pass
    assert not (tmp_path / "worker").exists()
    for path in ("../escape", "/etc/passwd", "E:\\wrong", ""):
        with pytest.raises(ValueError):
            child(tmp_path, path)
