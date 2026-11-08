import os
from pathlib import Path
import socket
import numpy as np
import pytest

from cropvolume.cli import plan_batch
from cropvolume.io import atomic_json, lease, read_json
from cropvolume.worker import work_sample


def test_planning_restart_is_idempotent(catalog, tmp_path):
    root = tmp_path / "planned"
    config = {"reserve_bytes": 0, "scene": {"plant_count": [1, 1]}, "capture": {"views": 24, "width": 32, "height": 24}}
    plan_batch(root, catalog, config, 2, 41, 1000)
    before = {p: p.read_bytes() for p in root.glob("samples/*/scene.json")}
    plan_batch(root, catalog, config, 2, 41, 1000)
    assert before == {p: p.read_bytes() for p in before}
    sample_id = next((root / "samples").iterdir()).name
    assert work_sample(root, sample_id)["state"] == "awaiting_capture"


def test_cpu_worker_prepares_and_reuses_completed_stage(sample):
    root = sample.parents[1]
    atomic_json(root / "dataset.json", {"config": {"reserve_bytes": 0}})
    atomic_json(sample / "PLAN_COMPLETE.json", {"scene_id": read_json(sample / "scene.json")["scene_id"]})
    first = work_sample(root, sample.name, recipe="depth_assisted", peak_bytes=1000)
    second = work_sample(root, sample.name, recipe="depth_assisted", peak_bytes=1000)
    assert first == second
    assert first["state"] == "prepared"
    assert read_json(root / "reservations.json") == {}
    assert not (sample / "COMPLETE.json").exists()


def test_dead_local_lease_is_recovered(tmp_path):
    path = tmp_path / "worker"
    path.mkdir()
    # Find an impossible PID on this Linux test host without touching any process.
    pid = 2**30
    atomic_json(path / "owner.json", {"pid": pid, "host": socket.gethostname()})
    with lease(path):
        assert read_json(path / "owner.json")["pid"] == os.getpid()


def test_remote_lease_is_never_stolen(tmp_path):
    path = tmp_path / "worker"
    path.mkdir()
    atomic_json(path / "owner.json", {"pid": 2**30, "host": "another-machine"})
    with pytest.raises(RuntimeError, match="another host"):
        with lease(path):
            pass


def test_gpu_failure_preserves_cpu_stage_and_releases_space(sample, monkeypatch):
    root = sample.parents[1]
    atomic_json(root / "dataset.json", {"config": {"reserve_bytes": 0}})
    atomic_json(sample / "PLAN_COMPLETE.json", {})
    import cropvolume.training
    def unavailable(*args, **kwargs):
        raise RuntimeError("GPU unavailable in test")
    monkeypatch.setattr(cropvolume.training, "train", unavailable)
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        work_sample(root, sample.name, "depth_assisted", prepare_only=False, peak_bytes=1000)
    failed = read_json(sample / "worker.json")
    assert failed["state"] == "failed" and (sample / failed["prepared"] / "dataset.json").exists()
    assert read_json(root / "reservations.json") == {}
    assert work_sample(root, sample.name, "depth_assisted", peak_bytes=1000)["state"] == "prepared"
