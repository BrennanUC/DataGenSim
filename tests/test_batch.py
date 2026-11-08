import numpy as np
import pytest

from cropvolume.batch import run_batch
from cropvolume.io import atomic_json, read_json
from cropvolume.storage import StoragePaused
from cropvolume.training import _initialization_indices


def test_failed_captures_release_storage_and_have_bounded_retries(catalog, tmp_path, monkeypatch):
    import cropvolume.batch as batch
    calls = []
    def unavailable(*args):
        calls.append(args[0])
        raise RuntimeError("Deliberately unavailable editor in CPU test")
    monkeypatch.setattr(batch, "capture", unavailable)
    root = tmp_path / "batch"
    config = {"reserve_bytes": 0, "scene": {"plant_count": [1, 1]}}
    for _ in range(4):
        result = run_batch(root, catalog, config, 2, 21, tmp_path / "runtime", "test-engine", peak_bytes=1000)
        assert result["failed"] == 2 and result["complete"] == 0
        assert read_json(root / "reservations.json") == {}
    assert len(calls) == 6  # Three real attempts per sample; fourth invocation is refused.
    assert not list(root.glob("samples/*/COMPLETE.json"))


def test_batch_reuses_sealed_cpu_preparation_without_invoking_editor(sample, catalog, monkeypatch):
    import cropvolume.cli
    import cropvolume.batch as batch
    root = sample.parents[1]
    config = {"reserve_bytes": 0, "reconstruction_recipe": "depth_assisted"}
    atomic_json(root / "dataset.json", {"config": config})
    atomic_json(sample / "PLAN_COMPLETE.json", {})
    monkeypatch.setattr(cropvolume.cli, "plan_batch", lambda *args: [sample.name])
    def must_not_render(*args):
        pytest.fail("Sealed captures must not be rendered again")
    monkeypatch.setattr(batch, "capture", must_not_render)
    first = run_batch(root, catalog, config, 1, 21, root, "test-engine", peak_bytes=1000)
    state = read_json(sample / "worker.json")
    second = run_batch(root, catalog, config, 1, 21, root, "test-engine", peak_bytes=1000)
    assert first == second and first["prepared"] == 1
    assert read_json(sample / "worker.json") == state
    assert not (sample / "COMPLETE.json").exists()


def test_low_space_pauses_batch_without_failing_sample(catalog, tmp_path, monkeypatch):
    import cropvolume.batch as batch
    class FullStorage:
        def __init__(self, *args):
            pass
        def reserve_job(self, *args):
            raise StoragePaused("Free-space reserve would be consumed")
    monkeypatch.setattr(batch, "Storage", FullStorage)
    result = run_batch(tmp_path / "batch", catalog, {"reserve_bytes": 0, "scene": {"plant_count": [1, 1]}},
                       2, 21, tmp_path / "runtime", "test-engine", peak_bytes=1000)
    assert result["paused"] and result["failed"] == 0 and result["samples"] == []


def test_rgb_seed_sampling_keeps_rare_green_points():
    colors = np.tile([.5, .3, .2], (1000, 1))
    colors[:4] = [.1, .7, .1]
    chosen = _initialization_indices(colors, 100, np.random.default_rng(3), True)
    assert set(range(4)).issubset(chosen) and len(set(chosen)) == 100
