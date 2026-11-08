"""Restartable stage runner. Missing capture is a waiting state, never a fake success."""
from pathlib import Path
import shutil
import uuid

from .dataset import publish
from .io import atomic_json, child, lease, read_json, sha256
from .prepare import prepare, verify_capture, verify_prepared
from .storage import Storage
from .profiling import profiled, span, annotate


@profiled('worker', root_arg='root', sample_arg='sample_id')
def work_sample(root, sample_id, recipe="rgb_known_pose", prepare_only=True,
                max_attempts=3, peak_bytes=2_000_000_000, min_psnr=15., min_ssim=.3):
    root = Path(root)
    sample = child(root / "samples", sample_id)
    if (sample / "RETIRED.json").exists():
        raise ValueError("Sample retired after placement validation; use a new dataset root")
    if not (sample / "PLAN_COMPLETE.json").exists():
        raise ValueError("Sample has no completed generation plan")
    if (sample / "COMPLETE.json").exists():
        with span('completion_verification', reused=True):
            receipt = read_json(sample / "COMPLETE.json")
            if sha256(child(root, receipt["splat_path"])) != receipt["splat_sha256"] or sha256(child(root, receipt["label_path"])) != receipt["label_sha256"]:
                raise ValueError("An accepted artifact changed after publication")
            if receipt.get('metadata_path') and sha256(child(root, receipt['metadata_path'])) != receipt['metadata_sha256']:
                raise ValueError('Accepted metadata changed after publication')
        return {"state": "complete", "sample_id": sample_id}
    if not (sample / "CAPTURE_COMPLETE.json").exists():
        return {"state": "awaiting_capture", "sample_id": sample_id}
    config = read_json(root / "dataset.json").get("config", {})
    storage = Storage(root, config.get("budget_bytes", 800_000_000_000), config.get("reserve_bytes", 100_000_000_000))
    state_path = sample / "worker.json"
    with lease(root / "locks" / (sample_id + "_worker")):
        storage.reserve_job(sample_id, peak_bytes)
        try:
            previous = read_json(state_path) if state_path.exists() else {"attempts": 0}
            marker, _ = verify_capture(sample)
            prepared = None
            if previous.get("prepared") and previous.get("capture_sha256") == sha256(sample / "CAPTURE_COMPLETE.json"):
                candidate = child(sample, previous["prepared"])
                if (candidate / "dataset.json").exists() and verify_prepared(candidate)["reconstruction_recipe"] == recipe:
                    prepared = candidate
            if prepared is not None and prepare_only:
                annotate(reused=True)
                previous["state"] = "prepared"
                previous.pop("error", None)
                atomic_json(state_path, previous)
                return previous
            if previous.get("attempts", 0) >= max_attempts and previous.get("state") == "failed":
                raise RuntimeError("Bounded retry budget exhausted; inspect worker.json")
            attempt = previous.get("attempts", 0) + 1
            state = {"state": "preparing", "attempts": attempt, "sample_id": sample_id,
                     "capture_sha256": sha256(sample / "CAPTURE_COMPLETE.json")}
            atomic_json(state_path, state)
            try:
                if prepared is None:
                    prepared = sample / "work" / ("prepare_" + uuid.uuid4().hex)
                    prepare(sample, prepared, recipe, **config.get('preparation', {}))
                state.update(state="prepared", prepared=prepared.relative_to(sample).as_posix())
                atomic_json(state_path, state)
                if prepare_only:
                    return state
                output = sample / "work" / ("train_" + uuid.uuid4().hex)
                recovered = False
                if previous.get("training_output") and previous.get("capture_sha256") == state["capture_sha256"]:
                    candidate = child(sample, previous["training_output"])
                    if (candidate / "metrics.json").exists() and (candidate / "config.json").exists():
                        settings = read_json(candidate / "config.json")
                        if settings["prepared_sha256"] == sha256(prepared / "dataset.json"):
                            output, recovered = candidate, True
                if not recovered:
                    from .training import train
                    # A per-host GPU lease; captures on Windows do not rely on this lock.
                    import socket
                    with lease(root / "locks" / (socket.gethostname() + "_gpu0")):
                        state["state"] = "training"
                        state["training_output"] = output.relative_to(sample).as_posix()
                        atomic_json(state_path, state)
                        train(prepared, output, **config.get("training", {}))
                else:
                    annotate(reused_training=True)
                state["training_output"] = output.relative_to(sample).as_posix()
                for name in ("splat.ply", "splat.manifest.json", "metrics.json"):
                    temp = sample / (name + ".tmp")
                    shutil.copyfile(output / name, temp)
                    temp.replace(sample / name)
                publish(root, sample_id, min_psnr, min_ssim,
                        quality_policy=config.get('publication', {}).get('quality_policy', 'thresholds'))
                state["state"] = "complete"
                atomic_json(state_path, state)
                return state
            except BaseException as error:
                state.update(state="failed", error=str(error))
                atomic_json(state_path, state)
                raise
        finally:
            storage.release_job(sample_id)
