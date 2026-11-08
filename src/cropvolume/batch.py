"""Sequential local Windows capture/WSL reconstruction with restartable stages."""
import os
from pathlib import Path
import shutil
import subprocess

from .configuration import local_path
from .io import atomic_json, identity, lease, read_json
from .prepare import seal_capture
from .storage import Storage, StoragePaused
from .worker import work_sample
from .profiling import profiled, subprocess_environment


def windows_path(path):
    if os.name == "nt":
        return str(Path(path).resolve())
    return subprocess.check_output(["wslpath", "-w", str(Path(path).resolve())], text=True).strip()


@profiled('unreal_process', root_arg='sample')
def capture(sample, catalog, runtime_root, engine_root, timeout):
    script = Path(__file__).resolve().parents[2] / "tools/run_unreal_capture.ps1"
    if not script.exists():
        raise RuntimeError("Run batch from the workspace checkout containing tools/run_unreal_capture.ps1")
    powershell = shutil.which("powershell.exe") or "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                    "-File", windows_path(script), "-Sample", windows_path(sample),
                    "-Catalog", windows_path(catalog), "-DatasetRoot", windows_path(runtime_root),
                    "-EngineRoot", windows_path(local_path(engine_root)), "-TimeoutSeconds", str(timeout)], check=True, env=subprocess_environment())


@profiled('batch', root_arg='root')
def run_batch(root, catalog, config, count, seed, runtime_root, engine_root,
              train=False, peak_bytes=2_000_000_000, capture_timeout=900, min_psnr=15., min_ssim=.3,
              stop_requested=None):
    from .cli import plan_batch
    root, catalog, runtime_root = Path(root), Path(catalog), Path(runtime_root)
    sample_ids = plan_batch(root, catalog, config, count, seed, peak_bytes)
    storage = Storage(root, config.get("budget_bytes", 800_000_000_000), config.get("reserve_bytes", 100_000_000_000))
    batch_id = identity({"samples": sample_ids})[:16]
    report = {"samples": [], "complete": 0, "prepared": 0, "failed": 0, "paused": False}
    report_path = root / "batches" / (batch_id + ".json")
    # One local batch at a time; its report cannot race another invocation.
    with lease(root / "locks" / ("batch_" + batch_id)):
        for sample_id in sample_ids:
            if stop_requested and stop_requested():
                report['stopped'] = True
                atomic_json(report_path, report)
                break
            sample = root / "samples" / sample_id
            try:
                if not (sample / "COMPLETE.json").exists() and not (sample / "CAPTURE_COMPLETE.json").exists():
                    job = "capture_" + sample_id
                    storage.reserve_job(job, peak_bytes)
                    try:
                        # All dataset roots share this staged editor/map.
                        with lease(runtime_root / "locks" / "unreal_capture"):
                            if not (sample / "RENDER_FINISHED.json").exists():
                                status_path = sample / "producer.json"
                                attempts = read_json(status_path).get("attempts", 0) if status_path.exists() else 0
                                if attempts >= 3:
                                    raise RuntimeError("Capture retry budget exhausted; inspect producer.json")
                                status = {"attempts": attempts + 1, "state": "capturing"}
                                atomic_json(status_path, status)
                                try:
                                    capture(sample, catalog, runtime_root, engine_root, capture_timeout)
                                except Exception as error:
                                    atomic_json(status_path, {**status, "state": "failed", "error": str(error)})
                                    raise
                            seal_capture(sample)
                            atomic_json(sample / "producer.json", {"attempts": read_json(sample / "producer.json").get("attempts", 0)
                                        if (sample / "producer.json").exists() else 0, "state": "sealed"})
                    finally:
                        storage.release_job(job)
                result = work_sample(root, sample_id, config.get("reconstruction_recipe", "rgb_known_pose"),
                                     not train, peak_bytes=peak_bytes, min_psnr=min_psnr, min_ssim=min_ssim)
                state = result["state"]
                if state not in ("complete", "prepared"):
                    raise RuntimeError("Worker did not finish requested stage: " + state)
                report[state] += 1
                report["samples"].append({"sample_id": sample_id, "state": state})
            except StoragePaused as error:
                report.update(paused=True, pause_reason=str(error))
                atomic_json(report_path, report)
                break
            except Exception as error:
                report["failed"] += 1
                report["samples"].append({"sample_id": sample_id, "state": "failed", "error": str(error)})
            atomic_json(report_path, report)
    return report
