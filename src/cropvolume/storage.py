from pathlib import Path
import os
import shutil

from .io import atomic_json, lease, read_json

from .configuration import configuration

DEFAULT_ROOT = configuration()['output_root']


class StoragePaused(RuntimeError):
    """Pause batch admission without treating a full disk as a sample failure."""


def tree_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file() and not p.is_symlink())


def admission(used, free, reserved, estimate, budget=800_000_000_000, reserve=100_000_000_000):
    if min(used, free, reserved, estimate, budget, reserve) < 0:
        raise ValueError("Storage values must be nonnegative")
    if used + reserved + estimate > budget:
        raise StoragePaused("Dataset budget reached; new jobs paused")
    if free - reserved - estimate < reserve:
        raise StoragePaused("Free-space reserve would be consumed; new jobs paused")


class Storage:
    def __init__(self, root=DEFAULT_ROOT, budget=800_000_000_000, reserve=100_000_000_000):
        self.root, self.budget, self.reserve = Path(root), budget, reserve

    def reserve_job(self, job_id, peak_bytes):
        # Probe writing before any costly computation, including on the WSL E: mount.
        self.root.mkdir(parents=True, exist_ok=True)
        with lease(self.root / "locks" / "admission"):
            ledger_path = self.root / "reservations.json"
            ledger = read_json(ledger_path) if ledger_path.exists() else {}
            others = sum(v for k, v in ledger.items() if k != job_id)
            admission(tree_bytes(self.root), shutil.disk_usage(self.root).free, others,
                      peak_bytes, self.budget, self.reserve)
            ledger[job_id] = peak_bytes
            atomic_json(ledger_path, ledger)

    def release_job(self, job_id):
        with lease(self.root / "locks" / "admission"):
            path = self.root / "reservations.json"
            ledger = read_json(path) if path.exists() else {}
            ledger.pop(job_id, None)
            atomic_json(path, ledger)
