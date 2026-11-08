from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import time
import uuid


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def child(root, relative):
    """Manifest paths are portable and may not escape their declared root."""
    if "\\" in relative or ":" in relative:
        raise ValueError("Manifest paths must use relative POSIX notation")
    part = PurePosixPath(relative)
    if part.is_absolute() or not part.parts or ".." in part.parts:
        raise ValueError("Unsafe relative path: " + relative)
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Path escapes root")
    return result


@contextmanager
def lease(path):
    """Cross-process mkdir lease. Recover only a demonstrably dead local owner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError:
        try:
            owner = read_json(path / "owner.json")
            if owner["host"] != socket.gethostname():
                raise RuntimeError("Lease belongs to another host")
            try:
                os.kill(owner["pid"], 0)
            except ProcessLookupError:
                (path / "owner.json").unlink()
                path.rmdir()
                path.mkdir()
            else:
                raise RuntimeError("Lease owner is still running")
        except (OSError, KeyError, RuntimeError) as error:
            raise RuntimeError(f"Worker lease exists: {path}. {error}") from None
    try:
        atomic_json(path / "owner.json", {"pid": os.getpid(), "host": socket.gethostname(), "started": time.time()})
        yield
    finally:
        (path / "owner.json").unlink(missing_ok=True)
        path.rmdir()
