"""Low-overhead wall-clock spans, also importable by Unreal's standalone Python.

Sidecars live beside job directories so profiling never changes sealed artifacts,
fresh-output checks, or trainer inputs. One atomic file per span avoids writer races.
"""
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid
import warnings

_active = ContextVar('cropvolume_profile', default=None)
_warned = set()


def job_root(path):
    path = Path(path).resolve()
    if path.suffix in ('.json', '.jsonl', '.ply', '.npz'):
        path = path.parent
    for candidate in (path, *path.parents):
        if any((candidate / name).exists() for name in ('scene.json', 'solid_scene.json', 'reference.json')):
            return candidate
    return path


def profile_directory(root):
    root = Path(root).resolve()
    return root.parent / '.cropvolume_profiles' / root.name


def _process_identity(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _pid_namespace():
    try:
        return os.readlink('/proc/self/ns/pid')
    except OSError:
        return None


class Span:
    def __init__(self, stage, root=None, parent=None, total=None, unit=None, sync=None, **metadata):
        self.parent = parent or _active.get()
        self.root = Path(root).resolve() if root is not None else (self.parent.root if self.parent else None)
        self.sync = sync
        self.token = None
        self.enabled = self.root is not None and os.environ.get('CROPVOLUME_PROFILING', '1') != '0'
        self.id = uuid.uuid4().hex
        inherited = dict(self.parent.record.get('metadata', {})) if self.parent else {}
        inherited.update(metadata)
        self.record = {'schema': 'cropvolume-profile-span-v1', 'span_id': self.id, 'stage': stage,
            'run_id': self.parent.record['run_id'] if self.parent else os.environ.get('CROPVOLUME_PROFILE_RUN_ID', uuid.uuid4().hex),
            'parent_span_id': self.parent.id if self.parent else os.environ.get('CROPVOLUME_PROFILE_PARENT_ID'),
            'job_root': str(self.root), 'host': socket.gethostname(), 'platform': os.name,
            'pid': os.getpid(), 'process_identity': _process_identity(os.getpid()),
            'pid_namespace': _pid_namespace(),
            'status': 'running', 'metadata': inherited}
        if total is not None:
            self.record['progress'] = {'current': 0, 'total': total, 'unit': unit or 'items'}
        self.started = None

    def _write(self):
        if not self.enabled:
            return
        temporary = None
        try:
            folder = profile_directory(self.root)
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / (self.id + '.json')
            temporary = target.with_suffix('.' + uuid.uuid4().hex + '.tmp')
            self.record['updated_unix'] = time.time()
            with temporary.open('w', encoding='utf-8') as stream:
                json.dump(self.record, stream, allow_nan=False, separators=(',', ':'))
                stream.flush()
            os.replace(temporary, target)
        except (OSError, ValueError, TypeError) as error:
            self.enabled = False
            key = str(self.root)
            if key not in _warned:
                _warned.add(key)
                warnings.warn(f'Profiling unavailable at {self.root}: {error}', RuntimeWarning)
        finally:
            if temporary:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def start(self):
        if self.started is not None:
            return self
        if self.sync and self.enabled:
            self.sync()
        self.started = time.perf_counter()
        self.record.update(started_unix=time.time(), started_utc=datetime.now(timezone.utc).isoformat())
        self._write()
        return self

    def update(self, current=None, total=None, **metadata):
        if self.record['status'] != 'running':
            return
        if current is not None:
            self.record.setdefault('progress', {})['current'] = current
        if total is not None:
            self.record.setdefault('progress', {})['total'] = total
        self.record['metadata'].update(metadata)
        self._write()

    def finish(self, error=None, status=None):
        if self.started is None or self.record['status'] != 'running':
            return
        if self.sync and self.enabled:
            try:
                self.sync()
            except Exception:
                if error is None:
                    raise
        if status is None:
            status = 'completed' if error is None else ('interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed')
            if error is not None and type(error).__name__ == 'StoragePaused':
                status = 'paused'
        self.record.update(status=status, duration_s=max(0., time.perf_counter()-self.started), ended_unix=time.time())
        if error is not None:
            self.record['error'] = {'type': type(error).__name__, 'message': str(error)[:2000]}
        if status == 'completed' and 'total' in self.record.get('progress', {}):
            self.record['progress']['current'] = self.record['progress']['total']
        self._write()

    def __enter__(self):
        self.start()
        self.token = _active.set(self)
        return self

    def __exit__(self, kind, error, traceback):
        try:
            self.finish(error)
        finally:
            if self.token is not None:
                _active.reset(self.token)
        return False


def span(stage, root=None, **kwargs):
    return Span(stage, root=root, **kwargs)


def annotate(**metadata):
    current = _active.get()
    if current:
        current.update(**metadata)


def subprocess_environment():
    result = dict(os.environ)
    current = _active.get()
    if current:
        result.update(CROPVOLUME_PROFILE_RUN_ID=current.record['run_id'], CROPVOLUME_PROFILE_PARENT_ID=current.id)
        # WSL forwards only declared custom variables to Windows processes.
        values = result.get('WSLENV', '').split(':')
        for name in ('CROPVOLUME_PROFILE_RUN_ID', 'CROPVOLUME_PROFILE_PARENT_ID', 'CROPVOLUME_PROFILING'):
            if name in result and name not in values:
                values.append(name)
        result['WSLENV'] = ':'.join(v for v in values if v)
    return result


def run_process(command, root, **kwargs):
    with span('unreal_process', root=root):
        return subprocess.run(command, env=subprocess_environment(), **kwargs)


def profiled(stage, root_arg, sample_arg=None):
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            active = _active.get()
            root = active.root if active else job_root(arguments.arguments[root_arg])
            metadata = {}
            if sample_arg:
                metadata['sample_id'] = str(arguments.arguments[sample_arg])
                if not active:
                    root = Path(root) / 'samples' / metadata['sample_id']
            with span(stage, root=root, **metadata) as timer:
                result = function(*args, **kwargs)
                if isinstance(result, dict) and 'state' in result:
                    timer.update(result_state=result['state'])
                return result
        return wrapped
    return decorate
