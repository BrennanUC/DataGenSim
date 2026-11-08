from concurrent.futures import ThreadPoolExecutor
import json
import os
import socket

import pytest

from cropvolume.dashboard import Monitor
from cropvolume.profiling import Span, profiled, profile_directory, span, subprocess_environment, _pid_namespace


def records(root):
    return [json.loads(p.read_text()) for p in profile_directory(root).glob('*.json')]


def test_nested_timings_use_monotonic_time_and_do_not_double_count(sample, monkeypatch):
    import cropvolume.profiling as module
    clock = [0.]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    with span('pipeline', root=sample) as parent:
        clock[0] = 2
        with span('child', total=4, unit='views') as child:
            clock[0] = 3
            child.update(current=2)
            live = next(r for r in records(sample) if r['span_id'] == child.id)
            assert live['status'] == 'running' and live['progress']['current'] == 2
            clock[0] = 6
        clock[0] = 10
    data = Monitor([sample.parents[1]]).snapshot()
    rows = {r['stage']: r for r in data['profiles'] if r['span_id'] in (parent.id, child.id)}
    assert rows['pipeline']['duration_s'] == 10
    assert rows['child']['duration_s'] == 4
    assert rows['pipeline']['exclusive_s'] == 6
    assert rows['child']['progress'] == {'current': 4, 'total': 4, 'unit': 'views'}
    assert rows['child']['parent_span_id'] == parent.id
    assert rows['pipeline']['run_id'] == rows['child']['run_id']


def test_failures_and_interruptions_are_preserved(tmp_path):
    with pytest.raises(ValueError):
        with span('failing', root=tmp_path):
            raise ValueError('deliberate test failure')
    with pytest.raises(KeyboardInterrupt):
        with span('interrupted', root=tmp_path):
            raise KeyboardInterrupt()
    by_stage = {r['stage']: r for r in records(tmp_path)}
    assert by_stage['failing']['status'] == 'failed'
    assert by_stage['failing']['error']['message'] == 'deliberate test failure'
    assert by_stage['interrupted']['status'] == 'interrupted'


def test_sidecars_do_not_create_or_modify_fresh_artifact_directories(tmp_path):
    output = tmp_path / 'future-capture'
    with span('planning', root=output):
        assert not output.exists()
    assert not output.exists()
    assert len(records(output)) == 1


def test_concurrent_spans_have_independent_atomic_records(tmp_path):
    def run(i):
        with span('parallel', root=tmp_path, item=i) as timer:
            timer.update(current=i)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(run, range(24)))
    rows = records(tmp_path)
    assert len(rows) == len({r['span_id'] for r in rows}) == 24
    assert all(r['status'] == 'completed' for r in rows)
    assert not list(profile_directory(tmp_path).glob('*.tmp'))


def test_optional_telemetry_cannot_break_a_successful_operation(tmp_path, monkeypatch):
    from pathlib import Path
    original = Path.mkdir
    def denied(path, *args, **kwargs):
        if '.cropvolume_profiles' in path.parts:
            raise PermissionError('test read-only telemetry')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'mkdir', denied)
    with pytest.warns(RuntimeWarning, match='Profiling unavailable'):
        with span('work', root=tmp_path):
            result = 42
    assert result == 42


def test_disabling_profiling_skips_writes_and_gpu_synchronization(tmp_path, monkeypatch):
    monkeypatch.setenv('CROPVOLUME_PROFILING', '0')
    calls = []
    with span('work', root=tmp_path, sync=lambda: calls.append(1)):
        pass
    assert not calls and not profile_directory(tmp_path).exists()


def test_invalid_optional_metadata_does_not_break_work_or_leave_partial_records(tmp_path):
    with pytest.warns(RuntimeWarning, match='Profiling unavailable'):
        with span('work', root=tmp_path, unsupported=object()):
            result = 42
    assert result == 42
    assert records(tmp_path) == []
    assert not list(profile_directory(tmp_path).glob('*.tmp'))


def test_windows_subprocess_parent_context_is_forwarded_without_global_env_changes(tmp_path, monkeypatch):
    monkeypatch.setenv('WSLENV', 'EXISTING/u')
    monkeypatch.setenv('CROPVOLUME_PROFILING', '0')
    monkeypatch.delenv('CROPVOLUME_PROFILE_PARENT_ID', raising=False)
    with span('capture', root=tmp_path) as parent:
        env = subprocess_environment()
    assert env['CROPVOLUME_PROFILE_PARENT_ID'] == parent.id
    assert env['CROPVOLUME_PROFILE_RUN_ID'] == parent.record['run_id']
    assert 'EXISTING/u' in env['WSLENV']
    assert env['CROPVOLUME_PROFILING'] == '0' and 'CROPVOLUME_PROFILING' in env['WSLENV'].split(':')
    assert 'CROPVOLUME_PROFILE_PARENT_ID' not in os.environ


def test_abandoned_local_process_is_interrupted_without_inventing_an_end_time(sample):
    timer = Span('unfinished', root=sample).start()
    path = profile_directory(sample) / (timer.id + '.json')
    data = json.loads(path.read_text())
    data.update(pid=2**30, host=socket.gethostname(), platform=os.name, pid_namespace=_pid_namespace())
    path.write_text(json.dumps(data))
    snapshot = Monitor([sample.parents[1]]).snapshot()
    row = next(r for r in snapshot['profiles'] if r['span_id'] == timer.id)
    assert row['status'] == 'interrupted' and row['elapsed_s'] is None
