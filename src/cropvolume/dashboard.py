"""Read-only local monitoring of actual dataset state and profiling sidecars."""
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import mimetypes
import os
from pathlib import Path, PureWindowsPath, PurePosixPath
import shutil
import socket
import statistics
import secrets
import threading
import time
from urllib.parse import unquote, urlparse

from .profiling import profile_directory, _process_identity, _pid_namespace

STATIC = Path(__file__).resolve().parents[2] / 'dashboard/dist'
ACTIVE = {'preparing', 'training', 'capturing', 'running'}


def path_key(value):
    value = str(value)
    if os.name == 'posix' and '\\' in value:
        p = PureWindowsPath(value)
        if p.drive.lower().startswith('\\\\wsl'):
            value = '/' + '/'.join(p.parts[1:])
        elif len(p.drive) == 2 and p.drive[1] == ':':
            value = '/mnt/' + p.drive[0].lower() + '/' + '/'.join(p.parts[1:])
    return str(Path(value).resolve())


def number(value):
    return value if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) else None


def safe_child(root, relative):
    if not isinstance(relative, str) or '\\' in relative or ':' in relative:
        return None
    result = (root / relative).resolve()
    return result if result.is_relative_to(root.resolve()) else None


def count_written(root, frames):
    # One directory scan per image folder avoids thousands of cross-WSL stat calls
    # on every live refresh. DirEntry checks also exclude symlinks and directories.
    folders = {}
    for frame in frames:
        relative = frame.get('image') if isinstance(frame, dict) else None
        if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
            continue
        path = PurePosixPath(relative)
        if path.is_absolute() or '..' in path.parts:
            continue
        folders.setdefault(str(path.parent), set()).add(path.name)
    total = 0
    for relative, names in folders.items():
        folder = safe_child(root, relative)
        if folder is None:
            continue
        try:
            with os.scandir(folder) as entries:
                total += sum(entry.name in names and entry.is_file(follow_symlinks=False) for entry in entries)
        except OSError:
            pass
    return total


class Monitor:
    def __init__(self, roots, artifacts=None):
        self.roots = [Path(p).resolve() for p in roots]
        self.artifacts = Path(artifacts).resolve() if artifacts else None
        self._lock = threading.Lock()
        self._cached, self._cached_at = None, 0
        self._previews = {}

    def add_roots(self, roots):
        with self._lock:
            new = list(dict.fromkeys([*self.roots,*[Path(p).resolve() for p in roots]]))
            if new != self.roots:
                self.roots = new
                self._cached = None

    def snapshot(self, fresh=False):
        with self._lock:
            if not fresh and self._cached and time.monotonic()-self._cached_at < 1.5:
                return self._cached
            self._cached = self._collect()
            self._cached_at = time.monotonic()
            return self._cached

    def _collect(self):
        warnings = []
        def read(path):
            try:
                def reject_constant(value):
                    raise ValueError('nonfinite JSON value')
                data = json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)
                if not isinstance(data, dict):
                    raise ValueError('expected a JSON object')
                return data
            except FileNotFoundError:
                return {}
            except (OSError, ValueError) as error:
                warnings.append(f'{path.name}: {error}')
                return {}
        datasets = {}
        for root in self.roots:
            if not root.exists():
                warnings.append(f'Root is unavailable: {root}')
                continue
            candidates = [root, *(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))]
            for candidate in candidates:
                if (candidate / 'samples').is_dir():
                    datasets[str(candidate)] = candidate
        jobs, dataset_rows, profile_roots = [], [], set(datasets.values())
        terminal_runs = {}
        self._previews = {}
        for dataset in sorted(datasets.values()):
            dataset_id = hashlib.sha256(str(dataset).encode()).hexdigest()[:12]
            retired = (dataset / 'DATASET_RETIRED.json').exists()
            metadata = read(dataset / 'dataset.json')
            generation = read(dataset / 'generation.json')
            generation_status = read(dataset/'generation_status.json')
            generation_state = generation_status.get('state')
            if generation_state in ('stopped','failed','complete','finished_with_errors'):
                terminal_runs[str(dataset)] = number(generation_status.get('updated_unix')) or time.time()
            dataset_rows.append({'id': dataset_id, 'name': dataset.name, 'path': str(dataset), 'retired': retired,
                                 'config': metadata.get('config', {}),
                                 'requested_count': generation.get('settings', {}).get('count')})
            for sample in sorted((dataset / 'samples').iterdir()):
                if sample.is_dir() and not sample.name.startswith('.'):
                    job = self._job(sample, dataset_id, 'sample', read)
                    if generation_state in ('stopped','failed') and job['status'] in (ACTIVE | {'validating'}):
                        job['status']='interrupted'
                    if retired:
                        job['status'] = 'retired'
                    jobs.append(job)
                    profile_roots.add(sample)
        if self.artifacts and self.artifacts.exists():
            for path in sorted(self.artifacts.iterdir()):
                if not path.is_dir() or path.name.startswith('.'):
                    continue
                kind = ('capture' if (path / 'capture.json').exists() else
                        'region' if (path / 'region.json').exists() else
                        'inventory' if (path / 'export_request.json').exists() else None)
                if kind:
                    jobs.append(self._job(path, 'sources', kind, read))
                    profile_roots.add(path)
            # Also collect standalone measurements without inventing dataset jobs.
            sidecars = self.artifacts / '.cropvolume_profiles'
            if sidecars.exists():
                profile_roots.update(self.artifacts / p.name for p in sidecars.iterdir() if p.is_dir())
        profiles = []
        now = time.time()
        for root in sorted(profile_roots):
            directory = profile_directory(root)
            if not directory.exists():
                continue
            for path in directory.glob('*.json'):
                record = read(path)
                if record.get('schema') != 'cropvolume-profile-span-v1':
                    continue
                if not all(isinstance(record.get(k), str) for k in ('span_id', 'stage', 'status')) or not isinstance(record.get('metadata'), dict):
                    warnings.append('Ignored an incomplete profiling record: ' + path.name)
                    continue
                record['job_root'] = path_key(record.get('job_root', root))
                if record.get('status')=='running' and any((record['job_root']==r or Path(record['job_root']).is_relative_to(r))
                        and (number(record.get('started_unix')) or 0)<=ended for r,ended in terminal_runs.items()):
                    # A killed Windows process cannot finalize its timing sidecar.
                    # The coordinator confirms shutdown; do not display ghost activity
                    # or invent a duration for the unfinished span.
                    record['status']='interrupted'
                if record.get('status') == 'running':
                    if (record.get('host') == socket.gethostname() and record.get('platform') == os.name
                            and record.get('pid_namespace') == _pid_namespace()):
                        try:
                            os.kill(int(record['pid']), 0)
                            observed = _process_identity(record['pid'])
                            if record.get('process_identity') and observed != record['process_identity']:
                                record['status'] = 'interrupted'
                        except ProcessLookupError:
                            record['status'] = 'interrupted'
                        except (PermissionError, KeyError, ValueError):
                            pass
                    record['elapsed_s'] = max(0., now-(number(record.get('started_unix')) or now)) if record['status'] == 'running' else None
                else:
                    record['elapsed_s'] = number(record.get('duration_s'))
                profiles.append(record)
        by_id = {p['span_id']: p for p in profiles}
        by_path = {j['path']: j for j in jobs}
        by_sample = {(j['dataset_id'], j['name']): j for j in jobs}
        children_by_id = {}
        for p in profiles:
            children_by_id.setdefault(p.get('parent_span_id'), []).append(p)
        for p in profiles:
            p['dataset_id'] = next((d['id'] for d in dataset_rows
                if p['job_root'] == d['path'] or Path(p['job_root']).is_relative_to(d['path'])), 'sources')
            current, visited, sample_id = p, set(), None
            while current and current['span_id'] not in visited:
                visited.add(current['span_id'])
                sample_id = current.get('metadata', {}).get('sample_id')
                if sample_id:
                    break
                current = by_id.get(current.get('parent_span_id'))
            job = by_path.get(p['job_root'])
            if not job and sample_id:
                job = by_sample.get((p['dataset_id'], sample_id))
            p['job_id'] = job['id'] if job else None
            if job:
                job['profile_ids'].append(p['span_id'])
            children = children_by_id.get(p['span_id'], [])
            duration = p.get('elapsed_s')
            p['exclusive_s'] = max(0., duration-sum(c.get('elapsed_s') or 0 for c in children)) if duration is not None else None
            p['has_children'] = bool(children)
        for job in jobs:
            spans = [by_id[i] for i in job['profile_ids']]
            active = [p for p in spans if p['status'] == 'running']
            if active:
                current = max(active, key=lambda p: p.get('started_unix', 0))
                job['active_stage'] = current['stage']
                job['active_elapsed_s'] = current['elapsed_s']
                job['active_progress'] = current.get('progress')
                job['updated_unix'] = max(job['updated_unix'], current.get('updated_unix', 0))
            elif job['status'] in ACTIVE and any(p['status'] == 'interrupted' for p in spans):
                job['status'] = 'interrupted'
        samples = [j for j in jobs if j['kind'] == 'sample' and j['status'] != 'retired' and not j['test_fixture']]
        disks = []
        for root in self.roots:
            try:
                disk = shutil.disk_usage(root)
                disks.append({'path': str(root), 'total_bytes': disk.total, 'free_bytes': disk.free})
            except OSError:
                pass
        return {'schema': 'cropvolume-dashboard-v1', 'generated_unix': now,
            'datasets': dataset_rows, 'jobs': jobs, 'profiles': profiles,
            'summary': {'total': len(samples), 'accepted': sum(j['status'] == 'complete' for j in samples),
                        'failed': sum(j['status'] in ('failed', 'interrupted') for j in samples),
                        'active': sum(j['active_stage'] is not None or j['status'] in ACTIVE for j in samples)},
            'timings': timing_rows(profiles), 'disks': disks, 'warnings': warnings[:20],
            'preferred_dataset': next((d['id'] for d in dataset_rows if d['name'] == 'authored-field-pilot-96views'), 'all')}

    def _job(self, path, dataset_id, kind, read):
        job_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        worker, producer = read(path / 'worker.json'), read(path / 'producer.json')
        complete, volume = read(path / 'COMPLETE.json'), read(path / 'VOLUME_COMPLETE.json')
        scene = read(path / 'scene.json') or read(path / 'solid_scene.json')
        plan, label = read(path / 'capture.json'), read(path / 'labels/volume.json')
        metrics, history, settings, prepared = read(path / 'metrics.json'), [], {}, {}
        training = safe_child(path, worker.get('training_output', ''))
        if training and training != path:
            metrics = metrics or read(training / 'metrics.json')
            settings = read(training / 'config.json')
            try:
                history = json.loads((training / 'history.json').read_text())
                if not isinstance(history, list):
                    history = []
            except (OSError, ValueError):
                pass
        prepared_path = safe_child(path, worker.get('prepared', ''))
        if prepared_path and prepared_path != path:
            prepared = read(prepared_path / 'dataset.json')
        frames = plan.get('frames') or []
        if not isinstance(frames, list):
            frames = []
        written = count_written(path, frames)
        retired = (path / 'RETIRED.json').exists()
        error = worker.get('error') or producer.get('error')
        source_error = read(path / 'REFERENCE_RENDER_FAILED.json') or read(path / 'EXPORT_FAILED.json')
        if retired:
            status = 'retired'
        elif complete or (kind != 'sample' and (volume or (path / 'REFERENCE_COMPLETE.json').exists()
                or (path / 'REGION_COMPLETE.json').exists() or (path / 'INVENTORY_COMPLETE.json').exists())):
            status = 'complete'
        elif worker.get('state') in ('failed','interrupted') or source_error:
            status, error = worker.get('state') if worker.get('state') in ('failed','interrupted') else 'failed', error or source_error.get('error')
        elif worker.get('state') in ('preparing', 'training', 'prepared'):
            status = worker['state']
        elif producer.get('state') == 'capturing' or (written and not (path / 'CAPTURE_COMPLETE.json').exists()):
            status = 'capturing' if written < len(frames) else 'validating'
        elif (path / 'CAPTURE_COMPLETE.json').exists():
            status = 'ready'
        else:
            status = 'planned'
        phases = {'planning': bool(scene or (path / 'PLAN_COMPLETE.json').exists() or plan),
            'capture': bool((path / 'CAPTURE_COMPLETE.json').exists() or (path / 'SOLID_RENDER_FINISHED.json').exists()
                            or (path / 'REFERENCE_RENDER_FINISHED.json').exists()),
            'volume': bool(label), 'preparation': bool(prepared or complete), 'training': metrics.get('state') == 'complete',
            'publication': bool(complete)}
        modified = [p.stat().st_mtime for p in path.glob('*.json') if p.is_file()]
        job = {'id': job_id, 'name': path.name, 'path': str(path), 'dataset_id': dataset_id, 'kind': kind,
            'status': status, 'error': str(error)[:2000] if error else None, 'phases': phases,
            'capture': {'done': written, 'total': len(frames)}, 'attempts': worker.get('attempts', producer.get('attempts', 0)),
            'gaussians': metrics.get('gaussians'), 'points': prepared.get('point_count'),
            'steps': settings.get('steps'), 'step': history[-1].get('step') if history else None,
            'last_loss': number(history[-1].get('loss')) if history else None,
            'volume_m3': number(label.get('plant_solid_volume_m3')), 'label_scope': label.get('label_scope'),
            'psnr_db': number(metrics.get('test', {}).get('mean_psnr_db')),
            'ssim': number(metrics.get('test', {}).get('mean_ssim')),
            'historical_fit_s': number(metrics.get('elapsed_s')),
            'target_plants': len(label.get('instances', [])) if label else None,
            'updated_unix': max(modified, default=0), 'profile_ids': [], 'active_stage': None,
            'test_fixture': bool(scene.get('test_fixture')), 'preview_url': None}
        candidates = []
        if training and training != path:
            candidates += sorted(training.glob('test-*/*-comparison.jpg'))[:1]
        candidates += [path / n for n in ('preview.jpg', 'region_preview.png')]
        candidates += sorted((path / 'images').glob('*.png'))[-1:]
        for candidate in candidates:
            if candidate.is_file() and candidate.resolve().is_relative_to(path.resolve()):
                self._previews[job_id] = candidate
                job['preview_url'] = '/api/preview/' + job_id
                break
        return job


def timing_rows(profiles):
    groups = {}
    for p in profiles:
        if p.get('metadata', {}).get('diagnostic'):
            continue
        groups.setdefault(p['stage'], []).append(p)
    result = []
    for name, rows in groups.items():
        finished = [r for r in rows if r['status'] == 'completed' and number(r.get('exclusive_s')) is not None]
        durations = [r['exclusive_s'] for r in finished]
        result.append({'stage': name, 'calls': len(rows), 'completed': len(finished),
            'failed': sum(r['status'] in ('failed', 'interrupted') for r in rows),
            'active': sum(r['status'] == 'running' for r in rows),
            'total_s': sum(durations), 'failed_s': sum(r.get('exclusive_s') or 0 for r in rows if r['status'] in ('failed', 'interrupted')),
            'p50_s': statistics.median(durations) if durations else None,
            'min_s': min(durations) if durations else None, 'max_s': max(durations) if durations else None})
    return sorted(result, key=lambda r: r['total_s'], reverse=True)


def make_handler(monitor, generation=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, data, content_type, status=200, filename=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self'; connect-src 'self'; script-src 'self'")
            if filename:
                self.send_header('Content-Disposition', 'attachment; filename="'+filename+'"')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            route = unquote(urlparse(self.path).path)
            if generation and route == '/api/generation/options':
                if not self.local_host():
                    return self.send(b'Local host required', 'text/plain', 403)
                return self.send(json.dumps(generation.options()).encode(), 'application/json')
            if generation and route.startswith('/api/generation/') and route.endswith('/log'):
                try:
                    root = generation._root(route.split('/')[3])
                    with (root / 'generation.log').open('rb') as log:
                        log.seek(0, 2)
                        log.seek(max(0, log.tell()-64000))
                        return self.send(log.read(), 'text/plain; charset=utf-8')
                except (OSError, ValueError, IndexError):
                    return self.send(b'Log unavailable', 'text/plain', 404)
            if route in ('/api/status', '/api/report.json'):
                if generation:monitor.add_roots(p for p in generation.output_parents if p.exists())
                snapshot = dict(monitor.snapshot())
                snapshot['generation_runs'] = generation.runs() if generation else []
                body = json.dumps(snapshot, allow_nan=False).encode()
                return self.send(body, 'application/json', filename='cropvolume-report.json' if route.endswith('.json') else None)
            if route == '/api/timings.csv':
                output = io.StringIO()
                fields = ['stage', 'calls', 'completed', 'failed', 'active', 'total_s', 'failed_s', 'p50_s', 'min_s', 'max_s']
                writer = csv.DictWriter(output, fields)
                writer.writeheader()
                writer.writerows(monitor.snapshot()['timings'])
                return self.send(output.getvalue().encode(), 'text/csv', filename='cropvolume-timings.csv')
            if route.startswith('/api/preview/'):
                monitor.snapshot()
                path = monitor._previews.get(route.removeprefix('/api/preview/'))
                if path:
                    try:
                        return self.send(path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'image/jpeg')
                    except OSError:
                        pass
            elif route in ('/', '/app.js', '/generation.js', '/style.css'):
                path = STATIC / {'/': 'index.html', '/app.js': 'app.js', '/generation.js': 'generation.js', '/style.css': 'style.css'}[route]
                if path.is_file():
                    return self.send(path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'text/plain')
            self.send(b'Not found', 'text/plain', 404)

        def local_host(self):
            return urlparse('http://' + self.headers.get('Host', '')).hostname in ('localhost', '127.0.0.1', '::1')

        def do_POST(self):
            if generation is None:
                return self.send(b'Generation is unavailable', 'text/plain', 405)
            origin = self.headers.get('Origin')
            if (not self.local_host() or (origin and origin != 'http://' + self.headers.get('Host', ''))
                    or not secrets.compare_digest(self.headers.get('X-CropVolume-Token', ''), generation.token)):
                return self.send(b'Local dashboard session required', 'text/plain', 403)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16384:
                    raise ValueError('Request body must be 1–16384 bytes')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError('Expected a JSON object')
                route = unquote(urlparse(self.path).path)
                if route == '/api/generation/estimate':
                    return self.send(json.dumps(generation.estimate(body.get('settings')), allow_nan=False).encode(), 'application/json')
                if route == '/api/generation':
                    run_id = (generation.start(body.get('settings'),output_parent=body['output_parent'])
                        if 'output_parent' in body else generation.start(body.get('settings')))
                else:
                    parts = route.strip('/').split('/')
                    if len(parts) != 4 or parts[:2] != ['api', 'generation'] or parts[3] not in ('stop','resume','stop-now','clear-temporary','delete','cleanup-preview'):
                        return self.send(b'Not found', 'text/plain', 404)
                    action=parts[3];target=parts[2]
                    if action=='cleanup-preview':
                        return self.send(json.dumps(generation.cleanup_preview(target,body.get('scope','temporary'))).encode(),'application/json')
                    if action in ('clear-temporary','delete'):
                        method=generation.clear_temporary if action=='clear-temporary' else generation.delete
                        run_id=method(target,body.get('confirm_run_id'))
                    else:
                        method=generation.stop_now if action=='stop-now' else getattr(generation,action)
                        run_id=method(target)
                monitor.add_roots(p for p in generation.output_parents if p.exists())
                monitor.snapshot(fresh=True)
                return self.send(json.dumps({'run_id': run_id, 'output_parent':str(generation.parent)}).encode(), 'application/json', 202)
            except (ValueError, KeyError, TypeError, OSError) as error:
                return self.send(json.dumps({'error': str(error)}).encode(), 'application/json', 400)
            except RuntimeError as error:
                return self.send(json.dumps({'error': str(error)}).encode(), 'application/json', 409)
    return Handler


def serve(roots, artifacts=None, port=8765, generation=None):
    monitor = Monitor(roots, artifacts)
    server = ThreadingHTTPServer(('127.0.0.1', port), make_handler(monitor, generation))
    server.daemon_threads = True
    print(f'CropVolume dashboard: http://localhost:{server.server_port}', flush=True)
    print('Local dataset generator and live monitor. Press Ctrl+C to stop the dashboard.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
