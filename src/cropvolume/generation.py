"""Dashboard-controlled random-field generation using the existing local pipeline."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time
import uuid

from .io import atomic_json, lease, read_json
from .profiling import _process_identity
from .storage import DEFAULT_ROOT
from .configuration import configuration, local_path, save_preferences

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {'name': 'random-field', 'count': 10, 'seed': 923, 'species': 'basil',
    'field_width_m': 2., 'field_length_m': 2.,
    'layout': 'scatter', 'spacing_m': .08, 'scale_min': .8, 'scale_max': 1.2,
    'image_width': 1024, 'image_height': 768, 'views': 64,
    'camera_height_m': 3., 'target_height_m': .15, 'steps': 4000, 'max_gaussians': 100000}
RUN_NAME = re.compile(r'random-field-[a-z0-9-]+\Z')
POPULATION_POLICY = {'method': 'field-area-density-v1', 'plants_per_m2': [5, 10],
                     'maximum_plants': 10000}
LEGACY_BASESIM_FIELD = {'backend':'unreal_basesim_editor', 'source_map':'/Game/FlightSchool/Maps/BasilField',
    'center_enu_m':[50.,0.], 'bed_top_m':.08,
    'appearance_policy':'authored_map_lighting_materials_postprocess',
    'context_policy':'Preserve authored plants outside target bounds; replace whole original plants intersecting target bounds',
    'replace_meshes':[f'/Game/FlightSchool/Basil/BasilClump_v{i}.BasilClump_v{i}' for i in range(4)]+
        ['/Game/FlightSchool/Basil/BasilClumpContext.BasilClumpContext']}
BASESIM_FIELD = {**LEGACY_BASESIM_FIELD,
    'context_policy':'bounded_field_black_exterior_v1',
    'row_layout':'authored_basesim_anchors_v1'}


def plant_count_range(width, length, layout='scatter', scale_max=1.2, spacing=.08):
    if layout == 'rows':
        from .basesim_rows import row_count_range
        return row_count_range(width, length, scale_max, spacing)
    area = width * length
    low, high = [max(1, math.floor(area * density + 1e-9))
                 for density in POPULATION_POLICY['plants_per_m2']]
    if high > POPULATION_POLICY['maximum_plants']:
        raise ValueError('Field area is too large for the 10,000-plant limit; use at most 1,000 m²')
    return [low, high]


def validate_settings(value, *, derive_population=True):
    if not isinstance(value, dict) or set(value)-set(DEFAULTS):
        raise ValueError('Settings must contain only supported generation controls')
    result = {**DEFAULTS, **value}
    for key, low, high in [('count', 1, 10000), ('seed', 0, 2**31-1),
            ('image_width', 64, 4096), ('image_height', 64, 4096),
            ('views', 24, 512), ('steps', 100, 100000), ('max_gaussians', 4, 2000000)]:
        n = result[key]
        if isinstance(n, bool) or not isinstance(n, int) or not low <= n <= high:
            raise ValueError(f'{key} must be an integer from {low} to {high}')
    for key, low, high in [('field_width_m', .1, 100.), ('field_length_m', .1, 100.),
            ('spacing_m', 0., 10.), ('scale_min', .1, 10.), ('scale_max', .1, 10.),
            ('camera_height_m', .1, 1000.), ('target_height_m', 0., 100.)]:
        n = result[key]
        if isinstance(n, bool) or not isinstance(n, (float, int)) or not math.isfinite(n) or not low <= n <= high:
            raise ValueError(f'{key} must be a number from {low} to {high}')
    if result['scale_min'] > result['scale_max']:
        raise ValueError('Minimum values must not exceed maximum values')
    if result['camera_height_m'] <= result['target_height_m']:
        raise ValueError('Camera height must exceed camera target height')
    if result['species'] not in ('basil', 'cilantro', 'mixed') or result['layout'] not in ('scatter', 'rows'):
        raise ValueError('Choose basil, cilantro, or mixed species and scatter or rows layout')
    if not isinstance(result['name'], str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,39}', result['name']):
        raise ValueError('Run name must use 1–40 lowercase letters, numbers or hyphens')
    if derive_population:
        plant_count_range(result['field_width_m'], result['field_length_m'], result['layout'], result['scale_max'], result['spacing_m'])
    return result


def pipeline_config(settings):
    s = validate_settings(settings)
    counts = plant_count_range(s['field_width_m'], s['field_length_m'], s['layout'], s['scale_max'], s['spacing_m'])
    config = _pipeline_config(s, counts)
    config['scene']['population_policy'] = {**POPULATION_POLICY,
        'area_m2': s['field_width_m'] * s['field_length_m']}
    if s['layout'] == 'rows':
        config['scene']['population_policy'] = {'method':'authored-row-capacity-v1',
            'occupancy_fraction':[.9,1.], 'maximum_plants':10000,
            'area_m2':s['field_width_m']*s['field_length_m']}
    config['scene']['field'] = BASESIM_FIELD.copy()
    return config


def _pipeline_config(s, counts):
    return {'budget_bytes': 800_000_000_000, 'reserve_bytes': 100_000_000_000,
        'scene': {'species': s['species'], 'patch_m': [s['field_width_m'], s['field_length_m']],
            'plant_count': counts, 'layout': s['layout'],
            'min_center_spacing_m': s['spacing_m'], 'scale': [s['scale_min'], s['scale_max']],
            'anisotropic': False, 'geometry_augmentation': 'transform_only',
            'capture_validation': 'structural_only'},
        'capture': {'width': s['image_width'], 'height': s['image_height'], 'views': s['views'],
            'target_height_m': s['target_height_m'], 'camera_height_m': s['camera_height_m']},
        'preparation': {'max_width': s['image_width'], 'max_points': min(30000, s['max_gaussians'])},
        'reconstruction_recipe': 'rgb_known_pose',
        'training': {'steps': s['steps'], 'init_points': min(30000, s['max_gaussians']),
            'max_gaussians': s['max_gaussians']},
        'publication': {'quality_policy': 'record_only'}}


def saved_job_settings(job):
    """Resume earlier runs exactly; new runs always derive counts from field area."""
    value = job['settings']
    legacy = job.get('schema') == 'cropvolume-generation-v1' and 'plants_min' in value and 'plants_max' in value
    if legacy:
        counts = [value['plants_min'], value['plants_max']]
        if any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 10000 for n in counts) or counts[0] > counts[1]:
            raise ValueError('Invalid saved plant count range')
        settings = validate_settings({k: v for k, v in value.items() if k not in ('plants_min', 'plants_max')},
                                     derive_population=False)
        config = _pipeline_config(settings, counts)
    elif job.get('schema') in ('cropvolume-generation-v2','cropvolume-generation-v3'):
        settings = validate_settings(value, derive_population=False)
        config = _pipeline_config(settings, plant_count_range(settings['field_width_m'], settings['field_length_m']))
        config['scene']['population_policy'] = {**POPULATION_POLICY,
            'area_m2':settings['field_width_m']*settings['field_length_m']}
        if job['schema'] == 'cropvolume-generation-v3':
            config['scene']['field'] = LEGACY_BASESIM_FIELD.copy()
    else:
        settings = validate_settings(value)
        config = pipeline_config(settings)
    if job['config'] != config:
        raise ValueError('Saved generation controls/config disagree')
    return settings


class GenerationManager:
    def __init__(self, parent=None, catalog=None, runtime_root=None,
                 engine_root=None, launcher=None, preferences=None, output_parents=None):
        config = configuration()
        self.parent = local_path(parent or config['output_root'])
        self.output_parents = list(dict.fromkeys([self.parent, *[local_path(p) for p in (output_parents or [])]]))
        self.preferences = preferences
        self.catalog = Path(catalog or ROOT / 'assets/plants/catalog.json').resolve()
        self.runtime_root = local_path(runtime_root or config['runtime_root'])
        self.engine_root = engine_root if engine_root is not None else config['engine_root']
        self.launcher = Path(launcher or ROOT / 'tools/run_training.py').resolve()
        self.token = secrets.token_urlsafe(32)
        self.children = {}
        self._memory_estimator = None

    def preflight(self):
        from .geometry import load_catalog
        catalog = load_catalog(self.catalog)
        if not catalog['assets'] or any(a.get('test_fixture') or not a.get('unreal_asset') for a in catalog['assets']):
            raise ValueError('Generation requires the real approved Unreal plant catalog')
        source = ROOT / 'simulator/FlightSchool/Content'
        if not (source / 'FlightSchool/Maps/BasilField.umap').is_file():
            raise ValueError('Bundled BaseSim field map is missing; extract the complete release including simulator assets')
        if not self.launcher.is_file():
            raise ValueError('GPU launcher is missing')
        config = configuration()
        if not config['training_python'].is_file():
            raise ValueError('Training Python is missing; configure CROPVOLUME_TRAINING_PYTHON')
        if not self.engine_root or not (local_path(self.engine_root)/'Engine/Binaries/Win64/UnrealEditor-Cmd.exe').is_file():
            raise ValueError('Configure the Unreal installation with tools/configure.py --engine-root or CROPVOLUME_ENGINE_ROOT')
        if os.name != 'nt' and not str(self.runtime_root).startswith('/mnt/'):
            raise ValueError('Windows Unreal needs a runtime folder on a Windows drive; configure --runtime-root /mnt/c/... (output may be anywhere)')

    def options(self):
        try:
            self.preflight()
            unavailable = None
        except (OSError, ValueError, KeyError) as error:
            unavailable = str(error)
        try:
            species = sorted({a['species'] for a in read_json(self.catalog)['assets']})
        except (OSError, ValueError, KeyError):
            species = [DEFAULTS['species']]
        return {'defaults': {**DEFAULTS, 'layout':'rows'}, 'output_parent': str(self.parent), 'output_parents':[str(p) for p in self.output_parents], 'available': unavailable is None,
                'unavailable_reason': unavailable, 'token': self.token, 'quality_policy': 'record_only',
                'population_policy': POPULATION_POLICY, 'memory_estimation': True,
                'catalog_species': species + (['mixed'] if len(species)>1 else []),
                'run_controls':['stop-now','clear-temporary','delete'],
                'render_profile':'BaseSim field beds and clumps; blank outside the selected field'}

    def estimate(self, settings, counts=None):
        from .memory import MemoryEstimator
        settings = validate_settings(settings, derive_population=counts is None)
        if self._memory_estimator is None:
            self._memory_estimator = MemoryEstimator(self.catalog)
        return self._memory_estimator.estimate(settings, counts)

    def _root(self, run_id):
        if not isinstance(run_id, str) or not RUN_NAME.fullmatch(run_id):
            raise ValueError('Unknown generation run')
        matches = [parent/run_id for parent in self.output_parents if (parent/run_id).exists() or (parent/run_id).is_symlink()]
        if len(matches)>1:raise ValueError('Run ID appears in more than one output folder')
        path = matches[0] if matches else self.parent/run_id
        if path.is_symlink() or path.resolve()!=path or path.parent not in self.output_parents:
            raise ValueError('Invalid generation path')
        return path

    def _managed_run(self,run_id):
        root=self._root(run_id)
        for name in ('samples','locks','generation_status.json','generation_process.json','reservations.json'):
            if (root/name).is_symlink():raise ValueError('Managed run paths must not be symlinks')
        if not (root/'generation.json').is_file() or (root/'generation.json').is_symlink():
            raise ValueError('Unknown managed generation run')
        job=read_json(root/'generation.json')
        if job.get('schema') not in ('cropvolume-generation-v1','cropvolume-generation-v2','cropvolume-generation-v3','cropvolume-generation-v4'):
            raise ValueError('Not a managed generation run')
        return root,job

    def runs(self):
        for key, process in list(self.children.items()):
            if process.poll() is not None:
                del self.children[key]
        result = []
        for path in (path for parent in self.output_parents for path in parent.glob('random-field-*/generation.json')):
            try:
                if path.parent.is_symlink():continue
                job = read_json(path)
                state = read_json(path.parent / 'generation_status.json')
                if state.get('state') in ('starting', 'running', 'stopping'):
                    try:
                        os.kill(state['pid'], 0)
                        if state.get('process_identity') and _process_identity(state['pid']) != state['process_identity']:
                            state = {**state, 'state': 'interrupted'}
                    except (ProcessLookupError, KeyError):
                        state = {**state, 'state': 'interrupted'}
                # Return immutable controls along with actual pair progress.
                result.append({'id': path.parent.name, 'path': str(path.parent), 'settings': job['settings'],
                    'plant_count': job['config']['scene']['plant_count'],
                    'resource_estimate': job.get('resource_estimate'),
                    'state': state.get('state', 'interrupted'), 'error': state.get('error'),
                    'requested': job['settings']['count'],
                    'complete': len(list((path.parent / 'samples').glob('*/COMPLETE.json'))),
                    'stop_requested': (path.parent / 'STOP_REQUESTED').exists(),
                    'memory_released':state.get('memory_released',False),
                    'temporary_files_cleared':state.get('temporary_files_cleared',False),
                    'created_unix': job['created_unix'], 'updated_unix': state.get('updated_unix'),
                    'report': state.get('report'), 'log_url': '/api/generation/' + path.parent.name + '/log'})
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(result, key=lambda r: r['created_unix'], reverse=True)

    def select_output_parent(self, value):
        import tempfile
        path = local_path(value, absolute=True)
        if path == ROOT or any(path.is_relative_to(ROOT/name) for name in
                ('src','tools','tests','configs','dashboard','simulator','assets','third_party','pipeline','docs')):
            raise ValueError('Choose a dataset output folder, not a source or asset folder')
        path.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryFile(dir=path):
            pass
        parents = list(dict.fromkeys([*self.output_parents,path]))
        if self.preferences:
            save_preferences({'output_root':str(path),'output_roots':[str(p) for p in parents]},self.preferences)
        self.parent = path
        self.output_parents = parents
        return path

    def start(self, settings, output_parent=None):
        settings = validate_settings(settings)
        self.preflight()
        species = {a['species'] for a in read_json(self.catalog)['assets']}
        if settings['species'] != 'mixed' and settings['species'] not in species:
            raise ValueError('Selected plant species is not available in the configured field catalog')
        # Estimates are advisory and never change the requested generation settings.
        resource_estimate = self.estimate(settings)
        with lease(self.runtime_root / 'locks/dashboard_generation_launch'):
            self._ensure_idle()
            if output_parent is not None:
                self.select_output_parent(output_parent)
                self._ensure_idle()
            stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
            root = self.parent / f"random-field-{settings['name']}-{stamp}-{uuid.uuid4().hex[:6]}"
            (root / 'samples').mkdir(parents=True)
            atomic_json(root / 'generation.json', {'schema': 'cropvolume-generation-v4', 'settings': settings,
                'config': pipeline_config(settings), 'created_unix': time.time(), 'catalog': str(self.catalog),
                'runtime_root': str(self.runtime_root), 'engine_root': self.engine_root,
                'resource_estimate': resource_estimate})
            self._spawn(root)
        return root.name

    def _ensure_idle(self):
        if any(r['state'] in ('starting', 'running', 'stopping') for r in self.runs()):
            raise RuntimeError('A generation run is already active. Stop it after the current pair before starting another.')

    def _spawn(self, root):
        (root / 'STOP_REQUESTED').unlink(missing_ok=True)
        # The detached worker survives closing the browser or restarting the dashboard.
        atomic_json(root / 'generation_status.json', {'state': 'starting', 'pid': os.getpid(),
            'process_identity': _process_identity(os.getpid()), 'updated_unix': time.time()})
        try:
            with (root / 'generation.log').open('ab') as log:
                process = subprocess.Popen([sys.executable, str(self.launcher), '--script',
                    str(ROOT / 'tools/generate_dataset.py'), '--job', str(root / 'generation.json')],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            self.children[root.name] = process
            atomic_json(root/'generation_process.json',{'pid':process.pid,
                'process_identity':_process_identity(process.pid),'process_group':process.pid,'started_unix':time.time()})
            atomic_json(root / 'generation_status.json', {'state': 'starting', 'pid': process.pid,
                'process_identity': _process_identity(process.pid), 'updated_unix': time.time()})
        except OSError as error:
            atomic_json(root / 'generation_status.json', {'state': 'failed', 'error': str(error), 'updated_unix': time.time()})
            raise

    def stop(self, run_id):
        with lease(self.runtime_root/'locks/dashboard_generation_launch'):
            root,_=self._managed_run(run_id)
            atomic_json(root / 'STOP_REQUESTED', {'requested_unix': time.time()})
        return run_id

    def _stop_now(self,root,job):
        from .run_control import stop_processes,release_run_bookkeeping
        path=root/'generation_status.json'
        state=read_json(path) if path.exists() else {}
        original_state=state.get('state')
        atomic_json(root/'STOP_REQUESTED',{'requested_unix':time.time(),'stop_mode':'immediate','requested_by':'dashboard'})
        atomic_json(path,{**state,'state':'stopping','updated_unix':time.time()})
        try:
            stopped=stop_processes(root,state)
            release_run_bookkeeping(root,Path(job.get('runtime_root',self.runtime_root)),stopped)
        except Exception as error:
            atomic_json(path,{**state,'state':original_state or 'interrupted','control_error':str(error),'updated_unix':time.time()})
            raise
        for worker_path in (root/'samples').glob('*/worker.json'):
            if worker_path.parent.is_symlink() or worker_path.is_symlink():continue
            worker=read_json(worker_path)
            if worker.get('state') in ('preparing','training'):
                atomic_json(worker_path,{**worker,'state':'interrupted','interrupted_by_user':True,
                    'error':'Stopped immediately from the dashboard; files preserved.'})
        # A completed dataset stays complete when a memory-release check is requested.
        state.update(state='complete' if original_state=='complete' else 'stopped',
            stopped_by_user=True,memory_released=True,stop_mode='immediate',
            updated_unix=time.time(),process_stop=stopped)
        state.pop('error',None);state.pop('control_error',None)
        atomic_json(path,state)
        if self._memory_estimator:
            with self._memory_estimator.hardware.lock:self._memory_estimator.hardware.updated=0
        return stopped

    def stop_now(self,run_id):
        with lease(self.runtime_root/'locks/dashboard_generation_launch'):
            root,job=self._managed_run(run_id)
            self._stop_now(root,job)
        return run_id

    def cleanup_preview(self,run_id,scope='temporary'):
        from .run_control import temporary_targets,path_usage,profile_target
        if scope not in ('temporary','run'):raise ValueError('Unknown cleanup scope')
        root,_=self._managed_run(run_id)
        targets=temporary_targets(root) if scope=='temporary' else [root,profile_target(root)]
        usages=[path_usage(path) for path in targets]
        return {'run_id':run_id,'scope':scope,'files':sum(v['files'] for v in usages),
            'bytes':sum(v['bytes'] for v in usages),
            'published_pairs':len(list((root/'samples').glob('*/COMPLETE.json'))),
            'keeps':'Photos, labels, published pairs, metadata, logs and timing records' if scope=='temporary' else None}

    def clear_temporary(self,run_id,confirmation):
        if confirmation!=run_id:raise ValueError('Confirm the run ID before clearing temporary files')
        from .run_control import temporary_targets,remove_path,reset_after_cleanup
        with lease(self.runtime_root/'locks/dashboard_generation_launch'):
            root,job=self._managed_run(run_id)
            self._stop_now(root,job)
            for path in temporary_targets(root):remove_path(path)
            reset_after_cleanup(root)
            status=read_json(root/'generation_status.json')
            atomic_json(root/'generation_status.json',{**status,'temporary_files_cleared':True,'updated_unix':time.time()})
        return run_id

    def delete(self,run_id,confirmation):
        if confirmation!=run_id:raise ValueError('Confirm the run ID before deleting the run')
        from .run_control import remove_path,profile_target
        with lease(self.runtime_root/'locks/dashboard_generation_launch'):
            root,job=self._managed_run(run_id)
            profiles=profile_target(root)
            self._stop_now(root,job)
            remove_path(root)
            remove_path(profiles)
        return run_id

    def resume(self, run_id):
        self.preflight()
        with lease(self.runtime_root / 'locks/dashboard_generation_launch'):
            root,job=self._managed_run(run_id)
            saved_job_settings(job)
            self._ensure_idle()
            self._spawn(root)
        return run_id


def execute_job(path):
    from .batch import run_batch
    from .storage import StoragePaused
    path = Path(path).resolve()
    root, job = path.parent, read_json(path)
    settings = saved_job_settings(job)
    status_path = root / 'generation_status.json'
    state = {'state': 'running', 'pid': os.getpid(), 'process_identity': _process_identity(os.getpid()),
             'updated_unix': time.time()}
    try:
        # Different dashboard processes and roots share the existing local renderer/GPU.
        with lease(Path(job['runtime_root']) / 'locks/dashboard_generation_worker'):
            atomic_json(status_path, state)
            with_stop = lambda: (root / 'STOP_REQUESTED').exists()
            report = run_batch(root, job['catalog'], job['config'], settings['count'], settings['seed'],
                job['runtime_root'], job['engine_root'], train=True, stop_requested=with_stop,
                peak_bytes=max(2_000_000_000, settings['image_width'] * settings['image_height'] * settings['views'] * 12))
            status = ('paused' if report['paused'] else 'stopped' if report.get('stopped') else
                      'complete' if report['complete'] == settings['count'] else 'finished_with_errors')
            state.update(state=status, report=report)
    except StoragePaused as error:
        state.update(state='paused', error=str(error))
    except BaseException as error:
        state.update(state='failed', error=str(error))
        raise
    finally:
        state['updated_unix'] = time.time()
        atomic_json(status_path, state)
    return state
