"""Machine-local paths; source checkouts and release tarballs are relocatable."""
import json
import os
from pathlib import Path, PureWindowsPath
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_KEYS = {'output_root':'CROPVOLUME_OUTPUT_ROOT','runtime_root':'CROPVOLUME_RUNTIME_ROOT',
    'engine_root':'CROPVOLUME_ENGINE_ROOT','training_python':'CROPVOLUME_TRAINING_PYTHON',
    'backend':'CROPVOLUME_GPU_BACKEND','gsplat_source':'CROPVOLUME_GSPLAT_SOURCE',
    'cache_root':'CROPVOLUME_CACHE_ROOT'}


def local_config_path():
    return Path(os.environ.get('CROPVOLUME_CONFIG', PROJECT_ROOT/'cropvolume.local.json')).expanduser().resolve()


def local_path(value, *, absolute=False, resolve_links=True):
    if not isinstance(value,(str,Path)) or not str(value).strip() or '\x00' in str(value):
        raise ValueError('Choose a valid output folder')
    value = str(value).strip()
    win = PureWindowsPath(value)
    if os.name != 'nt' and len(win.drive)==2 and win.drive[1]==':':
        value = '/mnt/'+win.drive[0].lower()+'/'+'/'.join(win.parts[1:])
    path = Path(value).expanduser()
    if absolute and not path.is_absolute():
        raise ValueError('Output folder must be an absolute path (or start with ~)')
    path = path if path.is_absolute() else PROJECT_ROOT/path
    return path.resolve() if resolve_links else path.absolute()


def load_preferences(path=None):
    path = Path(path) if path else local_config_path()
    if not path.exists():return {}
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value,dict):raise ValueError('Local configuration must be a JSON object')
    return value


def configuration():
    defaults = {'output_root':str(PROJECT_ROOT/'data'), 'runtime_root':str(PROJECT_ROOT/'data'),
        'engine_root':'', 'training_python':sys.executable, 'backend':'rocm',
        'gsplat_source':str(PROJECT_ROOT/'third_party/gsplat-rocm'),
        'cache_root':str(Path.home()/'.cache/cropvolume'), 'output_roots':[]}
    saved = load_preferences()
    result = {**defaults,**saved}
    for key,env in ENV_KEYS.items():
        if os.environ.get(env):result[key]=os.environ[env]
    # Earlier host configurations remain accepted without checked-in host paths.
    if os.environ.get('CROPVOLUME_ROCM_PYTHON') and not os.environ.get('CROPVOLUME_TRAINING_PYTHON'):
        result['training_python']=os.environ['CROPVOLUME_ROCM_PYTHON']
    if 'runtime_root' not in saved and not os.environ.get('CROPVOLUME_RUNTIME_ROOT'):
        result['runtime_root']=result['output_root']
    if result['backend'] not in ('rocm','cuda'):
        raise ValueError('GPU backend must be rocm or cuda')
    for key in ('output_root','runtime_root','training_python','gsplat_source','cache_root'):
        result[key]=local_path(result[key],resolve_links=key!='training_python')
    return result


def save_preferences(changes, path=None):
    from .io import atomic_json
    path = Path(path) if path else local_config_path()
    value = {**load_preferences(path), **changes}
    atomic_json(path, value)
