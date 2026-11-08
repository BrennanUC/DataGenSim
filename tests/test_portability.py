import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import pytest

from cropvolume.configuration import PROJECT_ROOT,configuration,local_path,save_preferences
from cropvolume.generation import GenerationManager
from cropvolume.geometry import load_catalog
from cropvolume.io import atomic_json,read_json


def test_configuration_is_relocatable_and_environment_overrides_local_preferences(tmp_path,monkeypatch):
    for key in ('CROPVOLUME_OUTPUT_ROOT','CROPVOLUME_RUNTIME_ROOT','CROPVOLUME_TRAINING_PYTHON','CROPVOLUME_ROCM_PYTHON'):
        monkeypatch.delenv(key,raising=False)
    assert configuration()['output_root']==PROJECT_ROOT/'data'
    save_preferences({'output_root':str(tmp_path/'chosen')})
    assert configuration()['output_root']==tmp_path/'chosen'
    monkeypatch.setenv('CROPVOLUME_OUTPUT_ROOT',str(tmp_path/'override'))
    assert configuration()['output_root']==tmp_path/'override'
    assert configuration()['runtime_root']==tmp_path/'override'


def test_output_paths_accept_windows_drives_under_wsl_and_reject_relative_paths():
    import os
    if os.name!='nt':assert local_path(r'E:\Portable data\pairs')==Path('/mnt/e/Portable data/pairs')
    with pytest.raises(ValueError,match='absolute'):local_path('relative/data',absolute=True)


def test_bundled_assets_are_complete_without_original_projects_or_external_catalog():
    catalog=load_catalog(PROJECT_ROOT/'assets/plants/catalog.json')
    assert len(catalog['assets'])==4
    for asset in catalog['assets']:
        path=PROJECT_ROOT/'simulator/FlightSchool/Content'/Path(asset['unreal_asset'].removeprefix('/Game/')).with_suffix('.uasset')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==asset['unreal_uasset_sha256']
    proof=read_json(PROJECT_ROOT/'configs/basesim_row_anchors.json')
    source=PROJECT_ROOT/'simulator/FlightSchool/Content/FlightSchool/Maps/BasilField.umap'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==proof['source_map_sha256']


def test_output_selection_persists_and_retains_controls_for_previous_roots(tmp_path,monkeypatch,catalog):
    parent=tmp_path/'first';preferences=tmp_path/'preferences.json'
    manager=GenerationManager(parent,catalog=catalog,preferences=preferences)
    monkeypatch.setattr(manager,'preflight',lambda:None)
    monkeypatch.setattr(manager,'estimate',lambda value:{})
    def spawn(root):atomic_json(root/'generation_status.json',{'state':'complete'})
    monkeypatch.setattr(manager,'_spawn',spawn)
    first=manager.start({'count':1})
    second_parent=tmp_path/'second folder'
    second=manager.start({'count':1},output_parent=str(second_parent))
    assert manager._root(first)==parent/first and manager._root(second)==second_parent/second
    assert {r['id'] for r in manager.runs()}=={first,second}
    assert read_json(preferences)['output_root']==str(second_parent)
    reloaded=GenerationManager(second_parent,output_parents=read_json(preferences)['output_roots'])
    assert reloaded._root(first)==parent/first
    reloaded.delete(first,first)
    assert not (parent/first).exists() and (second_parent/second).exists()
    with pytest.raises(ValueError):manager.select_output_parent(str(PROJECT_ROOT/'src/generated'))


def test_duplicate_run_ids_across_output_roots_are_rejected(tmp_path):
    for name in ('one','two'):(tmp_path/name/'random-field-duplicate').mkdir(parents=True)
    manager=GenerationManager(tmp_path/'one',output_parents=[tmp_path/'two'])
    with pytest.raises(ValueError,match='more than one'):manager._root('random-field-duplicate')


def test_release_builder_excludes_runs_config_caches_and_git(tmp_path):
    spec=importlib.util.spec_from_file_location('package_release',PROJECT_ROOT/'tools/package_release.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'source';source.mkdir()
    (source/'pyproject.toml').write_text('[project]\nversion="1.0"\n')
    for name in ['README.md','run.py','src/pkg/__init__.py','dashboard/dist/app.js','assets/mesh.ply',
                 'src/pkg/__pycache__/x.pyc','data/run/photo.png','.git/config',
                 'cropvolume.local.json','.venv/bin/python','reports/private.json']:
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
    output=tmp_path/'release.tar.gz'
    result=module.build_release(source,output)
    with tarfile.open(output) as archive:
        names=set(archive.getnames())
        assert 'cropvolume-1.0/run.py' in names
        assert 'cropvolume-1.0/dashboard/dist/app.js' in names
        assert 'cropvolume-1.0/assets/mesh.ply' in names
        assert not any('/data/' in n or '/.git/' in n or '__pycache__' in n or 'local.json' in n for n in names)
        manifest=json.load(archive.extractfile('cropvolume-1.0/release-manifest.json'))
        for name,digest in manifest['files'].items():
            assert hashlib.sha256(archive.extractfile('cropvolume-1.0/'+name).read()).hexdigest()==digest
    assert result['sha256']==hashlib.sha256(output.read_bytes()).hexdigest()


def test_training_interpreter_keeps_virtual_environment_symlink(tmp_path):
    import sys
    interpreter=tmp_path/'python'
    interpreter.symlink_to(Path(sys.executable).resolve())
    save_preferences({'training_python':str(interpreter)})
    assert configuration()['training_python']==interpreter


def test_dashboard_cli_uses_bundled_catalog_independent_of_output(tmp_path,monkeypatch):
    import sys
    spec=importlib.util.spec_from_file_location('dashboard_entry',PROJECT_ROOT/'tools/dashboard.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    called=[]
    monkeypatch.setenv('CROPVOLUME_OUTPUT_ROOT',str(tmp_path/'new-output'))
    monkeypatch.setattr(sys,'argv',['dashboard.py','--port','0'])
    monkeypatch.setattr(module,'serve',lambda roots,artifacts,port,generation:called.append(generation))
    module.main()
    assert called[0].catalog==PROJECT_ROOT/'assets/plants/catalog.json'
    assert called[0].parent==tmp_path/'new-output'
