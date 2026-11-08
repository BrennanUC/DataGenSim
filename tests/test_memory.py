import json
from pathlib import Path

import pytest

from cropvolume.generation import DEFAULTS, GenerationManager
from cropvolume.io import read_json
from cropvolume.memory import GIB, HardwareMemory, MemoryEstimator, estimate_memory, linux_memory, ply_counts


MESHES = [{'species':'basil','vertices':7000,'faces':14000},
          {'species':'cilantro','vertices':32000,'faces':64000}]


def test_memory_models_follow_resolution_gaussians_and_geometry():
    base = estimate_memory(DEFAULTS,MESHES)
    pixels = estimate_memory({**DEFAULTS,'image_width':2048,'image_height':1536},MESHES)
    splats = estimate_memory({**DEFAULTS,'max_gaussians':500000},MESHES)
    field = estimate_memory({**DEFAULTS,'field_width_m':20,'field_length_m':20},MESHES)
    complex_mesh = estimate_memory({**DEFAULTS,'species':'cilantro'},MESHES)
    assert pixels['gpu_memory']['high_bytes'] > base['gpu_memory']['high_bytes']
    assert splats['gpu_memory']['high_bytes'] > base['gpu_memory']['high_bytes']
    assert field['volume_overlap_worst_case_bytes'] > base['volume_overlap_worst_case_bytes']
    assert complex_mesh['volume_overlap_worst_case_bytes'] > base['volume_overlap_worst_case_bytes']
    assert field['assumptions']['overlap_group_plants'] == 64
    assert all(row['ram']['high_bytes'] >= row['ram']['low_bytes'] > 0 for row in base['stages'])


def test_sequential_batch_count_does_not_multiply_peak_memory():
    first=estimate_memory({**DEFAULTS,'count':1},MESHES)
    many=estimate_memory({**DEFAULTS,'count':1000},MESHES)
    for field in ('worker_ram','unreal_ram','gpu_memory','volume_overlap_worst_case_bytes'):
        assert first[field]==many[field]


def test_capacity_warnings_and_unknown_memory_are_explicit():
    hardware={'worker_ram':{'total_bytes':GIB,'available_bytes':GIB},
        'host_ram':{'total_bytes':2*GIB,'available_bytes':GIB},
        'gpu':{'total_bytes':GIB,'available_bytes':None}}
    low=estimate_memory(DEFAULTS,MESHES,hardware)
    assert {w['resource'] for w in low['warnings']} == {'WSL/worker RAM','Unreal host RAM','GPU VRAM'}
    unknown=estimate_memory(DEFAULTS,MESHES)
    assert unknown['hardware']=={} and unknown['warnings']==[]


def test_memory_probe_respects_container_limits_and_excludes_swap(tmp_path):
    proc=tmp_path/'proc';(proc/'self').mkdir(parents=True)
    (proc/'meminfo').write_text('MemTotal: 16777216 kB\nMemAvailable: 8388608 kB\nSwapTotal: 33554432 kB\n')
    (proc/'self/cgroup').write_text('0::/scope\n')
    cg=tmp_path/'cgroup';(cg/'scope').mkdir(parents=True)
    (cg/'scope/memory.max').write_text(str(4*GIB));(cg/'scope/memory.current').write_text(str(GIB))
    value=linux_memory(proc,cg)
    assert value['total_bytes']==4*GIB and value['available_bytes']==3*GIB and value['swap_bytes']==32*GIB
    assert linux_memory(tmp_path/'missing')['available_bytes'] is None


def test_estimator_reads_mesh_headers_without_loading_geometry(catalog,monkeypatch):
    import cropvolume.geometry
    estimator=MemoryEstimator(catalog)
    monkeypatch.setattr(estimator.hardware,'snapshot',lambda:{})
    monkeypatch.setattr(cropvolume.geometry,'load_mesh',lambda *a,**k:pytest.fail('Estimate must not load meshes'))
    value=estimator.estimate(DEFAULTS)
    asset=read_json(catalog)['assets'][0]
    counts=ply_counts(catalog.parent/asset['mesh'])
    assert counts['faces']==12 and counts['vertices']==8
    assert value['plant_count']==[20,40]
    with pytest.raises(ValueError):estimator.estimate({'max_gaussians':0})


def test_hardware_probe_uses_64bit_gpu_capacity(monkeypatch):
    import cropvolume.memory as memory
    from types import SimpleNamespace
    monkeypatch.setattr(memory.shutil,'which',lambda name:'powershell.exe')
    data={'host_ram':{'total_bytes':32*GIB,'available_bytes':9*GIB},
        'gpu':{'name':'test GPU','total_bytes':20*GIB,'available_bytes':None}}
    monkeypatch.setattr(memory.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=json.dumps(data)))
    probe=HardwareMemory();probe._probe()
    assert probe.cached['gpu']['total_bytes']==20*GIB and probe.cached['gpu']['available_bytes'] is None


def test_saved_run_contains_advisory_estimate_without_changing_settings(tmp_path,monkeypatch,catalog):
    manager=GenerationManager(tmp_path,catalog=catalog)
    monkeypatch.setattr(manager,'preflight',lambda:None)
    estimate=estimate_memory(DEFAULTS,MESHES)
    monkeypatch.setattr(manager,'estimate',lambda settings:estimate)
    monkeypatch.setattr(manager,'_spawn',lambda root:None)
    run=manager.start(DEFAULTS)
    saved=read_json(tmp_path/run/'generation.json')
    assert saved['resource_estimate']==estimate and saved['settings']==DEFAULTS
