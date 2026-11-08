from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from cropvolume.cameras import capture_plan
from cropvolume.dashboard import Monitor, make_handler
from cropvolume.dataset import publish, rebuild_index
from cropvolume.generation import DEFAULTS, GenerationManager, execute_job, pipeline_config, validate_settings, plant_count_range, saved_job_settings
from cropvolume.geometry import measure_scene
from cropvolume.io import atomic_json, read_json, identity
from cropvolume.prepare import seal_capture
from cropvolume.scenes import generate
from test_lifecycle import fixture_splat


@pytest.mark.parametrize('value', [
    {'count': 0}, {'count': 1.5}, {'max_gaussians': 3}, {'image_height': 0},
    {'field_width_m': float('nan')}, {'camera_height_m': .1, 'target_height_m': .2},
    {'plants_min': 5, 'plants_max': 2}, {'name': '../escape'}, {'command': 'arbitrary'},
])
def test_generation_controls_reject_invalid_or_unknown_values(value):
    with pytest.raises(ValueError):
        validate_settings(value)


def test_requested_controls_reach_capture_training_and_publication():
    settings = validate_settings({'field_width_m': 3., 'field_length_m': 5., 'image_width': 640,
        'image_height': 360, 'camera_height_m': 4., 'max_gaussians': 1000})
    config = pipeline_config(settings)
    frames = capture_plan({'scene_id':'example','patch_m':config['scene']['patch_m']}, **config['capture'])['frames']
    assert all((f['width'],f['height']) == (640,360) and f['c2w_cv'][2][3] == 4 for f in frames)
    assert config['training']['max_gaussians'] == config['training']['init_points'] == 1000
    assert config['preparation']['max_width'] == 640
    assert config['publication'] == {'quality_policy':'record_only'}


def test_random_fields_are_reproducible_with_matching_geometry_volume(catalog):
    config = pipeline_config({'field_width_m':.8,'field_length_m':.8})['scene']
    scene = generate(catalog, 43, config)
    assert scene == generate(catalog,43,config)
    other = generate(catalog,44,config)
    assert other['scene_id'] != scene['scene_id'] and other['instances'] != scene['instances']
    label = measure_scene(scene,catalog)
    assert label['scene_id'] == scene['scene_id'] and len(label['instances']) == len(scene['instances'])
    assert 3 <= len(scene['instances']) <= 6
    assert 0 < label['plant_solid_volume_m3'] <= label['sum_instance_volume_m3'] * 1.00001


@pytest.mark.parametrize('width,length,expected',[(2,2,[20,40]),(1,1,[5,10]),(2,4,[40,80]),
    (.8,.8,[3,6]),(.1,.1,[1,1])])
def test_population_scales_with_field_area(width,length,expected):
    config=pipeline_config({'field_width_m':width,'field_length_m':length})
    assert plant_count_range(width,length)==plant_count_range(length,width)==expected
    assert config['scene']['plant_count']==expected
    assert config['scene']['population_policy']['area_m2']==width*length
    assert 'plants_min' not in DEFAULTS and 'plants_max' not in DEFAULTS


def test_new_runs_cannot_override_automatic_population_or_exceed_capacity():
    for settings in [{'plants_min':1,'plants_max':2},{'field_width_m':100,'field_length_m':100}]:
        with pytest.raises(ValueError):
            pipeline_config(settings)


def test_earlier_saved_runs_keep_their_original_population_on_resume(tmp_path,monkeypatch):
    import cropvolume.batch
    config=pipeline_config({})
    config['scene'].pop('population_policy')
    config['scene'].pop('field')
    config['scene']['plant_count']=[2,3]
    job={'schema':'cropvolume-generation-v1','settings':{**DEFAULTS,'plants_min':2,'plants_max':3},
         'config':config,'catalog':'catalog','runtime_root':str(tmp_path/'runtime'),'engine_root':'engine'}
    assert saved_job_settings(job)==DEFAULTS
    called=[]
    def batch(*args,**kwargs):
        called.append(args[2]['scene']['plant_count'])
        return {'complete':DEFAULTS['count'],'failed':0,'paused':False}
    monkeypatch.setattr(cropvolume.batch,'run_batch',batch)
    path=tmp_path/'generation.json';atomic_json(path,job)
    assert execute_job(path)['state']=='complete' and called==[[2,3]]
    job['config']['scene']['plant_count']=[5,6]
    with pytest.raises(ValueError,match='disagree'):
        saved_job_settings(job)


def test_low_reconstruction_scores_publish_and_metadata_is_bound_to_pair(sample):
    fixture_splat(sample)
    root = sample.parents[1]
    config = pipeline_config({})
    atomic_json(root/'dataset.json', {'config':config})
    metrics = read_json(sample/'metrics.json')
    metrics['test'] = {'mean_psnr_db':1.,'mean_ssim':-.2}
    atomic_json(sample/'metrics.json',metrics)
    row = publish(root,sample.name,allow_fixture=True,quality_policy='record_only')
    assert row['quality_policy'] == 'record_only' and row['quality_thresholds'] is None
    metadata = read_json(root/row['metadata_path'])
    assert metadata['metrics']['test'] == metrics['test']
    assert metadata['seed'] == read_json(sample/'scene.json')['seed']
    assert metadata['training_settings']['max_gaussians'] == DEFAULTS['max_gaussians']
    assert row == publish(root,sample.name,allow_fixture=True,quality_policy='record_only')
    metadata['seed'] = -1
    atomic_json(root/row['metadata_path'],metadata)
    with pytest.raises(ValueError,match='metadata was modified'):
        rebuild_index(root,allow_fixture=True)


def test_record_only_keeps_structural_checks(sample):
    fixture_splat(sample)
    atomic_json(sample.parents[1]/'dataset.json',{'config':pipeline_config({})})
    (sample/'splat.ply').write_bytes(b'broken file')
    with pytest.raises(Exception):
        publish(sample.parents[1],sample.name,allow_fixture=True,quality_policy='record_only')
    assert not (sample/'COMPLETE.json').exists()


@pytest.mark.parametrize('structural_only',[False,True])
def test_uniform_images_are_not_a_quality_gate_for_random_field_runs(sample,structural_only):
    # The explicit analytic fixture has uniform RGB. Change its scene consistently
    # to test validation policy; it remains a fixture and cannot enter production.
    scene=read_json(sample/'scene.json')
    scene.pop('scene_id')
    if structural_only:
        scene['resolved_config']['capture_validation']='structural_only'
    scene['scene_id']=identity(scene)
    atomic_json(sample/'scene.json',scene)
    for name in ['capture.json','labels/volume.json']:
        data=read_json(sample/name);data['scene_id']=scene['scene_id'];atomic_json(sample/name,data)
    atomic_json(sample/'RENDER_FINISHED.json',{'scene_id':scene['scene_id']})
    (sample/'CAPTURE_COMPLETE.json').unlink()
    if structural_only:
        assert seal_capture(sample)['scene_id']==scene['scene_id']
    else:
        with pytest.raises(ValueError,match='nearly uniform'):
            seal_capture(sample)


def test_saved_run_executes_real_batch_contract_and_honors_stop(tmp_path,monkeypatch):
    import cropvolume.batch
    config=pipeline_config({'count':2})
    job=tmp_path/'run/generation.json'
    atomic_json(job,{'settings':validate_settings({'count':2}),'config':config,
        'catalog':'real-catalog','runtime_root':str(tmp_path/'runtime'),'engine_root':'engine'})
    called=[]
    def batch(*args,**kwargs):
        called.append((args,kwargs))
        assert args[2]['publication']['quality_policy']=='record_only' and kwargs['train'] is True
        assert not kwargs['stop_requested']()
        (job.parent/'STOP_REQUESTED').touch()
        assert kwargs['stop_requested']()
        return {'paused':False,'complete':1,'failed':0,'stopped':True}
    monkeypatch.setattr(cropvolume.batch,'run_batch',batch)
    assert execute_job(job)['state']=='stopped' and len(called)==1
    assert read_json(job.parent/'generation_status.json')['report']['complete']==1


def test_batch_stops_before_launching_next_pair(catalog,tmp_path,monkeypatch):
    import cropvolume.batch as batch
    def must_not_capture(*args):
        pytest.fail('A stopped batch must not start a capture')
    monkeypatch.setattr(batch,'capture',must_not_capture)
    result=batch.run_batch(tmp_path/'dataset',catalog,{'scene':{'plant_count':[1,1]},'reserve_bytes':0},
        2,23,tmp_path/'runtime','engine',stop_requested=lambda:True)
    assert result['stopped'] and result['complete']==0 and result['samples']==[]


def test_manager_saves_controls_launches_once_and_can_resume(tmp_path,monkeypatch,catalog):
    manager=GenerationManager(tmp_path,catalog=catalog)
    monkeypatch.setattr(manager,'preflight',lambda:None)
    monkeypatch.setattr(manager,'estimate',lambda settings:{'model':'test-estimate','worker_ram':{}})
    launched=[]
    def spawn(root):
        launched.append(root)
        (root/'STOP_REQUESTED').unlink(missing_ok=True)
        atomic_json(root/'generation_status.json',{'state':'complete'})
    monkeypatch.setattr(manager,'_spawn',spawn)
    run_id=manager.start({'count':2,'seed':11})
    assert len(launched)==1
    saved=read_json(tmp_path/run_id/'generation.json')
    assert saved['settings']['count']==2 and saved['settings']['seed']==11
    manager.stop(run_id)
    assert (tmp_path/run_id/'STOP_REQUESTED').exists()
    manager.resume(run_id)
    assert launched==[tmp_path/run_id]*2 and not (tmp_path/run_id/'STOP_REQUESTED').exists()
    with pytest.raises(ValueError):
        manager.stop('../outside')


def test_generation_http_requires_local_session_and_validates_before_launch(tmp_path,monkeypatch):
    manager=GenerationManager(tmp_path)
    calls=[]
    def start(settings):
        validate_settings(settings)
        calls.append(settings)
        return 'random-field-test'
    monkeypatch.setattr(manager,'start',start)
    monkeypatch.setattr(manager,'preflight',lambda:None)
    def estimate(settings):
        validate_settings(settings)
        return {'schema':'cropvolume-memory-estimate-v1','worker_ram':{'low_bytes':1024,'high_bytes':2048}}
    monkeypatch.setattr(manager,'estimate',estimate)
    try:
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(Monitor([tmp_path]),manager))
    except PermissionError:
        pytest.skip('Local sockets restricted')
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}'
    def request(body,token=None,origin=None):
        headers={'Content-Type':'application/json'}
        if token:headers['X-CropVolume-Token']=token
        if origin:headers['Origin']=origin
        return Request(url+'/api/generation',json.dumps(body).encode(),headers)
    try:
        options=json.load(urlopen(url+'/api/generation/options'))
        memory_request=Request(url+'/api/generation/estimate',json.dumps({'settings':{'count':1}}).encode(),
            {'Content-Type':'application/json','X-CropVolume-Token':options['token']})
        with urlopen(memory_request) as response:
            assert json.load(response)['schema']=='cropvolume-memory-estimate-v1'
        assert calls==[] and not list(tmp_path.iterdir())
        for token,origin in [(None,None),(manager.token,'http://external.example')]:
            with pytest.raises(HTTPError) as error:urlopen(request({'settings':{}},token,origin))
            assert error.value.code==403
        with pytest.raises(HTTPError) as error:urlopen(request({'settings':{'count':0}},options['token']))
        assert error.value.code==400 and calls==[]
        response=urlopen(request({'settings':{'count':1}},options['token']))
        assert response.status==202 and json.load(response)['run_id']=='random-field-test' and len(calls)==1
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)
