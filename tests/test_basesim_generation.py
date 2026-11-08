import copy
import numpy as np
import pytest

from cropvolume.cameras import capture_plan, export_cameras
from cropvolume.generation import BASESIM_FIELD, LEGACY_BASESIM_FIELD, DEFAULTS, GenerationManager, pipeline_config, saved_job_settings
from cropvolume.geometry import load_catalog, measure_scene
from cropvolume.io import atomic_json, read_json
from cropvolume.prepare import seal_capture
from cropvolume.scenes import generate
from cropvolume.dataset import publish
from test_lifecycle import fixture_splat


def test_default_dashboard_uses_reference_field_clump_library():
    manager = GenerationManager()
    assert manager.catalog.as_posix().endswith('/assets/plants/catalog.json')
    assert manager.options()['defaults']['layout'] == 'rows'
    config=pipeline_config({})
    assert config['scene']['field']['source_map']=='/Game/FlightSchool/Maps/BasilField'
    assert config['scene']['field']['appearance_policy']=='authored_map_lighting_materials_postprocess'


def test_random_targets_and_cameras_share_original_field_coordinates(catalog):
    config=pipeline_config({'field_width_m':2.,'field_length_m':3.})
    scene=generate(catalog,17,config['scene'])
    assert scene['center_enu_m']==[50.,0.] and scene['field']==BASESIM_FIELD
    assert scene['appearance']['policy']==BASESIM_FIELD['appearance_policy']
    assert 'exposure_ev100' not in scene['appearance']
    bounds=np.asarray(load_catalog(catalog)['assets'][0]['bounds_m'])
    corners=np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
    for instance in scene['instances']:
        matrix=np.asarray(instance['local_to_world_m'])
        points=corners@matrix[:3,:3].T+matrix[:3,3]
        assert points[:,0].min()>=49.-1e-8 and points[:,0].max()<=51.+1e-8
        assert points[:,1].min()>=-1.5-1e-8 and points[:,1].max()<=1.5+1e-8
        assert points[:,2].min()==pytest.approx(.08)
    plan=capture_plan(scene,**config['capture'])
    positions=np.asarray([f['c2w_cv'] for f in plan['frames']])[:,:3,3]
    assert np.allclose(positions.mean(axis=0)[:2],[50.,0.])
    assert np.allclose(positions[:,2],DEFAULTS['camera_height_m'])


def test_preexisting_v2_runs_do_not_silently_switch_render_profiles():
    config=pipeline_config({});config['scene'].pop('field')
    job={'schema':'cropvolume-generation-v2','settings':DEFAULTS,'config':config}
    assert saved_job_settings(job)==DEFAULTS
    changed=copy.deepcopy(job);changed['config']['scene']['field']=BASESIM_FIELD
    with pytest.raises(ValueError,match='disagree'):saved_job_settings(changed)


@pytest.mark.parametrize('valid',[False,True])
def test_basesim_capture_requires_environment_and_placement_proof(sample,catalog,valid):
    config=pipeline_config({})
    scene=generate(catalog,17,config['scene'])
    plan=capture_plan(scene,views=24,width=32,height=24)
    atomic_json(sample/'scene.json',scene)
    atomic_json(sample/'labels/volume.json',measure_scene(scene,catalog))
    atomic_json(sample/'capture.json',plan);export_cameras(sample,plan['frames'])
    (sample/'CAPTURE_COMPLETE.json').unlink()
    proof={'scene_id':scene['scene_id'],'backend':'unreal_basesim_random',
        'source_map':BASESIM_FIELD['source_map'],'appearance_policy':BASESIM_FIELD['appearance_policy'],
        'environment_before_sha256':'fixture','environment_after_sha256':'fixture' if valid else 'changed',
        'all_plant_transforms_verified':True,'maximum_plant_transform_error':0,
        'target_plant_count':len(scene['instances']),
        'context_policy':BASESIM_FIELD['context_policy'], 'exterior':'black',
        'primitive_render_mode':'show_only_components','clipped_ground_components':2,
        'shown_components':len(scene['instances'])+2,'excluded_original_plant_instances':98118,
        'region_center_enu_m':scene['center_enu_m'],'region_size_m':scene['patch_m']}
    atomic_json(sample/'RENDER_FINISHED.json',proof)
    if valid:
        assert seal_capture(sample)['scene_id']==scene['scene_id']
        atomic_json(sample.parents[1]/'dataset.json',{'config':config})
        fixture_splat(sample)
        row=publish(sample.parents[1],sample.name,allow_fixture=True,quality_policy='record_only')
        assert row['label_scope']==scene['label_scope'] and row['target_region']['center_enu_m']==[50.,0.]
        assert row['target_instance_ids']==[i['instance_id'] for i in scene['instances']]
        assert read_json(sample/'metadata.json')['render_environment']==BASESIM_FIELD['source_map']
        assert row==publish(sample.parents[1],sample.name,allow_fixture=True,quality_policy='record_only')
    else:
        with pytest.raises(ValueError,match='BaseSim environment'):seal_capture(sample)


def test_rows_reuse_native_anchors_with_bounds_and_reproducible_counts(catalog):
    from cropvolume.basesim_rows import anchors
    from cropvolume.generation import plant_count_range
    config = pipeline_config({'layout':'rows','field_width_m':4.,'field_length_m':4.})
    scene = generate(catalog,923,config['scene'])
    assert scene == generate(catalog,923,config['scene'])
    assert scene['placement_version'] == 'basesim-authored-rows-v1'
    assert config['scene']['plant_count'] == plant_count_range(4.,4.,'rows')
    assert config['scene']['plant_count'][0] <= len(scene['instances']) <= config['scene']['plant_count'][1]
    xy, indices, proof = anchors()
    source = dict(zip(indices,xy))
    bounds = np.asarray(load_catalog(catalog)['assets'][0]['bounds_m'])
    corners = np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
    for instance in scene['instances']:
        matrix = np.asarray(instance['local_to_world_m'])
        assert np.array_equal(matrix[:2,3],source[instance['source_anchor_index']])
        points = corners @ matrix[:3,:3].T + matrix[:3,3]
        assert (points[:,:2] >= [48.,-2.]).all() and (points[:,:2] <= [52.,2.]).all()
    assert 'exterior is blank' in scene['label_scope']


def test_v3_row_runs_resume_with_saved_grid_context_and_population():
    settings = {**DEFAULTS,'layout':'rows','field_width_m':4.,'field_length_m':4.}
    config = pipeline_config(settings)
    config['scene']['field'] = LEGACY_BASESIM_FIELD.copy()
    config['scene']['plant_count'] = [80,160]
    config['scene']['population_policy'] = {'method':'field-area-density-v1',
        'plants_per_m2':[5,10],'maximum_plants':10000,'area_m2':16.}
    job = {'schema':'cropvolume-generation-v3','settings':settings,'config':config}
    assert saved_job_settings(job) == settings
    bad = copy.deepcopy(job); bad['config']['scene']['field'] = BASESIM_FIELD.copy()
    with pytest.raises(ValueError,match='disagree'):
        saved_job_settings(bad)


def test_bounded_capture_requires_exterior_proof(sample,catalog):
    config = pipeline_config({})
    scene = generate(catalog,17,config['scene'])
    atomic_json(sample/'scene.json',scene)
    (sample/'CAPTURE_COMPLETE.json').unlink()
    proof = {'scene_id':scene['scene_id'],'backend':'unreal_basesim_random',
        'source_map':BASESIM_FIELD['source_map'],'appearance_policy':BASESIM_FIELD['appearance_policy'],
        'environment_before_sha256':'same','environment_after_sha256':'same',
        'all_plant_transforms_verified':True,'maximum_plant_transform_error':0,
        'target_plant_count':len(scene['instances'])}
    atomic_json(sample/'RENDER_FINISHED.json',proof)
    with pytest.raises(ValueError,match='bounded field'):
        seal_capture(sample)
