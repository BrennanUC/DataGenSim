import numpy as np
import pytest

from cropvolume.scenes import generate
from cropvolume.geometry import load_catalog
from cropvolume.io import read_json, atomic_json, identity
from cropvolume.prepare import seal_capture


def test_game_bed_plants_are_upright_and_on_the_bed_surface(catalog):
    field={'backend':'unreal_game','bed_count':2,'bed_spacing_m':1.016,'bed_top_m':.09,
           'plant_step_m':.17,'lane_offsets_m':[-.22,0,.22]}
    config={'layout':'game_beds','field':field,'patch_m':[1.36,2.032],'plant_count':[12,12],'scale':[1,1]}
    scene=generate(catalog,33,config)
    assert scene==generate(catalog,33,config)
    bounds=np.asarray(load_catalog(catalog)['assets'][0]['bounds_m'])
    for instance in scene['instances']:
        m=np.asarray(instance['local_to_world_m'])
        assert np.allclose(m[:3,2],[0,0,1])
        assert m[2,3]+bounds[0,2]==pytest.approx(.09)
        centers=[(b-.5)*1.016+v for b in range(2) for v in field['lane_offsets_m']]
        assert min(abs(m[1,3]-v) for v in centers)<.009


def test_game_capture_rejects_bad_native_placement_before_sealing(sample):
    scene=read_json(sample/'scene.json'); scene.pop('scene_id')
    scene['field']={'backend':'unreal_game'}; scene['scene_id']=identity(scene)
    atomic_json(sample/'scene.json',scene)
    (sample/'CAPTURE_COMPLETE.json').unlink()
    atomic_json(sample/'RENDER_FINISHED.json',{'scene_id':scene['scene_id'],'backend':'unreal_game',
                'all_plant_transforms_verified':True,'maximum_plant_transform_error':0,'minimum_plant_clearance_m':-.05})
    with pytest.raises(ValueError,match='placement'):
        seal_capture(sample)
