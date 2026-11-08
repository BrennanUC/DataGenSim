import numpy as np
import pytest

from cropvolume.geometry import measure_scene, overlap_groups
from cropvolume.io import read_json


def scene_for(catalog,positions,scales=None):
    data=read_json(catalog)
    instances=[]
    for i,position in enumerate(positions):
        transform=np.eye(4)
        transform[:3,3]=position
        if scales:transform[:3,:3]=np.diag(scales[i])
        instances.append({'role':'plant','instance_id':str(i),'asset_id':data['assets'][0]['id'],
            'local_to_world_m':transform.tolist()})
    return {'scene_id':'analytic-cache-test','catalog_id':data['catalog_id'],'instances':instances}


def test_isolated_instances_use_cached_volume_without_loading_meshes(catalog,monkeypatch):
    import cropvolume.geometry as geometry
    scene=scene_for(catalog,[(0,0,0),(10,0,0)],[(1,1,1),(2,3,4)])
    monkeypatch.setattr(geometry,'load_mesh',lambda *a,**k:pytest.fail('Isolated plants must use their cached volume'))
    monkeypatch.setattr(geometry,'solid_union',lambda *a,**k:pytest.fail('No overlap means no Boolean'))
    label=measure_scene(scene,catalog)
    base=read_json(catalog)['assets'][0]['volume_m3']
    assert label['plant_solid_volume_m3']==pytest.approx(base*25)
    assert label['calculation']['cached_instances']==2 and label['calculation']['boolean_instances']==0


def test_only_overlap_group_loads_geometry_and_counts_intersections_once(catalog,monkeypatch):
    import cropvolume.geometry as geometry
    scene=scene_for(catalog,[(0,0,0),(.1,0,0),(10,0,0),(20,0,0)])
    actual=geometry.solid_union;calls=[]
    def union(meshes,**kwargs):
        calls.append(len(meshes));return actual(meshes,**kwargs)
    monkeypatch.setattr(geometry,'solid_union',union)
    label=measure_scene(scene,catalog)
    assert calls==[2] and label['calculation']['cached_instances']==2
    assert label['plant_solid_volume_m3']==pytest.approx(.024+2*.016,rel=1e-6)


def test_overlap_chain_stays_one_group(catalog):
    label=measure_scene(scene_for(catalog,[(0,0,0),(.15,0,0),(.3,0,0)]),catalog)
    assert label['calculation']['largest_overlap_group']==3
    assert label['plant_solid_volume_m3']==pytest.approx(.04,rel=1e-6)


@pytest.mark.parametrize('positions,expected',[
    ([(0,0,0),(.15,0,0),(.3,0,0)],.04),
    ([(0,0,0),(0,0,0),(0,0,0)],.016),
    ([(0,0,0),(.1,0,0),(.15,0,0)],.028)])
def test_streamed_overlap_volume_counts_triple_intersections_once(catalog,monkeypatch,positions,expected):
    import cropvolume.geometry as geometry
    monkeypatch.setattr(geometry,'MAX_BOOLEAN_GROUP',2)
    label=measure_scene(scene_for(catalog,positions),catalog)
    assert label['calculation']['streamed_overlap_groups']==1
    assert label['plant_solid_volume_m3']==pytest.approx(expected,rel=1e-6)


def test_rotated_bounds_and_nonuniform_scale_are_conservative(catalog):
    scene=scene_for(catalog,[(100000,100000,0),(100000,100000,0)],[(2,3,4),(2,3,4)])
    label=measure_scene(scene,catalog)
    base=read_json(catalog)['assets'][0]['volume_m3']
    assert label['plant_solid_volume_m3']==pytest.approx(base*24,rel=2e-6)
    assert label['sum_instance_volume_m3']==pytest.approx(base*48)


def test_sparse_large_field_does_not_build_whole_field_union(catalog,monkeypatch):
    import cropvolume.geometry as geometry
    scene=scene_for(catalog,[(i%100,i//100,0) for i in range(2000)])
    monkeypatch.setattr(geometry,'solid_union',lambda *a,**k:pytest.fail('Sparse field must not require a union'))
    label=measure_scene(scene,catalog)
    assert label['calculation']['cached_instances']==2000
    assert label['plant_solid_volume_m3']==pytest.approx(2000*.016,rel=1e-6)


def test_spatial_groups_match_a_brute_force_graph():
    rng=np.random.default_rng(6)
    lows=rng.uniform(0,4,(80,3));highs=lows+rng.uniform(.1,1,(80,3))
    unseen=set(range(len(lows)));expected=[]
    while unseen:
        reached=set();queue=[unseen.pop()]
        while queue:
            i=queue.pop();reached.add(i)
            adjacent={j for j in unseen if np.all(highs[i]>lows[j]) and np.all(highs[j]>lows[i])}
            unseen-=adjacent;queue.extend(adjacent)
        expected.append(frozenset(reached))
    actual=overlap_groups(list(zip(lows,highs)))
    assert set(map(frozenset,actual))==set(expected)
