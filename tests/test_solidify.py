import numpy as np
import pytest
import trimesh

from cropvolume.solidify import thicken_surface, cap_planar_stem_ends, solidify_export
from cropvolume.io import atomic_json
from plyfile import PlyData, PlyElement


def test_leaf_disk_gets_explicit_solid_thickness():
    plane = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[1,1,0],[0,1,0]], faces=[[0,1,2],[0,2,3]], process=False)
    solid = thicken_surface(plane, .0003)
    assert solid.is_volume
    assert solid.volume == pytest.approx(.0003)
    assert np.allclose(solid.bounds[:, 2], [-.00015, .00015])


def test_invalid_thickness_and_closed_input_fail():
    with pytest.raises(ValueError):
        thicken_surface(trimesh.creation.box(), .001)
    with pytest.raises(ValueError):
        thicken_surface(trimesh.creation.box(), float("nan"))


def test_explicit_stem_end_caps_preserve_tube_volume():
    tube = trimesh.creation.cylinder(radius=.005, height=.2, sections=16)
    expected = tube.volume
    tube.update_faces(np.abs(tube.face_normals[:, 2]) < .5)
    repaired = cap_planar_stem_ends(tube)
    assert repaired.is_volume and repaired.volume == pytest.approx(expected)


def test_leaf_shell_preserves_source_uvs_on_both_sides(tmp_path):
    vertices = np.array([(0,0,0),(1,0,0),(1,1,0),(0,1,0)], dtype=[('x','f4'),('y','f4'),('z','f4')])
    faces = np.empty(2,dtype=[('vertex_indices','i4',(3,)),('material_id','i4')])
    faces['vertex_indices']=[[0,1,2],[0,2,3]]; faces['material_id']=0
    source = tmp_path/'leaf.ply'
    PlyData([PlyElement.describe(vertices,'vertex'),PlyElement.describe(faces,'face')]).write(source)
    uv=np.array([[[0,0],[1,0],[1,1]],[[0,0],[1,1],[0,1]]],float)
    atomic_json(source.with_suffix('.uv0.json'),uv.tolist())
    colors=np.array([[[.2,.6,.1,1]]*3,[[.3,.7,.15,1]]*3])
    atomic_json(source.with_suffix('.color0.json'),colors.tolist())
    result=solidify_export(source,tmp_path/'closed.json')
    actual=np.asarray(result['triangle_uv0'])
    assert np.array_equal(actual[:2],uv)
    assert np.array_equal(actual[2:4],uv[:,::-1])
    assert np.array_equal(np.asarray(result['triangle_colors'])[:2],colors)
    assert np.array_equal(np.asarray(result['triangle_colors'])[2:4],colors[:,::-1])
    assert result['plant_solid_volume_m3']==pytest.approx(.0003)
