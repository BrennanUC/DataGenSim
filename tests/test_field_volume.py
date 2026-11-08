from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from cropvolume.field_volume import validate_render_proof, validate_solid_materials
from cropvolume.geometry import audit_catalog, measure_scene
from cropvolume.io import atomic_json, read_json


def render_fixture():
    scene = {'scene_id': 'solid-region', 'instances': [{'instance_id': 'original_1', 'asset_id': 'solid_v1',
        'source_component_path': 'Field.Foliage.Plants', 'source_instance_index': 8371,
        'local_to_world_m': [[1,0,0,50], [0,1,0,0], [0,0,1,.08], [0,0,0,1]]}]}
    proof = {'solid_scene_id': 'solid-region', 'scene_id': 'solid-region', 'field_unchanged': True,
        'field_before_sha256': 'frozen', 'field_after_sha256': 'frozen',
        'binding': {'all_authored_transforms_preserved': True, 'non_target_before_sha256': 'context',
                    'non_target_after_sha256': 'context', 'target_count': 1},
        'target_instances': [{**deepcopy(scene['instances'][0]), 'forced_lod': 0}]}
    return scene, proof


def test_render_proof_binds_the_original_target_id_and_transform():
    scene, proof = render_fixture()
    validate_render_proof(scene, proof)


@pytest.mark.parametrize('change', ['move', 'asset', 'source_index', 'context', 'missing', 'lod'])
def test_render_proof_rejects_target_or_context_changes(change):
    scene, proof = render_fixture()
    if change == 'move':
        proof['target_instances'][0]['local_to_world_m'][0][3] += .001
    elif change == 'asset':
        proof['target_instances'][0]['asset_id'] = 'wrong_solid'
    elif change == 'source_index':
        proof['target_instances'][0]['source_instance_index'] += 1
    elif change == 'context':
        proof['binding']['non_target_after_sha256'] = 'changed'
    elif change == 'missing':
        proof['target_instances'] = []
    else:
        proof['target_instances'][0]['forced_lod'] = 1
    with pytest.raises(ValueError):
        validate_render_proof(scene, proof)


@pytest.mark.parametrize('change', ['wind', 'mask', 'nanite'])
def test_render_geometry_rejects_shader_motion_cutouts_and_nanite(change):
    spec = {'nanite_enabled': False, 'material_inspection': [{'blend_mode': 'BLEND_OPAQUE',
        'opacity_mask_connected': False, 'displacement_connected': True, 'displacement_constant_zero': True}]}
    validate_solid_materials(spec)
    if change == 'wind':
        spec['material_inspection'][0]['displacement_constant_zero'] = False
    elif change == 'mask':
        spec['material_inspection'][0]['opacity_mask_connected'] = True
    else:
        spec['nanite_enabled'] = True
    with pytest.raises(ValueError):
        validate_solid_materials(spec)


def test_authored_overlap_union_with_tilt_scale_and_far_field_origin(tmp_path):
    trimesh.creation.box().export(tmp_path / 'cube.ply')
    atomic_json(tmp_path / 'source.json', {'assets': [{'id': 'solid_cube', 'mesh': 'cube.ply', 'units': 'm',
        'species': 'fixture', 'geometry_matches_render': True, 'material_geometry_reviewed': True,
        'geometry_version': 'analytic-test-only'}]})
    catalog = audit_catalog(tmp_path / 'source.json', tmp_path / 'catalog.json')
    first = np.eye(4)
    first[:3, :3] = Rotation.from_euler('xyz', [14, -9, 37], degrees=True).as_matrix() @ np.diag([2,3,4])
    first[:3, 3] = [50, -20, .08]
    second = first.copy()
    second[:3, 3] += first[:3, :3] @ [.5, 0, 0]
    scene = {'scene_id': 'analytic', 'catalog_id': catalog['catalog_id'], 'instances': [
        {'instance_id': 'a', 'role': 'plant', 'asset_id': 'solid_cube', 'local_to_world_m': first.tolist()},
        {'instance_id': 'b', 'role': 'plant', 'asset_id': 'solid_cube', 'local_to_world_m': second.tolist()},
        {'role': 'soil', 'asset_id': 'not-a-plant'}, {'role': 'context_plant', 'asset_id': 'not-a-target'}]}
    label = measure_scene(scene, tmp_path / 'catalog.json', boolean_origin_m=[50, -20, 0])
    assert label['sum_instance_volume_m3'] == pytest.approx(48)
    assert label['plant_solid_volume_m3'] == pytest.approx(36, rel=2e-6)
    assert label['boolean_origin_m'] == [50, -20, 0]
    assert scene['instances'][0]['local_to_world_m'] == first.tolist()


def test_invalid_boolean_origin_is_rejected(catalog):
    with pytest.raises(ValueError, match='Boolean origin'):
        measure_scene({'catalog_id': read_json(catalog)['catalog_id'], 'instances': []},
                      catalog, boolean_origin_m=[float('nan'), 0, 0])
