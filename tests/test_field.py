import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cropvolume.field import load_inventory, seal_inventory, select_region, ue_to_enu, verify_region
from cropvolume.io import atomic_json, identity, read_json, sha256


def finish_export(root, data):
    data.pop('inventory_id', None)
    data['instances_sha256'] = sha256(root / 'instances.jsonl')
    data['inventory_id'] = identity(data)
    atomic_json(root / 'inventory.json', data)
    atomic_json(root / 'EXPORT_FINISHED.json', {'inventory_id': data['inventory_id'],
        'inventory_sha256': sha256(root / 'inventory.json'), 'instance_count': data['instance_count']})


@pytest.fixture
def inventory(tmp_path):
    root = tmp_path / 'inventory'
    root.mkdir()
    plant, context, soil = '/Game/Plant.Plant', '/Game/Context.Context', '/Game/Soil.Soil'
    content = {'Field.umap': 'map', 'Plant.uasset': 'plant', 'Context.uasset': 'context', 'Soil.uasset': 'soil'}
    roles = {plant: 'plant', context: 'plant_context', soil: 'soil'}
    request = {'source_map': '/Game/Field', 'source_map_sha256': 'map', 'reference_scene_id': 'reference',
               'mesh_roles': roles, 'expected_content': content}
    atomic_json(root / 'export_request.json', request)
    atomic_json(root / 'source_content.json', {key: content for key in ('files', 'files_after', 'staged_files', 'staged_files_after')})
    components, rows = [], []
    q = Rotation.from_euler('xyz', [14, -7, 32], degrees=True)
    for key, mesh, positions in [('plants', plant, [[0, 5000, 8], [0, 5095, 8], [0, 5150, 8], [0, 5400, 8]]),
                                 ('context', context, [[50, 5000, 8]]), ('soil', soil, [[0, 5000, 0]])]:
        components.append({'component_id': key, 'actor_path': 'Field.Foliage', 'component_path': 'Field.Foliage.' + key,
                           'mesh': mesh, 'count': len(positions)})
        for index, pos in enumerate(positions):
            matrix = np.eye(4)
            matrix[:3, :3] = q.as_matrix() @ np.diag([1.1, .8, .9])
            matrix[:3, 3] = pos
            rows.append({'instance_id': key + '_' + str(index), 'component_id': key, 'source_instance_index': index,
                'local_to_world_ue_cm': matrix.tolist(), 'translation_ue_cm': pos,
                'rotation_quat_xyzw': q.as_quat().tolist(), 'scale': [1.1, .8, .9]})
    assets = {mesh: {'bounds_local_ue_cm': [[-20, -10, -5], [20, 10, 40]],
                    'role': role, 'package_sha256': content[mesh.split('/')[-1].split('.')[0] + '.uasset']}
              for mesh, role in roles.items()}
    data = {'source_map': '/Game/Field', 'source_map_sha256': 'map', 'reference_scene_id': 'reference',
            'assets': assets, 'components': components, 'instances_file': 'instances.jsonl', 'instance_count': len(rows),
            'field_unchanged': True, 'field_before_sha256': 'unchanged', 'field_after_sha256': 'unchanged',
            'export_request_sha256': sha256(root / 'export_request.json')}
    (root / 'instances.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    finish_export(root, data)
    seal_inventory(root)
    return root


def test_coordinate_conversion_preserves_tilt_nonuniform_scale_and_metric_position():
    matrix = np.eye(4)
    rotation = Rotation.from_euler('xyz', [17, -11, 48], degrees=True).as_matrix()
    matrix[:3, :3] = rotation @ np.diag([1.2, .8, .6])
    matrix[:3, 3] = [100, 200, 300]
    got = ue_to_enu(matrix)
    assert np.allclose(got[:3, 3], [2, 1, 3])
    # A known local point, evaluated independently in UE centimetres then converted to ENU.
    local_enu_m = np.array([.2, .4, .7, 1.])
    point_ue_cm = np.array([40., 20., 70.])
    world_ue_cm = matrix[:3, :3] @ point_ue_cm + matrix[:3, 3]
    assert np.allclose((got @ local_enu_m)[:3], world_ue_cm[[1, 0, 2]] / 100)
    assert abs(got[2, 0]) > .01
    assert np.linalg.det(got[:3, :3]) == pytest.approx(1.2*.8*.6)


def test_selection_keeps_whole_targets_boundary_context_and_original_transforms(inventory, tmp_path):
    region = select_region(inventory, tmp_path / 'region')
    assert region['counts'] == {'target_plant': 1, 'boundary_plant': 2, 'context_plant': 1, 'support': 1}
    by_id = {r['instance_id']: r for r in region['instances']}
    assert 'plants_3' not in by_id
    assert by_id['plants_1']['role'] == 'boundary_plant'
    assert by_id['context_0']['role'] != 'target_plant'
    target = np.asarray(by_id['plants_0']['local_to_world_m'])
    assert target[0, 3] == pytest.approx(50.)
    assert target[2, 3] == pytest.approx(.08)
    assert abs(target[2, 0]) > .01
    receipt = verify_region(tmp_path / 'region', inventory)
    assert receipt['transforms_preserved'] and not receipt['volume_labels']
    assert not (tmp_path / 'region/labels').exists()
    again = select_region(inventory, tmp_path / 'again')
    assert again['region_id'] == region['region_id']


def test_inventory_rejects_matrix_that_does_not_match_source_quaternion(inventory):
    path = inventory / 'instances.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['local_to_world_ue_cm'][0][0] += .01
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    finish_export(inventory, read_json(inventory / 'inventory.json'))
    with pytest.raises(ValueError, match='authored transform'):
        load_inventory(inventory, require_complete=False)


@pytest.mark.parametrize('change', ['duplicate', 'missing'])
def test_inventory_rejects_duplicate_and_missing_instances(inventory, change):
    path = inventory / 'instances.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == 'duplicate':
        rows[1] = rows[0]
    else:
        rows.pop()
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    finish_export(inventory, read_json(inventory / 'inventory.json'))
    with pytest.raises(ValueError, match='duplicate|Incomplete'):
        load_inventory(inventory, require_complete=False)


@pytest.mark.parametrize('change', ['move', 'boundary'])
def test_region_rejects_moving_a_plant_or_labeling_a_boundary_plant_as_target(inventory, tmp_path, change):
    output = tmp_path / 'region'
    data = select_region(inventory, output)
    data.pop('region_id')
    if change == 'move':
        data['instances'][0]['local_to_world_m'][0][3] += .01
    else:
        row = next(r for r in data['instances'] if r['instance_id'] == 'plants_1')
        row['role'] = 'target_plant'
        data['counts']['boundary_plant'] -= 1
        data['counts']['target_plant'] += 1
    data['region_id'] = identity(data)
    atomic_json(output / 'region.json', data)
    with pytest.raises(ValueError, match='authored instance|boundary'):
        verify_region(output, inventory)


def test_empty_or_invalid_regions_are_not_published(inventory, tmp_path):
    for center, patch in [([90, 20], [2, 2]), ([50, 0], [0, 2]), ([np.nan, 0], [2, 2])]:
        with pytest.raises(ValueError):
            select_region(inventory, tmp_path / 'bad_region', center, patch)
    assert not (tmp_path / 'bad_region').exists()


def test_region_cannot_silently_drop_neighboring_context(inventory, tmp_path):
    output = tmp_path / 'region'
    data = select_region(inventory, output)
    data.pop('region_id')
    data['instances'] = [r for r in data['instances'] if r['role'] != 'context_plant']
    data['counts']['context_plant'] = 0
    data['region_id'] = identity(data)
    atomic_json(output / 'region.json', data)
    with pytest.raises(ValueError, match='omitted'):
        verify_region(output, inventory)
