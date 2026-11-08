"""Select complete existing plant instances without regenerating the authored field."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from .io import atomic_json, child, identity, read_json, sha256
from .profiling import profiled

SWAP = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])


@profiled('inventory_planning', root_arg='output')
def prepare_inventory_export(output, reference):
    output, reference = Path(output), Path(reference)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty inventory output directory')
    receipt = read_json(reference / 'REFERENCE_COMPLETE.json')
    for relative, digest in receipt['files'].items():
        if sha256(child(reference, relative)) != digest:
            raise ValueError('Accepted reference changed: ' + relative)
    source = read_json(reference / 'reference.json')
    content = read_json(reference / 'source_content.json')['files']
    map_file = source['source_map'].removeprefix('/Game/') + '.umap'
    roles = {f'/Game/FlightSchool/Basil/BasilClump_v{i}.BasilClump_v{i}': 'plant' for i in range(4)}
    roles['/Game/FlightSchool/Basil/BasilClumpContext.BasilClumpContext'] = 'plant_context'
    roles['/Game/CROPSIM/Meshes/SoilBed_4m.SoilBed_4m'] = 'soil'
    request = {'schema': 'cropvolume-field-export-request-v1', 'source_map': source['source_map'],
        'source_map_sha256': content[map_file], 'reference_scene_id': source['scene_id'],
        'reference_receipt_sha256': sha256(reference / 'REFERENCE_COMPLETE.json'),
        'expected_content': content, 'mesh_roles': roles}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'export_request.json', request)
    return request


def ue_to_enu(matrix):
    matrix = np.asarray(matrix, float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-9)):
        raise ValueError('Invalid authored instance matrix')
    result = np.eye(4)
    result[:3, :3] = SWAP @ matrix[:3, :3] @ SWAP
    result[:3, 3] = SWAP @ matrix[:3, 3] * .01
    return result


def world_bounds(matrix, bounds_ue_cm):
    bounds = np.asarray(bounds_ue_cm, float)[:, [1, 0, 2]] * .01
    center = matrix[:3, :3] @ bounds.mean(0) + matrix[:3, 3]
    extent = np.abs(matrix[:3, :3]) @ ((bounds[1] - bounds[0]) / 2)
    return np.array([center - extent, center + extent])


def load_inventory(root, require_complete=True):
    root = Path(root)
    if require_complete:
        receipt = read_json(root / 'INVENTORY_COMPLETE.json')
        for name, digest in receipt['files'].items():
            if sha256(child(root, name)) != digest:
                raise ValueError('Inventory checksum mismatch: ' + name)
    data = read_json(root / 'inventory.json')
    inventory_id = data.pop('inventory_id')
    if identity(data) != inventory_id:
        raise ValueError('Inventory identity mismatch')
    data['inventory_id'] = inventory_id
    request = read_json(root / 'export_request.json')
    finished = read_json(root / 'EXPORT_FINISHED.json')
    if (sha256(root / 'export_request.json') != data['export_request_sha256']
            or finished['inventory_id'] != inventory_id
            or finished['inventory_sha256'] != sha256(root / 'inventory.json')
            or request['source_map_sha256'] != data['source_map_sha256']
            or request['source_map'] != data['source_map']
            or request['reference_scene_id'] != data['reference_scene_id']):
        raise ValueError('Inventory provenance mismatch')
    if (data.get('field_unchanged') is not True or not data.get('field_before_sha256')
            or data['field_before_sha256'] != data.get('field_after_sha256')):
        raise ValueError('Authored field changed during inventory export')
    content = read_json(root / 'source_content.json')
    if not request['expected_content'] or any(content.get(key) != request['expected_content']
           for key in ('files', 'files_after', 'staged_files', 'staged_files_after')):
        raise ValueError('Original assets differ from the accepted reference')
    components = {c['component_id']: c for c in data['components']}
    if len(components) != len(data['components']):
        raise ValueError('Duplicate source component identities')
    for mesh, asset in data['assets'].items():
        bounds = np.asarray(asset['bounds_local_ue_cm'])
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or (bounds[1] < bounds[0]).any():
            raise ValueError('Invalid source mesh bounds')
        if asset['role'] != request['mesh_roles'].get(mesh, 'scenery'):
            raise ValueError('Source mesh role changed')
        if mesh.startswith('/Game/'):
            relative = mesh.split('.')[0].removeprefix('/Game/') + '.uasset'
            if asset['package_sha256'] != request['expected_content'].get(relative):
                raise ValueError('Source mesh package changed')
    path = child(root, data['instances_file'])
    if sha256(path) != data['instances_sha256']:
        raise ValueError('Authored instance checksum mismatch')
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != data['instance_count'] or len(rows) != finished['instance_count']:
        raise ValueError('Incomplete authored instance export')
    seen, indices = set(), {key: set() for key in components}
    for row in rows:
        c = components.get(row['component_id'])
        index = row['source_instance_index']
        if (not c or c['mesh'] not in data['assets'] or not isinstance(index, int)
                or not 0 <= index < c['count'] or row['instance_id'] in seen
                or row['instance_id'] != row['component_id'] + '_' + str(index)
                or index in indices[row['component_id']]):
            raise ValueError('Invalid or duplicate source instance identity')
        seen.add(row['instance_id'])
        indices[row['component_id']].add(index)
    # Independently reconstruct all matrices from the original quaternion/scale/translation.
    matrices = np.asarray([r['local_to_world_ue_cm'] for r in rows], float)
    quats = np.asarray([r['rotation_quat_xyzw'] for r in rows], float)
    scales = np.asarray([r['scale'] for r in rows], float)
    positions = np.asarray([r['translation_ue_cm'] for r in rows], float)
    if (not rows or matrices.shape != (len(rows), 4, 4) or quats.shape != (len(rows), 4)
            or scales.shape != (len(rows), 3) or positions.shape != (len(rows), 3)
            or not all(np.isfinite(v).all() for v in (matrices, quats, scales, positions))
            or not np.allclose(np.linalg.norm(quats, axis=1), 1., atol=1e-6)
            or (np.abs(scales) < 1e-9).any()
            or not np.allclose(matrices[:, 3], [0., 0., 0., 1.], atol=1e-9)
            or not np.allclose(matrices[:, :3, 3], positions, atol=1e-7, rtol=0)
            or not np.allclose(matrices[:, :3, :3], Rotation.from_quat(quats).as_matrix() * scales[:, None, :], atol=1e-7, rtol=0)):
        raise ValueError('Exported matrix does not preserve original authored transform')
    if any(len(indices[key]) != c['count'] for key, c in components.items()):
        raise ValueError('Missing source component instances')
    return data, rows


@profiled('inventory_validation', root_arg='root')
def seal_inventory(root):
    root = Path(root)
    if (root / 'INVENTORY_COMPLETE.json').exists():
        load_inventory(root)
        return read_json(root / 'INVENTORY_COMPLETE.json')
    data, rows = load_inventory(root, require_complete=False)
    receipt = {'schema': 'cropvolume-field-inventory-complete-v1', 'inventory_id': data['inventory_id'],
        'instance_count': len(rows), 'field_unchanged': True,
        'files': {name: sha256(root / name) for name in ('inventory.json', 'instances.jsonl',
            'export_request.json', 'EXPORT_FINISHED.json', 'source_content.json')}}
    atomic_json(root / 'INVENTORY_COMPLETE.json', receipt)
    return receipt


def selection_role(asset_role, bounds, low, high, margin):
    if (bounds[1, :2] < low - margin).any() or (bounds[0, :2] > high + margin).any():
        return None
    if (asset_role == 'plant' and (bounds[0, :2] >= low - 1e-9).all()
            and (bounds[1, :2] <= high + 1e-9).all()):
        return 'target_plant'
    if asset_role in ('plant', 'plant_context'):
        intersects = (bounds[1, :2] >= low).all() and (bounds[0, :2] <= high).all()
        return 'boundary_plant' if intersects else 'context_plant'
    return 'support'


@profiled('region_selection', root_arg='output')
def select_region(inventory_root, output, center_enu_m=(50., 0.), patch_m=(2., 2.), context_margin_m=1.):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty region output directory')
    center, patch = np.asarray(center_enu_m, float), np.asarray(patch_m, float)
    if (center.shape != (2,) or patch.shape != (2,) or not np.isfinite([center, patch]).all()
            or (patch <= 0).any() or not np.isfinite(context_margin_m) or context_margin_m < 0):
        raise ValueError('Invalid field region bounds or context margin')
    data, rows = load_inventory(inventory_root)
    low, high = center - patch / 2, center + patch / 2
    components = {c['component_id']: c for c in data['components']}
    selected, counts = [], {'target_plant': 0, 'boundary_plant': 0, 'context_plant': 0, 'support': 0}
    for row in rows:
        c = components[row['component_id']]
        asset = data['assets'][c['mesh']]
        matrix = ue_to_enu(row['local_to_world_ue_cm'])
        bounds = world_bounds(matrix, asset['bounds_local_ue_cm'])
        role = selection_role(asset['role'], bounds, low, high, context_margin_m)
        if role is None:
            continue
        counts[role] += 1
        selected.append({'instance_id': row['instance_id'], 'source_component_id': row['component_id'],
            'source_actor_path': c['actor_path'], 'source_component_path': c['component_path'],
            'source_instance_index': row['source_instance_index'], 'mesh': c['mesh'], 'role': role,
            'local_to_world_m': matrix.tolist(), 'bounds_world_m': bounds.tolist()})
    if not counts['target_plant']:
        raise ValueError('Region contains no complete detailed plants; increase its size or choose another center')
    region = {'schema': 'cropvolume-authored-field-region-v1', 'source_inventory_id': data['inventory_id'],
        'source_inventory_receipt_sha256': sha256(Path(inventory_root) / 'INVENTORY_COMPLETE.json'),
        'source_map': data['source_map'], 'source_map_sha256': data['source_map_sha256'],
        'reference_scene_id': data['reference_scene_id'], 'coordinate_frame': 'ENU_m_Z_up',
        'center_enu_m': center.tolist(), 'patch_m': patch.tolist(), 'context_margin_m': context_margin_m,
        'selection_policy': 'Complete undisplaced source-mesh bounds inside target region; whole boundary plants kept as context',
        'field_policy': 'Logical selection only; original field and all source transforms retained',
        'volume_labels': False, 'counts': counts, 'instances': selected}
    region['region_id'] = identity(region)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'region.json', region)
    draw_region_preview(output, region)
    return region


def verify_region(region_root, inventory_root):
    root = Path(region_root)
    if (root / 'REGION_COMPLETE.json').exists():
        for relative, digest in read_json(root / 'REGION_COMPLETE.json')['files'].items():
            if sha256(child(root, relative)) != digest:
                raise ValueError('Completed region checksum mismatch: ' + relative)
    region = read_json(root / 'region.json')
    region_id = region.pop('region_id')
    if identity(region) != region_id:
        raise ValueError('Region identity mismatch')
    data, rows = load_inventory(inventory_root)
    if (region['source_inventory_id'] != data['inventory_id']
            or region['source_inventory_receipt_sha256'] != sha256(Path(inventory_root) / 'INVENTORY_COMPLETE.json')
            or region['source_map'] != data['source_map'] or region['source_map_sha256'] != data['source_map_sha256']
            or region['reference_scene_id'] != data['reference_scene_id']):
        raise ValueError('Region inventory provenance mismatch')
    by_id = {r['instance_id']: r for r in rows}
    components = {c['component_id']: c for c in data['components']}
    seen, counts = set(), dict.fromkeys(region['counts'], 0)
    for row in region['instances']:
        source = by_id.get(row['instance_id'])
        if not source or row['instance_id'] in seen:
            raise ValueError('Region contains a missing or repeated source instance')
        seen.add(row['instance_id'])
        c = components[source['component_id']]
        matrix = ue_to_enu(source['local_to_world_ue_cm'])
        if (row['source_component_id'] != source['component_id'] or row['source_instance_index'] != source['source_instance_index']
                or row['mesh'] != c['mesh'] or row['source_actor_path'] != c['actor_path']
                or row['source_component_path'] != c['component_path']
                or not np.allclose(matrix, row['local_to_world_m'], atol=1e-9, rtol=0)):
            raise ValueError('Region changed an authored instance or its provenance')
        bounds = world_bounds(matrix, data['assets'][c['mesh']]['bounds_local_ue_cm'])
        if not np.allclose(bounds, row['bounds_world_m'], atol=1e-9, rtol=0):
            raise ValueError('Region instance bounds changed')
        if row['role'] == 'target_plant':
            center, patch = np.asarray(region['center_enu_m']), np.asarray(region['patch_m'])
            if (data['assets'][c['mesh']]['role'] != 'plant'
                    or (bounds[0, :2] < center-patch/2-1e-9).any()
                    or (bounds[1, :2] > center+patch/2+1e-9).any()):
                raise ValueError('Target plant crosses the region boundary')
        counts[row['role']] += 1
    if counts != region['counts'] or not counts.get('target_plant'):
        raise ValueError('Region instance counts mismatch')
    center, patch = np.asarray(region['center_enu_m']), np.asarray(region['patch_m'])
    expected = {}
    for source in rows:
        c = components[source['component_id']]
        asset = data['assets'][c['mesh']]
        bounds = world_bounds(ue_to_enu(source['local_to_world_ue_cm']), asset['bounds_local_ue_cm'])
        role = selection_role(asset['role'], bounds, center-patch/2, center+patch/2, region['context_margin_m'])
        if role:
            expected[source['instance_id']] = role
    if expected != {r['instance_id']: r['role'] for r in region['instances']}:
        raise ValueError('Region omitted source instances or changed target/context selection')
    receipt = {'schema': 'cropvolume-authored-region-complete-v1', 'region_id': region_id,
        'source_inventory_id': data['inventory_id'], 'counts': counts, 'transforms_preserved': True,
        'volume_labels': False, 'files': {name: sha256(root / name) for name in ('region.json', 'region_preview.png')}}
    if not (root / 'REGION_COMPLETE.json').exists():
        atomic_json(root / 'REGION_COMPLETE.json', receipt)
    return receipt


def draw_region_preview(output, region):
    image = Image.new('RGB', (900, 900), '#171d21')
    draw = ImageDraw.Draw(image)
    center, patch = np.asarray(region['center_enu_m']), np.asarray(region['patch_m'])
    half = max(patch) / 2 + region['context_margin_m'] + .25
    scale = 760 / (2 * half)
    def xy(point):
        return (450 + (point[0]-center[0])*scale, 460 - (point[1]-center[1])*scale)
    colors = {'support': '#333c42', 'context_plant': '#58656b', 'boundary_plant': '#bb863e', 'target_plant': '#65bd70'}
    for role in ('support', 'context_plant', 'boundary_plant', 'target_plant'):
        for instance in region['instances']:
            if instance['role'] == role:
                b = np.asarray(instance['bounds_world_m'])
                draw.rectangle([xy([b[0, 0], b[1, 1]]), xy([b[1, 0], b[0, 1]])], outline=colors[role], width=1)
    draw.rectangle([xy(center+[-patch[0]/2, patch[1]/2]), xy(center+[patch[0]/2, -patch[1]/2])], outline='#67c4ef', width=4)
    draw.text((20, 18), 'Authored BaseSim field region: source-mesh bounds, top view (E right, N up)', fill='white')
    draw.text((20, 40), f"Green: {region['counts']['target_plant']} complete target plants | Amber: boundary context | Gray: nearby context", fill='white')
    draw.text((20, 870), 'Blue: target region. Existing plants remain whole at their original transforms. Field is unchanged.', fill='white')
    image.save(Path(output) / 'region_preview.png')
