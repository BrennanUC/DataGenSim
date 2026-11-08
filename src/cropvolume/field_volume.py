"""Bind selected authored plants to explicit solid versions and validate their labels."""
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

from .cameras import capture_plan, export_cameras, import_colmap
from .field import load_inventory, verify_region
from .geometry import audit_catalog, load_catalog, load_mesh, measure_scene, solid_union
from .io import atomic_json, child, identity, read_json, sha256
from .profiling import profiled, span, annotate


def validate_render_proof(scene, proof):
    if (proof.get('solid_scene_id') != scene['scene_id'] or proof.get('scene_id') != scene['scene_id']
            or proof.get('field_unchanged') is not True
            or not proof.get('field_before_sha256')
            or proof['field_before_sha256'] != proof.get('field_after_sha256')):
        raise ValueError('Solid render scene identity or frozen field state differs')
    binding = proof.get('binding', {})
    if (binding.get('all_authored_transforms_preserved') is not True
            or not binding.get('non_target_before_sha256')
            or binding['non_target_before_sha256'] != binding.get('non_target_after_sha256')
            or binding.get('target_count') != len(scene['instances'])):
        raise ValueError('Non-target field changed or selected target count differs')
    actual = {r['instance_id']: r for r in proof.get('target_instances', [])}
    if len(actual) != len(proof.get('target_instances', [])) or set(actual) != {r['instance_id'] for r in scene['instances']}:
        raise ValueError('Missing or duplicate rendered target plants')
    for row in scene['instances']:
        got = actual[row['instance_id']]
        if (got['asset_id'] != row['asset_id'] or got['source_component_path'] != row['source_component_path']
                or got['source_instance_index'] != row['source_instance_index'] or got.get('forced_lod') != 0
                or not np.allclose(got['local_to_world_m'], row['local_to_world_m'], atol=2e-8, rtol=0)):
            raise ValueError('Rendered target transform, source identity or geometry version differs')


def validate_solid_materials(spec):
    if spec.get('nanite_enabled') is not False or not spec.get('material_inspection'):
        raise ValueError('Target geometry needs an audited non-Nanite render mesh and materials')
    for material in spec['material_inspection']:
        if ('inspection_error' in material or material.get('opacity_mask_connected') is not False
                or 'BLEND_OPAQUE' not in material.get('blend_mode', '')
                or (material.get('displacement_connected') and material.get('displacement_constant_zero') is not True)):
            raise ValueError('Target material is masked, unreviewed, or moves geometry')


@profiled('solid_scene_planning', root_arg='output')
def plan_volume_region(output, region_root, inventory_root, reference_root, catalog_path,
                       recipe_root, source_export, capture_views=None):
    output, reference_root = Path(output), Path(reference_root)
    recipe_root, source_export = Path(recipe_root), Path(source_export)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh solid-region output directory')
    verify_region(region_root, inventory_root)
    inventory, _ = load_inventory(inventory_root)
    region = read_json(Path(region_root) / 'region.json')
    reference = read_json(reference_root / 'reference.json')
    if reference['scene_id'] != region['reference_scene_id']:
        raise ValueError('Region does not belong to this approved reference')
    for relative, digest in read_json(reference_root / 'REFERENCE_COMPLETE.json')['files'].items():
        if sha256(child(reference_root, relative)) != digest:
            raise ValueError('Approved reference was modified')
    catalog = load_catalog(catalog_path)
    recipes = {r['source_asset']: r for r in read_json(recipe_root / 'manifest.json')['assets']}
    sources = {r['unreal_asset']: r for r in read_json(source_export / 'assets.review.json')['assets']}
    approved = {a['id']: a for a in catalog['assets']}
    target_meshes = {r['mesh'] for r in region['instances'] if r['role'] == 'target_plant'}
    assets = {}
    for mesh in sorted(target_meshes):
        if mesh not in recipes or mesh not in sources:
            raise ValueError('Missing explicit source-to-solid geometry mapping: ' + mesh)
        recipe, source = recipes[mesh], sources[mesh]
        if (source['unreal_uasset_sha256'] != inventory['assets'][mesh]['package_sha256']
                or not source['geometry_version'].startswith('source-') or source['units'] != 'm'):
            raise ValueError('Solid recipe was built from a different original mesh')
        geometry_path = child(recipe_root, recipe['geometry'])
        geometry = read_json(geometry_path)
        geometry_id = geometry.pop('geometry_id')
        if (identity(geometry) != geometry_id or recipe['geometry_id'] != geometry_id
                or geometry['source_sha256'] != sha256(child(source_export, source['mesh']))
                or recipe['source_materials'] != source['materials']):
            raise ValueError('Solid geometry recipe/source provenance mismatch')
        for key, suffix in (('source_uv0_sha256', '.uv0.json'), ('source_colors_sha256', '.color0.json')):
            if key in geometry and geometry[key] != sha256(child(source_export, source['mesh']).with_suffix(suffix)):
                raise ValueError('Solid geometry source attributes changed')
        asset_id = recipe['id'] + recipe.get('asset_suffix', '_solid')
        asset = approved.get(asset_id)
        if (not asset or asset['solid_recipe']['geometry_id'] != geometry_id
                or asset['solid_recipe']['leaf_thickness_m'] != geometry['leaf_thickness_m']):
            raise ValueError('Approved solid catalog differs from the original-field geometry recipe')
        assets[mesh] = {key: deepcopy(asset[key]) for key in ('id', 'unreal_asset', 'unreal_uasset_sha256',
            'solid_recipe', 'bounds_m', 'species', 'geometry_version')}
        assets[mesh].update(source_asset=mesh, original_uasset_sha256=source['unreal_uasset_sha256'],
            source_ply_sha256=geometry['source_sha256'], recipe_file_sha256=sha256(geometry_path),
            stem_repair=geometry.get('stem_repair'))
    instances = []
    for row in region['instances']:
        if row['role'] != 'target_plant':
            continue
        asset = assets[row['mesh']]
        # Solid thickness must not silently change the complete-target selection.
        m, b = np.asarray(row['local_to_world_m']), np.asarray(asset['bounds_m'])
        center = m[:3, :3] @ b.mean(0) + m[:3, 3]
        extent = np.abs(m[:3, :3]) @ ((b[1]-b[0])/2)
        low, high = np.asarray(region['center_enu_m'])-np.asarray(region['patch_m'])/2, np.asarray(region['center_enu_m'])+np.asarray(region['patch_m'])/2
        if (center[:2]-extent[:2] < low-1e-8).any() or (center[:2]+extent[:2] > high+1e-8).any():
            raise ValueError('Thickened target no longer fits completely in the selected region')
        instances.append({'instance_id': row['instance_id'], 'role': 'plant', 'asset_id': asset['id'],
            'source_mesh': row['mesh'], 'source_actor_path': row['source_actor_path'],
            'source_component_path': row['source_component_path'], 'source_instance_index': row['source_instance_index'],
            'local_to_world_m': row['local_to_world_m']})
    scene = {'schema': 'cropvolume-authored-solid-scene-v1', 'source_region_id': region['region_id'],
        'source_inventory_id': inventory['inventory_id'], 'source_map': region['source_map'],
        'source_map_sha256': region['source_map_sha256'], 'reference_scene_id': region['reference_scene_id'],
        'reference_catalog_id': catalog['catalog_id'], 'center_enu_m': region['center_enu_m'], 'patch_m': region['patch_m'],
        'coordinate_frame': 'ENU_m_Z_up', 'assets': list(assets.values()), 'instances': instances,
        'field_policy': 'preserve_authored_placement_with_versioned_target_solids',
        'label_scope': 'Union of complete selected plant solids only; visible context plants, soil and scenery excluded',
        'thickness_notice': 'Explicit synthetic leaf thickness, not a measured biological volume',
        'expected_content': read_json(Path(inventory_root) / 'export_request.json')['expected_content']}
    scene['scene_id'] = identity(scene)
    plan = read_json(reference_root / 'capture.json')
    if capture_views is not None:
        first = plan['frames'][0]
        plan = capture_plan(scene, views=capture_views, width=first['width'], height=first['height'], split_policy='balanced_rings')
        shift = np.asarray(region['center_enu_m'])
    else:
        shift = np.asarray(region['center_enu_m'])-np.asarray(reference['center_enu_m'])
    plan['scene_id'] = scene['scene_id']
    for frame in plan['frames']:
        for i in range(2):
            frame['c2w_cv'][i][3] += float(shift[i])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'solid_scene.json', scene)
    atomic_json(output / 'reference.json', {k: scene[k] for k in ('scene_id', 'source_map', 'field_policy', 'patch_m', 'center_enu_m')})
    atomic_json(output / 'solid_assets.json', {'assets': [a['solid_recipe'] for a in scene['assets']]})
    atomic_json(output / 'capture.json', plan)
    export_cameras(output, plan['frames'])
    return scene


@profiled('volume_finalization', root_arg='root')
def finalize_volume_region(root):
    root = Path(root)
    if (root / 'VOLUME_COMPLETE.json').exists():
        annotate(reused=True)
        receipt = read_json(root / 'VOLUME_COMPLETE.json')
        for relative, digest in receipt['files'].items():
            if sha256(child(root, relative)) != digest:
                raise ValueError('Completed volume-region artifact changed: ' + relative)
        return receipt
    scene = read_json(root / 'solid_scene.json')
    scene_id = scene.pop('scene_id')
    if identity(scene) != scene_id:
        raise ValueError('Solid scene identity mismatch')
    scene['scene_id'] = scene_id
    proof = read_json(root / 'SOLID_RENDER_FINISHED.json')
    validate_render_proof(scene, proof)
    content = read_json(root / 'source_content.json')
    if (any(content.get(k) != scene['expected_content'] for k in ('files', 'files_after', 'staged_files', 'staged_files_after'))
            or not content.get('solid_files') or content['solid_files'] != content.get('solid_files_after')):
        raise ValueError('Original or solid assets changed during rendering')
    assets = {a['id']: a for a in scene['assets']}
    report = read_json(root / 'render_meshes/assets.review.json')
    if {s['id'] for s in report['assets']} != set(assets) or len(report['assets']) != len(assets):
        raise ValueError('Missing or duplicate final render meshes')
    checks = []
    report_by_id = {s['id']: s for s in report['assets']}
    for target in proof['target_instances']:
        if target.get('material_paths') != report_by_id[target['asset_id']]['materials']:
            raise ValueError('Rendered target uses materials different from the final mesh audit')
    for spec in report['assets']:
        asset = assets[spec['id']]
        validate_solid_materials(spec)
        if (spec['unreal_uasset_sha256'] != asset['unreal_uasset_sha256'] or spec['solid_recipe'] != asset['solid_recipe']):
            raise ValueError('Final render geometry differs from the bound asset version')
        for material in [*spec['materials'], *(m['base_material'] for m in spec['material_inspection'])]:
            package = material.split('.')[0].removeprefix('/Game/')+'.uasset'
            if 'CropVolume/' not in package or package not in content['solid_files']:
                raise ValueError('Final target material dependency was not checksummed')
        actual = float(solid_union([load_mesh(child(root / 'render_meshes', spec['mesh']))]).volume)
        expected = asset['solid_recipe']['expected_volume_m3']
        error = abs(actual-expected)/expected
        if not np.isfinite(error) or error > 1e-4:
            raise ValueError('Imported final render geometry changed its solid volume')
        spec.update(geometry_matches_render=True, material_geometry_reviewed=True,
                    geometry_version='solid-shell-v1-'+asset['solid_recipe']['geometry_id'])
        checks.append({'asset_id': spec['id'], 'volume_m3': actual, 'relative_roundtrip_error': error})
    atomic_json(root / 'render_meshes/assets.approved.json', report)
    catalog_path = root / 'geometry/catalog.json'
    catalog = audit_catalog(root / 'render_meshes/assets.approved.json', catalog_path)
    if catalog['rejected'] or len(catalog['assets']) != len(assets):
        raise ValueError('Fresh render geometry did not pass the solid catalog audit')
    measured_scene = {**scene, 'catalog_id': catalog['catalog_id']}
    origin = [*scene['center_enu_m'], 0.]
    with span('volume_union', target_plants=len(scene['instances'])):
        label = measure_scene(measured_scene, catalog_path, boolean_origin_m=origin)
    total, union = label['sum_instance_volume_m3'], label['plant_solid_volume_m3']
    if not 0 < union <= total*(1+1e-4):
        raise ValueError('Target union volume is inconsistent with individual plant volumes')
    label.update(source_region_id=scene['source_region_id'], source_inventory_id=scene['source_inventory_id'],
        label_scope=scene['label_scope'], thickness_notice=scene['thickness_notice'],
        render_receipt_sha256=sha256(root / 'SOLID_RENDER_FINISHED.json'),
        geometry_source='Freshly exported final Unreal render LOD0 of the bound target solids',
        overlap_fraction=max(0., 1-union/total))
    originals = {row['instance_id']: row for row in scene['instances']}
    for row in label['instances']:
        original = originals[row['instance_id']]
        row.update({key: original[key] for key in ('source_mesh', 'source_component_path',
                    'source_instance_index', 'local_to_world_m')})
    # Validate calibrated RGB independently before sealing the geometry/label receipt.
    plan = read_json(root / 'capture.json')
    if plan['scene_id'] != scene_id or proof['frames'] != len(plan['frames']):
        raise ValueError('Capture and solid scene identities or frame counts differ')
    error = proof.get('maximum_camera_error', 1)
    if not np.isfinite(error) or not 0 <= error <= 2e-5:
        raise ValueError('Solid render camera differs from the calibrated plan')
    frames = import_colmap(root)
    if len(frames) != len(plan['frames']):
        raise ValueError('Solid render camera export is incomplete')
    files = []
    for actual, expected in zip(frames, plan['frames']):
        if actual['name'] != expected['name'] or any(not np.allclose(actual[k], expected[k], atol=2e-5)
                for k in ('c2w_cv', 'K', 'width', 'height')):
            raise ValueError('Solid render calibration mismatch')
        with Image.open(child(root, actual['image'])) as image:
            if image.size != (actual['width'], actual['height']) or np.asarray(image.convert('RGB')).std() < 3:
                raise ValueError('Invalid or empty solid-region render')
        files.append(actual['image'])
    atomic_json(root / 'labels/volume.json', label)
    atomic_json(root / 'geometry/roundtrip_report.json', {'assets': checks,
        'max_relative_error': max(c['relative_roundtrip_error'] for c in checks)})
    preview = Image.new('RGB', (960, 720))
    for i, index in enumerate([0, 1, 3, len(frames)//2]):
        with Image.open(root / frames[index]['image']) as image:
            preview.paste(image.convert('RGB').resize((480, 360)), ((i%2)*480, (i//2)*360))
    preview.save(root / 'preview.jpg', quality=92)
    files += ['solid_scene.json', 'solid_assets.json', 'reference.json', 'capture.json', 'transforms.json',
        'sparse/0/cameras.txt', 'sparse/0/images.txt', 'SOLID_RENDER_FINISHED.json', 'source_content.json',
        'labels/volume.json', 'geometry/catalog.json', 'geometry/roundtrip_report.json', 'preview.jpg',
        'render_meshes/assets.review.json', 'render_meshes/assets.approved.json']
    files += ['geometry/'+a['mesh'] for a in catalog['assets']]
    files += ['render_meshes/'+s['mesh'] for s in report['assets']]
    receipt = {'schema': 'cropvolume-authored-volume-complete-v1', 'scene_id': scene_id,
        'source_region_id': scene['source_region_id'], 'target_plants': len(scene['instances']),
        'plant_solid_volume_m3': union, 'sum_instance_volume_m3': total, 'overlap_fraction': label['overlap_fraction'],
        'authored_transforms_preserved': True, 'non_target_field_preserved': True,
        'paired_gaussian_ply': False, 'files': {name: sha256(child(root, name)) for name in files}}
    atomic_json(root / 'VOLUME_COMPLETE.json', receipt)
    return receipt
