"""Read every authored field instance; no gameplay, new actors, or asset writes."""
import hashlib
import json
import os
from pathlib import Path

import unreal


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def xyz(v):
    return [v.x, v.y, v.z]


def transform(t):
    p, q, s = t.translation, t.rotation, t.scale3d
    # Column-vector affine matrix, in the original UE local/world centimetres.
    columns = []
    for unit in (unreal.Vector(1, 0, 0), unreal.Vector(0, 1, 0), unreal.Vector(0, 0, 1)):
        v = t.transform_location(unit) - p
        columns.append(xyz(v))
    matrix = [[columns[c][r] for c in range(3)] + [xyz(p)[r]] for r in range(3)] + [[0., 0., 0., 1.]]
    return {'local_to_world_ue_cm': matrix, 'translation_ue_cm': xyz(p),
            'rotation_quat_xyzw': [q.x, q.y, q.z, q.w], 'scale': xyz(s)}


def snapshot(actors):
    result = hashlib.sha256()
    for actor in sorted(actors, key=lambda a: a.get_path_name()):
        result.update(canonical([actor.get_path_name(), transform(actor.get_actor_transform())]))
        for c in sorted(actor.get_components_by_class(unreal.StaticMeshComponent), key=lambda c: c.get_name()):
            mesh = c.get_editor_property('static_mesh')
            if not mesh:
                continue
            result.update(canonical([c.get_path_name(), mesh.get_path_name(), transform(c.get_world_transform()),
                                    [m.get_path_name() if m else None for m in c.get_materials()]]))
            if isinstance(c, unreal.InstancedStaticMeshComponent):
                for i in range(c.get_instance_count()):
                    result.update(canonical([i, transform(c.get_instance_transform(i, True))]))
        for light in actor.get_components_by_class(unreal.LightComponent):
            result.update(canonical([light.get_path_name(), light.get_editor_property('intensity'),
                                     light.get_editor_property('light_color').export_text()]))
        if isinstance(actor, unreal.PostProcessVolume):
            result.update(actor.get_editor_property('settings').export_text().encode())
    return result.hexdigest()


def main(root):
    for name in ('EXPORT_FINISHED.json', 'INVENTORY_COMPLETE.json'):
        if (root / name).exists():
            raise RuntimeError('Use a fresh inventory output directory')
    request = json.loads((root / 'export_request.json').read_text())
    if not unreal.EditorLevelLibrary.load_level(request['source_map']):
        raise RuntimeError('Cannot load the original field map')
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    before = snapshot(actors)
    assets, components, count = {}, [], 0
    temporary = root / 'instances.jsonl.tmp'
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        for actor in sorted(actors, key=lambda a: a.get_path_name()):
            for c in sorted(actor.get_components_by_class(unreal.StaticMeshComponent), key=lambda c: c.get_name()):
                mesh = c.get_editor_property('static_mesh')
                if not mesh:
                    continue
                mesh_path = mesh.get_path_name()
                if mesh_path not in assets:
                    box = mesh.get_bounding_box()
                    relative = mesh_path.split('.')[0].removeprefix('/Game/') + '.uasset'
                    package = Path(unreal.Paths.project_content_dir()) / relative
                    assets[mesh_path] = {'bounds_local_ue_cm': [xyz(box.min), xyz(box.max)],
                        'role': request['mesh_roles'].get(mesh_path, 'scenery'),
                        'package_sha256': digest(package) if mesh_path.startswith('/Game/') else None}
                component_id = hashlib.sha256(canonical([request['source_map_sha256'], c.get_path_name()])).hexdigest()[:24]
                instanced = isinstance(c, unreal.InstancedStaticMeshComponent)
                size = c.get_instance_count() if instanced else 1
                components.append({'component_id': component_id, 'actor_path': actor.get_path_name(),
                    'actor_label': actor.get_actor_label(), 'component_path': c.get_path_name(),
                    'component_class': c.get_class().get_name(), 'mesh': mesh_path,
                    'materials': [m.get_path_name() if m else None for m in c.get_materials()],
                    'instanced': instanced, 'count': size})
                for index in range(size):
                    row = {'instance_id': component_id + '_' + str(index), 'component_id': component_id,
                           'source_instance_index': index,
                           **transform(c.get_instance_transform(index, True) if instanced else c.get_world_transform())}
                    stream.write(canonical(row).decode() + '\n')
                    count += 1
    temporary.replace(root / 'instances.jsonl')
    after = snapshot(actors)
    current = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    if before != after or {a.get_path_name() for a in current} != {a.get_path_name() for a in actors}:
        raise RuntimeError('Authored field changed during export')
    data = {'schema': 'cropvolume-authored-field-inventory-v1', 'source_map': request['source_map'],
        'source_map_sha256': request['source_map_sha256'], 'reference_scene_id': request['reference_scene_id'],
        'export_request_sha256': digest(root / 'export_request.json'), 'assets': assets, 'components': components,
        'actor_count': len(actors), 'instance_count': count, 'instances_file': 'instances.jsonl',
        'instances_sha256': digest(root / 'instances.jsonl'), 'coordinate_frame': 'UE_cm_Z_up',
        'field_before_sha256': before, 'field_after_sha256': after, 'field_unchanged': True,
        'engine': unreal.SystemLibrary.get_engine_version(),
        'bounds_scope': 'Undisplaced source-mesh bounds; original material animation retained',
        'export_script_sha256': digest(Path(__file__))}
    data['inventory_id'] = hashlib.sha256(canonical(data)).hexdigest()
    write_json(root / 'inventory.json', data)
    write_json(root / 'EXPORT_FINISHED.json', {'inventory_id': data['inventory_id'],
        'inventory_sha256': digest(root / 'inventory.json'), 'instance_count': count})
    unreal.log('Exported %d existing field instances without changing the map' % count)


try:
    main(Path(os.environ['CROPVOLUME_REFERENCE_OUTPUT']))
except Exception as error:
    write_json(Path(os.environ['CROPVOLUME_REFERENCE_OUTPUT']) / 'EXPORT_FAILED.json', {'error': str(error)})
    raise
