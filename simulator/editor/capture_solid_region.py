"""Render versioned solids only at selected authored plants' exact native transforms."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import unreal

SCRIPT_PATH = Path(os.environ['CROPVOLUME_REFERENCE_SCRIPT'])
sys.path.insert(0, str(SCRIPT_PATH.parent))
from capture_field_reference import ReferenceCapture, transform_bytes, write_json, finish_profiles
from cropvolume_profiling import span


def native_enu_matrix(t):
    p = t.translation
    # ENU local axes map to UE Y,X,Z. Column vectors retain tilt and all scales.
    cols = [t.transform_location(v)-p for v in (unreal.Vector(0,1,0), unreal.Vector(1,0,0), unreal.Vector(0,0,1))]
    return [[c.y for c in cols]+[p.y*.01], [c.x for c in cols]+[p.x*.01],
            [c.z for c in cols]+[p.z*.01], [0.,0.,0.,1.]]


def hash_remaining(components, excluded):
    result = hashlib.sha256()
    for path, c in sorted(components.items()):
        result.update(path.encode())
        if isinstance(c, unreal.InstancedStaticMeshComponent):
            entries = [hashlib.sha256(transform_bytes(c.get_instance_transform(i, True))).digest()
                       for i in range(c.get_instance_count()) if i not in excluded.get(path, set())]
            for entry in sorted(entries):
                result.update(entry)
        else:
            result.update(transform_bytes(c.get_world_transform()))
    return result.hexdigest()


class SolidRegionCapture(ReferenceCapture):
    expected_policy = 'preserve_authored_placement_with_versioned_target_solids'
    completion_filename = 'SOLID_RENDER_FINISHED.json'

    def prepare_field(self):
        scene = json.loads((self.root / 'solid_scene.json').read_text())
        if scene['scene_id'] != self.source['scene_id']:
            raise RuntimeError('Solid scene and camera identity differ')
        self.scene = scene
        components = {c.get_path_name(): c for a in self.original_actors
                      for c in a.get_components_by_class(unreal.StaticMeshComponent)}
        excluded, transforms = {}, {}
        for row in scene['instances']:
            c = components[row['source_component_path']]
            if c.get_editor_property('static_mesh').get_path_name() != row['source_mesh']:
                raise RuntimeError('Original source mesh does not match the selected plant')
            i = row['source_instance_index']
            t = c.get_instance_transform(i, True)
            actual = native_enu_matrix(t)
            if max(abs(actual[r][col]-row['local_to_world_m'][r][col]) for r in range(4) for col in range(4)) > 2e-8:
                raise RuntimeError('Source instance no longer matches its authored transform')
            transforms[row['instance_id']] = t
            excluded.setdefault(row['source_component_path'], set()).add(i)
        before = hash_remaining(components, excluded)
        assets = {a['id']: a for a in scene['assets']}
        # Audit the exact render LOD, material opacity and shader displacement in this runtime.
        os.environ['CROPVOLUME_AUDIT_DIR'] = str(self.root / 'render_meshes')
        os.environ['CROPVOLUME_ASSET_LIST'] = str(self.root / 'solid_assets.json')
        os.environ['CROPVOLUME_AUDIT_LOD'] = 'render'
        os.environ.pop('CROPVOLUME_EXPORT_UVS', None)
        with span('render_mesh_export'):
            runpy.run_path(str(SCRIPT_PATH.with_name('export_audit.py')))
        for path, indices in excluded.items():
            for i in sorted(indices, reverse=True):
                if not components[path].remove_instance(i):
                    raise RuntimeError('Could not remove the selected original surface instance')
        after = hash_remaining(components, {})
        if before != after:
            raise RuntimeError('A non-target field instance moved or changed during solid binding')
        self.target_proof = []
        for row in scene['instances']:
            asset = assets[row['asset_id']]
            package = Path(unreal.Paths.project_content_dir()) / (asset['unreal_asset'].split('.')[0].removeprefix('/Game/')+'.uasset')
            if hashlib.sha256(package.read_bytes()).hexdigest() != asset['unreal_uasset_sha256']:
                raise RuntimeError('Solid asset differs from its audited catalog')
            mesh = unreal.load_asset(asset['unreal_asset'])
            if unreal.EditorAssetLibrary.get_metadata_tag(mesh, 'CropVolumeGeometryID') != asset['solid_recipe']['geometry_id']:
                raise RuntimeError('Solid geometry version differs from the requested recipe')
            actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector())
            actor.static_mesh_component.set_static_mesh(mesh)
            actor.static_mesh_component.set_forced_lod_model(1)
            actor.set_actor_transform(transforms[row['instance_id']], False, True)
            actual = native_enu_matrix(actor.get_actor_transform())
            error = max(abs(actual[r][col]-row['local_to_world_m'][r][col]) for r in range(4) for col in range(4))
            if error > 2e-8:
                raise RuntimeError('Solid replacement changed the original plant transform')
            self.target_proof.append({'instance_id': row['instance_id'], 'asset_id': row['asset_id'],
                'source_component_path': row['source_component_path'], 'source_instance_index': row['source_instance_index'],
                'local_to_world_m': actual, 'transform_error': error,
                'forced_lod': int(actor.static_mesh_component.get_editor_property('forced_lod_model'))-1,
                'material_paths': [m.get_path_name() if m else None for m in actor.static_mesh_component.get_materials()]})
        self.binding_proof = {'non_target_before_sha256': before, 'non_target_after_sha256': after,
                              'target_count': len(self.target_proof), 'all_authored_transforms_preserved': True}

    def extra_completion(self):
        return {'solid_scene_id': self.scene['scene_id'], 'target_instances': self.target_proof,
                'binding': self.binding_proof, 'material_animation': 'Audited target solids are static; original context materials retained',
                'capture_helper_sha256': hashlib.sha256(SCRIPT_PATH.with_name('capture_field_reference.py').read_bytes()).hexdigest()}


try:
    unreal.EditorPythonScripting.set_keep_python_script_alive(True)
    CROPVOLUME_SOLID_REGION = SolidRegionCapture()
except Exception as error:
    finish_profiles(error)
    write_json(Path(os.environ['CROPVOLUME_REFERENCE_OUTPUT']) / 'REFERENCE_RENDER_FAILED.json', {'error': str(error)})
    unreal.EditorPythonScripting.set_keep_python_script_alive(False)
    raise
