"""Capture the supplied map in the editor; never clear, move, replace or save its field."""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

import unreal

sys.path.insert(0, str(Path(os.environ.get('CROPVOLUME_REFERENCE_SCRIPT') or os.environ['CROPVOLUME_CAPTURE_SCRIPT']).parent))
from cropvolume_profiling import Span

CAPTURE_PROFILES = []


def tracked_span(stage, **kwargs):
    timer = Span(stage, **kwargs).start()
    CAPTURE_PROFILES.append(timer)
    return timer


def finish_profiles(error=None):
    for timer in reversed(CAPTURE_PROFILES):
        timer.finish(error)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def transform_bytes(transform):
    p, q, s = transform.translation, transform.rotation, transform.scale3d
    return struct.pack('<10d', p.x, p.y, p.z, q.x, q.y, q.z, q.w, s.x, s.y, s.z)


def field_fingerprint(actors):
    digest = hashlib.sha256()
    inventory = []
    for actor in sorted(actors, key=lambda a: a.get_path_name()):
        digest.update(actor.get_path_name().encode())
        digest.update(transform_bytes(actor.get_actor_transform()))
        for component in sorted(actor.get_components_by_class(unreal.StaticMeshComponent), key=lambda c: c.get_name()):
            mesh = component.get_editor_property('static_mesh')
            if not mesh:
                continue
            digest.update(component.get_name().encode())
            digest.update(mesh.get_path_name().encode())
            digest.update(transform_bytes(component.get_world_transform()))
            for material in component.get_materials():
                digest.update((material.get_path_name() if material else 'None').encode())
            count = component.get_instance_count() if isinstance(component, unreal.InstancedStaticMeshComponent) else 1
            if isinstance(component, unreal.InstancedStaticMeshComponent):
                for index in range(count):
                    digest.update(transform_bytes(component.get_instance_transform(index, True)))
            inventory.append({'actor': actor.get_actor_label(), 'component': component.get_name(),
                              'mesh': mesh.get_path_name(), 'count': count})
        for light in actor.get_components_by_class(unreal.LightComponent):
            digest.update(str(light.get_editor_property('intensity')).encode())
            digest.update(light.get_editor_property('light_color').export_text().encode())
        if isinstance(actor, unreal.PostProcessVolume):
            digest.update(actor.get_editor_property('settings').export_text().encode())
    digest.update(json.dumps(inventory, sort_keys=True).encode())
    return digest.hexdigest(), inventory


class ReferenceCapture:
    expected_policy = 'preserve_all_authored_content'
    completion_filename = 'REFERENCE_RENDER_FINISHED.json'

    def prepare_field(self):
        pass

    def extra_completion(self):
        return {}

    def __init__(self):
        self.root = Path(os.environ['CROPVOLUME_REFERENCE_OUTPUT'])
        self.profile_total = tracked_span('unreal_render', root=self.root)
        for marker in ('REFERENCE_COMPLETE.json', 'REFERENCE_RENDER_FINISHED.json', 'VOLUME_COMPLETE.json', 'SOLID_RENDER_FINISHED.json'):
            if (self.root / marker).exists():
                raise RuntimeError('Refusing to overwrite a finished reference capture')
        self.source = json.loads((self.root / 'reference.json').read_text())
        self.plan = json.loads((self.root / 'capture.json').read_text())
        if self.source['field_policy'] != self.expected_policy or self.plan['scene_id'] != self.source['scene_id']:
            raise RuntimeError('Invalid original-field reference plan')
        with Span('level_load', parent=self.profile_total):
            if not unreal.EditorLevelLibrary.load_level(self.source['source_map']):
                raise RuntimeError('Could not load the supplied field map')
        self.world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        self.actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        self.original_actors = self.actor_subsystem.get_all_level_actors()
        with Span('field_binding', parent=self.profile_total):
            self.prepare_field()
        self.original_actors = self.actor_subsystem.get_all_level_actors()
        self.before, self.inventory = field_fingerprint(self.original_actors)
        # Only a capture camera is added. The original field and environment are untouched.
        self.camera = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector())
        self.component = self.camera.get_component_by_class(unreal.SceneCaptureComponent2D)
        for key, value in {'capture_every_frame': False, 'capture_on_movement': False,
                           'always_persist_rendering_state': True, 'post_process_blend_weight': 0.0,
                           'capture_source': unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR}.items():
            self.component.set_editor_property(key, value)
        frame = self.plan['frames'][0]
        self.target = unreal.RenderingLibrary.create_render_target2d(
            self.world, frame['width'], frame['height'], unreal.TextureRenderTargetFormat.RTF_RGBA8)
        self.component.set_editor_property('texture_target', self.target)
        self.index, self.maximum_camera_error = 0, 0.
        self.pose()
        self.settle = 180
        self.profile_warmup = tracked_span('render_warmup', parent=self.profile_total)
        self.profile_views = None
        self.callback = unreal.register_slate_post_tick_callback(self.tick)

    def pose(self):
        frame = self.plan['frames'][self.index]
        m, k = frame['c2w_cv'], frame['K']
        if (frame['width'] != self.target.size_x or frame['height'] != self.target.size_y
                or abs(k[0][2] - frame['width'] / 2) > 1e-6
                or abs(k[1][2] - frame['height'] / 2) > 1e-6 or abs(k[0][0] - k[1][1]) > 1e-6):
            raise RuntimeError('Reference capture requires fixed dimensions and centered square-pixel intrinsics')
        self.camera.set_actor_location(unreal.Vector(m[1][3]*100, m[0][3]*100, m[2][3]*100), False, False)
        forward = unreal.Vector(m[1][2], m[0][2], m[2][2])
        up = unreal.Vector(-m[1][1], -m[0][1], -m[2][1])
        self.camera.set_actor_rotation(unreal.MathLibrary.make_rot_from_xz(forward, up), False)
        self.component.set_editor_property('fov_angle', math.degrees(2*math.atan(frame['width']/(2*k[0][0]))))
        p = self.component.get_world_location()
        f, r, u = self.component.get_forward_vector(), self.component.get_right_vector(), self.component.get_up_vector()
        actual = [[r.y, -u.y, f.y, p.y*.01], [r.x, -u.x, f.x, p.x*.01],
                  [r.z, -u.z, f.z, p.z*.01], [0., 0., 0., 1.]]
        error = max(abs(actual[row][col]-m[row][col]) for row in range(4) for col in range(4))
        self.maximum_camera_error = max(self.maximum_camera_error, error)
        if error > 2e-5:
            raise RuntimeError('Actual reference camera differs from calibrated plan')
        self.settle = 16

    def stop(self, error=None):
        finish_profiles(error)
        unreal.unregister_slate_post_tick_callback(self.callback)
        unreal.EditorPythonScripting.set_keep_python_script_alive(False)

    def tick(self, delta):
        try:
            self.component.capture_scene()
            self.settle -= 1
            if self.settle > 0:
                return
            if self.profile_views is None:
                self.profile_warmup.finish()
                self.profile_views = tracked_span('render_views', parent=self.profile_total,
                                                  total=len(self.plan['frames']), unit='views')
            frame = self.plan['frames'][self.index]
            path = self.root / frame['image']
            path.parent.mkdir(exist_ok=True)
            unreal.RenderingLibrary.export_render_target(self.world, self.target, str(path.parent), path.name)
            if not path.exists():
                raise RuntimeError('Reference PNG was not written')
            self.index += 1
            self.profile_views.update(current=self.index)
            if self.index < len(self.plan['frames']):
                self.pose()
                return
            self.profile_views.finish()
            with Span('render_consistency', parent=self.profile_total):
                after, _ = field_fingerprint(self.original_actors)
            current = self.actor_subsystem.get_all_level_actors()
            if after != self.before or set(a.get_path_name() for a in current) != set(a.get_path_name() for a in [*self.original_actors, self.camera]):
                raise RuntimeError('Original field actors or transforms changed during capture')
            write_json(self.root / self.completion_filename, {
                'scene_id': self.source['scene_id'], 'source_map': self.source['source_map'],
                'frames': self.index, 'backend': 'unreal_existing_field_reference',
                'engine': unreal.SystemLibrary.get_engine_version(), 'field_unchanged': True,
                'field_before_sha256': self.before, 'field_after_sha256': after,
                'original_actor_count': len(self.original_actors), 'added_actors': ['SceneCapture2D'],
                'inventory': self.inventory, 'maximum_camera_error': self.maximum_camera_error,
                'material_animation': 'Original editor-time behavior retained; static geometry not certified',
                'capture_script_sha256': hashlib.sha256(Path(os.environ['CROPVOLUME_REFERENCE_SCRIPT']).read_bytes()).hexdigest(),
                **self.extra_completion()})
            self.stop()
        except Exception as error:
            write_json(self.root / 'REFERENCE_RENDER_FAILED.json', {'error': str(error), 'frame': self.index})
            unreal.log_error(str(error))
            self.stop(error)


if __name__ == '__main__':
    try:
        unreal.EditorPythonScripting.set_keep_python_script_alive(True)
        CROPVOLUME_REFERENCE = ReferenceCapture()
    except Exception as error:
        finish_profiles(error)
        write_json(Path(os.environ['CROPVOLUME_REFERENCE_OUTPUT']) / 'REFERENCE_RENDER_FAILED.json', {'error': str(error)})
        unreal.EditorPythonScripting.set_keep_python_script_alive(False)
        raise
