"""Render seeded targets with calibrated cameras in an editor-only working copy.

Run in the editor's Python console (not a NullRHI commandlet):
  set CROPVOLUME_SAMPLE and CROPVOLUME_CATALOG to Windows paths, then execute this file.
New BaseSim captures reuse authored beds inside a bounded field with a black exterior.
The dedicated generated-map path is retained for earlier immutable run configs.
After RENDER_FINISHED.json appears, the CPU seal-capture command independently
validates image dimensions, camera formats, scene identity and checksums.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import unreal

sys.path.insert(0, str(Path(os.environ['CROPVOLUME_CAPTURE_SCRIPT']).parent))
from cropvolume_profiling import Span

PROFILES = []


def tracked_span(stage, **kwargs):
    timer = Span(stage, **kwargs).start()
    PROFILES.append(timer)
    return timer


def finish_profiles(error=None):
    for timer in reversed(PROFILES):
        timer.finish(error)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_enu_matrix(t):
    p = t.translation
    cols = [t.transform_location(v)-p for v in (unreal.Vector(0,1,0),unreal.Vector(1,0,0),unreal.Vector(0,0,1))]
    return [[c.y for c in cols]+[p.y*.01],[c.x for c in cols]+[p.x*.01],
            [c.z for c in cols]+[p.z*.01],[0.,0.,0.,1.]]


class Capture:
    def __init__(self):
        self.root = Path(os.environ["CROPVOLUME_SAMPLE"])
        self.profile_total = tracked_span('unreal_render', root=self.root)
        self.profile_setup = tracked_span('field_binding', parent=self.profile_total)
        scene = json.loads((self.root / "scene.json").read_text())
        self.scene = scene
        self.basesim = scene.get('field', {}).get('backend') == 'unreal_basesim_editor'
        self.bounded = self.basesim and scene['field'].get('context_policy') == 'bounded_field_black_exterior_v1'
        self.capture_components = []
        self.environment_before = None
        self.maximum_plant_error = 0.
        catalog = json.loads(Path(os.environ["CROPVOLUME_CATALOG"]).read_text())
        if catalog["catalog_id"] != scene["catalog_id"] or scene.get("test_fixture"):
            raise RuntimeError("Catalog mismatch or test-only scene")
        if (self.root / "CAPTURE_COMPLETE.json").exists():
            raise RuntimeError("Refusing to overwrite a sealed capture")
        (self.root / "RENDER_FAILED.json").unlink(missing_ok=True)
        (self.root / "RENDER_FINISHED.json").unlink(missing_ok=True)
        self.exposure_ev100 = scene["appearance"].get("exposure_ev100")
        self.script_sha256 = digest(os.environ["CROPVOLUME_CAPTURE_SCRIPT"])
        self.plan = json.loads((self.root / "capture.json").read_text())
        assets = {a["id"]: a for a in catalog["assets"]}
        # Inspect all assets before touching the editor world.
        for asset_id in sorted({instance['asset_id'] for instance in scene['instances']}):
            asset = assets[asset_id]
            if not asset["approved"] or not asset.get("unreal_asset") or not asset.get("unreal_uasset_sha256"):
                raise RuntimeError("Rendering requires approved, editor-exported geometry")
            package_file = Path(unreal.Paths.project_content_dir()) / (asset["unreal_asset"].split(".")[0].removeprefix("/Game/") + ".uasset")
            if digest(package_file) != asset["unreal_uasset_sha256"]:
                raise RuntimeError("Unreal plant asset changed after audit")
        map_path = scene['field']['source_map'] if self.basesim else "/Game/CropVolume/DatasetField"
        if self.basesim:
            if not unreal.EditorLevelLibrary.load_level(map_path):
                raise RuntimeError('Could not load the original BaseSim environment')
            if self.bounded:
                self.prepare_bounded_environment()
            else:
                self.prepare_basesim_environment()
        elif unreal.EditorAssetLibrary.does_asset_exist(map_path):
            unreal.EditorLevelLibrary.load_level(map_path)
            subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
            # This dedicated generated map must contain no persistent hand-authored actors.
            for actor in subsystem.get_all_level_actors():
                if "CropVolumeGenerated" in [str(tag) for tag in actor.tags]:
                    subsystem.destroy_actor(actor)
        else:
            unreal.EditorLevelLibrary.new_level(map_path)
        self.world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        for instance in scene["instances"]:
            if instance["role"] != "plant":
                continue
            asset = assets[instance["asset_id"]]
            matrix = instance["local_to_world_m"]
            # The CPU exporter swaps local UE X/Y to ENU; conjugation reverses yaw.
            actor = self.spawn(unreal.StaticMeshActor, unreal.Vector(matrix[1][3] * 100, matrix[0][3] * 100, matrix[2][3] * 100))
            actor.static_mesh_component.set_static_mesh(unreal.load_asset(asset["unreal_asset"]))
            actor.static_mesh_component.set_forced_lod_model(1)
            if self.bounded:
                self.capture_components.append(actor.static_mesh_component)
            actor.set_actor_rotation(unreal.Rotator(pitch=0, yaw=-math.degrees(instance["augmentation"]["yaw_rad"]), roll=0), False)
            scale = instance["augmentation"]["scale"]
            actor.set_actor_scale3d(unreal.Vector(scale[1], scale[0], scale[2]))
            actual = native_enu_matrix(actor.get_actor_transform())
            error = max(abs(actual[r][c]-matrix[r][c]) for r in range(4) for c in range(4))
            self.maximum_plant_error = max(self.maximum_plant_error, error)
            if error > 2e-5:
                raise RuntimeError('Rendered plant transform differs from its volume-label transform')
        if not self.basesim:
            ground = self.spawn(unreal.StaticMeshActor, unreal.Vector(0, 0, -1))
            ground.static_mesh_component.set_static_mesh(unreal.load_asset("/Engine/BasicShapes/Plane"))
            ground.static_mesh_component.set_material(0, unreal.load_asset("/Game/CROPSIM/Materials/M_Soil"))
            ground.set_actor_scale3d(unreal.Vector(max(4, scene['patch_m'][1]*3), max(4, scene['patch_m'][0]*3), 1))
            sun = self.spawn(unreal.DirectionalLight, unreal.Vector(0, 0, 500))
            sun.set_actor_rotation(unreal.Rotator(pitch=-55, yaw=-30, roll=0), False)
            sun.get_component_by_class(unreal.DirectionalLightComponent).set_intensity(10000)
            self.spawn(unreal.SkyLight, unreal.Vector(0, 0, 300))
        self.camera = self.spawn(unreal.SceneCapture2D, unreal.Vector())
        self.component = self.camera.get_component_by_class(unreal.SceneCaptureComponent2D)
        self.component.set_editor_property("capture_every_frame", False)
        self.component.set_editor_property("capture_on_movement", False)
        self.component.set_editor_property("always_persist_rendering_state", True)
        self.component.set_editor_property("post_process_blend_weight", 0. if self.basesim else 1.)
        self.component.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
        if not self.basesim:
            settings = self.component.get_editor_property("post_process_settings")
            for key, value in {"override_auto_exposure_min_brightness": True, "override_auto_exposure_max_brightness": True,
                               "auto_exposure_min_brightness": self.exposure_ev100,
                               "auto_exposure_max_brightness": self.exposure_ev100,
                               "override_motion_blur_amount": True, "motion_blur_amount": 0.}.items():
                settings.set_editor_property(key, value)
            self.component.set_editor_property("post_process_settings", settings)
        if self.bounded:
            self.component.set_editor_property('primitive_render_mode', unreal.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST)
            for mesh_component in self.capture_components:
                self.component.show_only_component(mesh_component)
            self.component.set_editor_property('show_flag_settings', [
                unreal.EngineShowFlagsSetting(show_flag_name=name, enabled=False)
                for name in ('Atmosphere','Fog','VolumetricFog','Cloud',
                             'Bloom','LensFlares','MotionBlur','Vignette')])
        self.target = None
        self.index, self.settle = 0, 0
        self.pose()
        self.profile_setup.finish()
        self.settle = 180  # Warm up the first scene before writing photographs.
        self.profile_warmup = tracked_span('render_warmup', parent=self.profile_total)
        self.profile_views = None
        self.callback = unreal.register_slate_post_tick_callback(self.tick)

    def clipped_material(self, source, suffix):
        """Clone the authored material and clip fragments to the ENU rectangle.

        Only runtime copies are modified; the source material's shading graph and
        instance parameters remain intact. No post-render image crop is involved.
        """
        destination = '/Game/CropVolume/Bounded/' + self.scene['scene_id'][:16] + '_' + suffix
        if unreal.EditorAssetLibrary.does_asset_exist(destination):
            material = unreal.load_asset(destination)
        else:
            material = unreal.EditorAssetLibrary.duplicate_asset(source.get_path_name(), destination)
        if isinstance(material, unreal.MaterialInstanceConstant):
            parent = self.clipped_material(source.get_editor_property('parent'), suffix + '_Parent')
            unreal.MaterialEditingLibrary.set_material_instance_parent(material, parent)
            return material
        edit = unreal.MaterialEditingLibrary
        def node(cls, **properties):
            expression = edit.create_material_expression(material, cls)
            for name, value in properties.items():
                expression.set_editor_property(name, value)
            return expression
        def link(a, b, pin):
            if not edit.connect_material_expressions(a, '', b, '' if pin == 'Input' else pin):
                raise RuntimeError('Cannot connect field clipping material: ' + pin)
        world = node(unreal.MaterialExpressionWorldPosition)
        masks = []
        # UE X is north, UE Y is east, with centimetre coordinates.
        for axis, center, size in [(0,self.scene['center_enu_m'][1],self.scene['patch_m'][1]),
                                   (1,self.scene['center_enu_m'][0],self.scene['patch_m'][0])]:
            channel = node(unreal.MaterialExpressionComponentMask, r=axis==0, g=axis==1)
            link(world,channel,'Input')
            delta = node(unreal.MaterialExpressionSubtract, const_b=center*100.)
            link(channel,delta,'A')
            absolute = node(unreal.MaterialExpressionAbs)
            link(delta,absolute,'Input')
            inside = node(unreal.MaterialExpressionSubtract, const_a=size*50.)
            link(absolute,inside,'B')
            sharp = node(unreal.MaterialExpressionMultiply, const_b=10000.)
            link(inside,sharp,'A')
            mask = node(unreal.MaterialExpressionSaturate)
            link(sharp,mask,'Input')
            masks.append(mask)
        product = node(unreal.MaterialExpressionMultiply)
        link(masks[0],product,'A'); link(masks[1],product,'B')
        material.set_editor_property('blend_mode', unreal.BlendMode.BLEND_MASKED)
        if not edit.connect_material_property(product, '', unreal.MaterialProperty.MP_OPACITY_MASK):
            raise RuntimeError('Cannot bind field clipping opacity mask')
        edit.recompile_material(material)
        return material

    def prepare_bounded_environment(self):
        from capture_field_reference import field_fingerprint
        self.field_fingerprint = field_fingerprint
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
        self.removed = {}
        self.excluded_plants = 0
        self.clipped_ground_components = 0
        self.original_soil_instances = 0
        for actor in actors:
            for component in actor.get_components_by_class(unreal.StaticMeshComponent):
                mesh = component.get_editor_property('static_mesh')
                if not mesh:
                    continue
                if mesh.get_path_name() in self.scene['field']['replace_meshes']:
                    self.excluded_plants += component.get_instance_count()
                    component.clear_instances()
                    if component.get_instance_count():
                        raise RuntimeError('Could not clear original context plants from the runtime field')
                    continue
                soil = mesh.get_path_name() == '/Game/CROPSIM/Meshes/SoilBed_4m.SoilBed_4m'
                ground = actor.get_actor_label() == 'CS_FarField'
                if not (soil or ground):
                    continue
                if soil:
                    self.original_soil_instances += component.get_instance_count()
                    box = mesh.get_bounding_box()
                    corners = [unreal.Vector(x,y,z) for x in (box.min.x,box.max.x)
                        for y in (box.min.y,box.max.y) for z in (box.min.z,box.max.z)]
                    east, north = self.scene['center_enu_m']
                    width, length = self.scene['patch_m']
                    for index in reversed(range(component.get_instance_count())):
                        transform = component.get_instance_transform(index,True)
                        points = [transform.transform_location(v) for v in corners]
                        outside = (max(v.y for v in points)*.01 < east-width*.5
                            or min(v.y for v in points)*.01 > east+width*.5
                            or max(v.x for v in points)*.01 < north-length*.5
                            or min(v.x for v in points)*.01 > north+length*.5)
                        if outside and not component.remove_instance(index):
                            raise RuntimeError('Could not trim exterior soil from the runtime field')
                    self.retained_soil_instances = component.get_instance_count()
                for slot, source in enumerate(component.get_materials()):
                    if not source:
                        raise RuntimeError('Authored ground material is missing')
                    component.set_material(slot, self.clipped_material(source, 'Ground_%d_%d' % (self.clipped_ground_components,slot)))
                self.capture_components.append(component)
                self.clipped_ground_components += 1
        if self.clipped_ground_components != 2 or not self.excluded_plants:
            raise RuntimeError('Expected original BaseSim soil, ground and plant components')
        self.environment_actors = actors
        self.environment_before = field_fingerprint(actors)[0]

    def prepare_basesim_environment(self):
        from capture_field_reference import field_fingerprint
        self.field_fingerprint = field_fingerprint
        field = self.scene['field']
        subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        actors = subsystem.get_all_level_actors()
        center = field['center_enu_m']
        width, length = self.scene['patch_m']
        region = [center[0]-width*.5,center[1]-length*.5,center[0]+width*.5,center[1]+length*.5]
        self.removed = {}
        for actor in actors:
            for component in actor.get_components_by_class(unreal.InstancedStaticMeshComponent):
                mesh = component.get_editor_property('static_mesh')
                if not mesh or mesh.get_path_name() not in field['replace_meshes']:
                    continue
                box = mesh.get_bounding_box()
                corners = [unreal.Vector(x,y,z) for x in (box.min.x,box.max.x) for y in (box.min.y,box.max.y) for z in (box.min.z,box.max.z)]
                radius = max(math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z) for v in corners)*.01
                selected = []
                for i in range(component.get_instance_count()):
                    t = component.get_instance_transform(i,True)
                    p = t.translation
                    reach = radius*max(abs(t.scale3d.x),abs(t.scale3d.y),abs(t.scale3d.z))
                    if p.y*.01+reach < region[0] or p.y*.01-reach > region[2] or p.x*.01+reach < region[1] or p.x*.01-reach > region[3]:
                        continue
                    points = [t.transform_location(v) for v in corners]
                    if max(v.y for v in points)*.01 < region[0] or min(v.y for v in points)*.01 > region[2] or max(v.x for v in points)*.01 < region[1] or min(v.x for v in points)*.01 > region[3]:
                        continue
                    selected.append(i)
                for i in reversed(selected):
                    if not component.remove_instance(i):
                        raise RuntimeError('Could not remove an original plant inside the randomized region')
                self.removed[mesh.get_path_name()] = self.removed.get(mesh.get_path_name(),0)+len(selected)
        self.environment_actors = actors
        self.environment_before = field_fingerprint(actors)[0]

    def spawn(self, cls, location):
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(cls, location)
        actor.tags = ["CropVolumeGenerated"]
        return actor

    def pose(self):
        frame = self.plan["frames"][self.index]
        m = frame["c2w_cv"]
        self.camera.set_actor_location(unreal.Vector(m[1][3] * 100, m[0][3] * 100, m[2][3] * 100), False, False)
        forward = unreal.Vector(m[1][2], m[0][2], m[2][2])
        up = unreal.Vector(-m[1][1], -m[0][1], -m[2][1])
        self.camera.set_actor_rotation(unreal.MathLibrary.make_rot_from_xz(forward, up), False)
        if self.target is None:
            self.target = unreal.RenderingLibrary.create_render_target2d(self.world, frame["width"], frame["height"], unreal.TextureRenderTargetFormat.RTF_RGBA8)
            if self.bounded:
                self.target.set_editor_property('clear_color', unreal.LinearColor(0,0,0,1))
            self.component.set_editor_property("texture_target", self.target)
        K = frame["K"]
        if abs(K[0][2] - frame["width"] / 2) > 1e-6 or abs(K[1][2] - frame["height"] / 2) > 1e-6:
            raise RuntimeError("Prototype producer requires centered intrinsics")
        self.component.set_editor_property("fov_angle", math.degrees(2 * math.atan(frame["width"] / (2 * K[0][0]))))
        actual_forward = self.component.get_forward_vector()
        actual_right = self.component.get_right_vector()
        actual_up = self.component.get_up_vector()
        actual_position = self.component.get_world_location()
        cv = [[actual_right.y, -actual_up.y, actual_forward.y, actual_position.y * .01],
              [actual_right.x, -actual_up.x, actual_forward.x, actual_position.x * .01],
              [actual_right.z, -actual_up.z, actual_forward.z, actual_position.z * .01], [0., 0., 0., 1.]]
        if max(abs(cv[r][c] - m[r][c]) for r in range(4) for c in range(4)) > 2e-5:
            raise RuntimeError("Editor capture pose does not match the calibrated plan")
        self.settle = 8

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
            frame = self.plan["frames"][self.index]
            path = self.root / frame["image"]
            path.parent.mkdir(exist_ok=True)
            unreal.RenderingLibrary.export_render_target(self.world, self.target, str(path.parent), path.name)
            if not path.exists():
                raise RuntimeError("Expected PNG render was not written: " + str(path))
            self.index += 1
            self.profile_views.update(current=self.index)
            if self.index < len(self.plan["frames"]):
                self.pose()
            else:
                unreal.unregister_slate_post_tick_callback(self.callback)
                proof = {}
                if self.basesim:
                    after = self.field_fingerprint(self.environment_actors)[0]
                    if after != self.environment_before:
                        raise RuntimeError('The retained BaseSim environment changed during capture')
                    proof = {'source_map':self.scene['field']['source_map'], 'environment_before_sha256':self.environment_before,
                        'environment_after_sha256':after, 'original_plants_replaced':self.removed,
                        'appearance_policy':self.scene['field']['appearance_policy'],
                        'all_plant_transforms_verified':True,'maximum_plant_transform_error':self.maximum_plant_error,
                        'target_plant_count':len(self.scene['instances'])}
                if self.bounded:
                    proof.update({'context_policy':self.scene['field']['context_policy'],
                        'exterior':'black', 'primitive_render_mode':'show_only_components',
                        'excluded_original_plant_instances':self.excluded_plants,
                        'clipped_ground_components':self.clipped_ground_components,
                        'original_soil_instances':self.original_soil_instances,
                        'retained_soil_instances':self.retained_soil_instances,
                        'shown_components':len(self.capture_components),
                        'region_center_enu_m':self.scene['center_enu_m'], 'region_size_m':self.scene['patch_m']})
                (self.root / "RENDER_FINISHED.json").write_text(json.dumps({"scene_id": self.plan["scene_id"], "frames": self.index,
                    "backend": "unreal_basesim_random" if self.basesim else "unreal_editor", "engine": unreal.SystemLibrary.get_engine_version(),
                    "sun_intensity_lux": None if self.basesim else 10000,
                    "exposure_ev100": self.exposure_ev100, "capture_script_sha256": self.script_sha256, **proof}))
                unreal.log("RGB rendering finished. Run CPU seal-capture for independent validation.")
                finish_profiles()
                unreal.EditorPythonScripting.set_keep_python_script_alive(False)
        except Exception as error:
            finish_profiles(error)
            unreal.unregister_slate_post_tick_callback(self.callback)
            (self.root / "RENDER_FAILED.json").write_text(json.dumps({"error": str(error), "frame": self.index}))
            unreal.log_error(str(error))
            unreal.EditorPythonScripting.set_keep_python_script_alive(False)


# Keep the Python producer alive while the editor dispatches Slate tick callbacks.
try:
    unreal.EditorPythonScripting.set_keep_python_script_alive(True)
    CROPVOLUME_CAPTURE = Capture()
except Exception as error:
    finish_profiles(error)
    unreal.EditorPythonScripting.set_keep_python_script_alive(False)
    raise
