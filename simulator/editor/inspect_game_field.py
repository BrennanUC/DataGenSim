"""Read the authored game's field transforms without editing its assets."""
import json
import os
from pathlib import Path
import unreal


def vector(v):
    return [v.x, v.y, v.z]


def transform(t):
    r = t.rotation.rotator()
    return {"translation_cm": vector(t.translation), "scale": vector(t.scale3d),
            "rotation_deg": {"pitch": r.pitch, "yaw": r.yaw, "roll": r.roll}}


unreal.EditorLevelLibrary.load_level('/Game/FlightSchool/Maps/BasilField')
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
rows = []
for actor in actors:
    components = actor.get_components_by_class(unreal.StaticMeshComponent)
    for c in components:
        mesh = c.get_editor_property('static_mesh')
        if not mesh:
            continue
        name = mesh.get_name()
        row = {"actor": actor.get_actor_label(), "mesh": mesh.get_path_name(), "component_class": c.get_class().get_name()}
        if isinstance(c, unreal.InstancedStaticMeshComponent):
            row['count'] = c.get_instance_count()
            row['examples'] = [transform(c.get_instance_transform(i, True)) for i in range(min(3, row['count']))]
        else:
            row['transform'] = transform(c.get_world_transform())
        if name.startswith(('basil_', 'BasilClump', 'SoilBed', 'Context_basil')) or 'Light' in name:
            rows.append(row)
r = unreal.Rotator(0, 45, 0)
report = {"map": '/Game/FlightSchool/Maps/BasilField', "components": rows,
          "positional_rotator_0_45_0": {"pitch": r.pitch, "yaw": r.yaw, "roll": r.roll}}
Path(os.environ['CROPVOLUME_FIELD_INSPECTION']).write_text(json.dumps(report, indent=2))
unreal.log('Inspected authored field: %d components' % len(rows))
