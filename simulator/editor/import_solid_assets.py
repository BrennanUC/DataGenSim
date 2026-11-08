"""Import explicitly thickened geometry as new static assets; preserve original plants."""
import json
import os
from pathlib import Path
import unreal


def static_material(source_path, destination, has_uv=False):
    if unreal.EditorAssetLibrary.does_asset_exist(destination):
        return unreal.load_asset(destination)
    material = unreal.EditorAssetLibrary.duplicate_asset(source_path, destination)
    if isinstance(material, unreal.MaterialInstanceConstant):
        source = unreal.load_asset(source_path)
        parent = static_material(source.get_editor_property("parent").get_path_name(), destination + "_Parent", has_uv)
        unreal.MaterialEditingLibrary.set_material_instance_parent(material, parent)
        unreal.EditorAssetLibrary.save_loaded_asset(material)
        return material
    if unreal.MaterialEditingLibrary.get_used_textures(material) and not has_uv:
        raise RuntimeError("Solid import needs a UV-preserving path for textured plant materials")
    zero = unreal.MaterialEditingLibrary.create_material_expression(material, unreal.MaterialExpressionConstant)
    zero.set_editor_property("r", 0.)
    unreal.MaterialEditingLibrary.connect_material_property(zero, "", unreal.MaterialProperty.MP_WORLD_POSITION_OFFSET)
    unreal.MaterialEditingLibrary.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material)
    return material


def run():
    root = Path(os.environ["CROPVOLUME_SOLID_SOURCE"])
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    for spec in manifest["assets"]:
        geometry = json.loads((root / spec["geometry"]).read_text())
        materials = [static_material(p, "/Game/CropVolume/Materials/" + p.rsplit('/', 1)[-1].split('.')[0] + "_Static",
                                     "triangle_uv0" in geometry) for p in spec["source_materials"]]
        name = spec["id"] + spec.get("asset_suffix", "_solid")
        path = "/Game/CropVolume/Plants/" + name
        if unreal.EditorAssetLibrary.does_asset_exist(path):
            mesh = unreal.load_asset(path)
            if unreal.EditorAssetLibrary.get_metadata_tag(mesh, "CropVolumeGeometryID") != geometry["geometry_id"]:
                raise RuntimeError("Existing solid mesh differs; use a new geometry version/path")
        else:
            buffers = unreal.GeometryScriptSimpleMeshBuffers()
            if "triangle_uv0" in geometry:
                indices = [v for t in geometry["triangles"] for v in t]
                vertices = [geometry["vertices_m"][i] for i in indices]
                buffers.vertices = [unreal.Vector(v[1] * 100, v[0] * 100, v[2] * 100) for v in vertices]
                buffers.uv0 = [unreal.Vector2D(*uv) for t in geometry["triangle_uv0"] for uv in t]
                normals = [geometry["vertex_normals"][i] for i in indices]
                buffers.normals = [unreal.Vector(n[1], n[0], n[2]) for n in normals]
                if "triangle_colors" in geometry:
                    buffers.vertex_colors = [unreal.LinearColor(*c) for t in geometry["triangle_colors"] for c in t]
                buffers.triangles = [unreal.IntVector(3*i, 3*i+1, 3*i+2) for i in range(len(geometry["triangles"]))]
            else:
                buffers.vertices = [unreal.Vector(v[1] * 100, v[0] * 100, v[2] * 100) for v in geometry["vertices_m"]]
            # UE requires a UV channel even for these texture-free materials.
            # It does not affect geometry or the volume label.
                buffers.uv0 = [unreal.Vector2D(v[0], v[1]) for v in geometry["vertices_m"]]
                buffers.triangles = [unreal.IntVector(*t) for t in geometry["triangles"]]
            dynamic = unreal.DynamicMesh()
            dynamic, triangles = dynamic.append_buffers_to_mesh(buffers)
            ids = triangles.convert_index_list_to_array()
            if len(ids) != len(geometry["triangles"]):
                raise RuntimeError("Some solid triangles were rejected by Unreal")
            for triangle, material in zip(ids, geometry["material_ids"]):
                dynamic.set_triangle_material_id(triangle, material, True)
            options = unreal.GeometryScriptCreateNewStaticMeshAssetOptions(
                enable_recompute_normals="triangle_uv0" not in geometry, enable_recompute_tangents=True,
                enable_nanite=False, enable_collision=False)
            mesh, outcome = unreal.GeometryScript_NewAssetUtils.create_new_static_mesh_asset_from_mesh(dynamic, path, options)
            if outcome != unreal.GeometryScriptOutcomePins.SUCCESS:
                raise RuntimeError("Solid asset creation failed: " + path)
            mesh.set_editor_property("static_materials", [unreal.StaticMaterial(material_interface=m, material_slot_name=str(i))
                                                         for i, m in enumerate(materials)])
            unreal.EditorAssetLibrary.set_metadata_tag(mesh, "CropVolumeGeometryID", geometry["geometry_id"])
            unreal.EditorAssetLibrary.set_metadata_tag(mesh, "CropVolumeLeafThicknessM", str(geometry["leaf_thickness_m"]))
            unreal.EditorAssetLibrary.save_loaded_asset(mesh)
        rows.append({"id": name, "species": spec.get("species", spec["id"].split("_")[0]), "unreal_asset": path,
                     "geometry_id": geometry["geometry_id"], "leaf_thickness_m": geometry["leaf_thickness_m"],
                     "expected_volume_m3": geometry["plant_solid_volume_m3"]})
    output = Path(os.environ.get("CROPVOLUME_IMPORT_MANIFEST", str(Path(os.environ["CROPVOLUME_DATASET_ROOT"]) / "assets/solid_imported.json")))
    output.write_text(json.dumps({"assets": rows, "status": "requires_roundtrip_volume_validation"}, indent=2))
    unreal.log("Imported %d solid plant assets" % len(rows))


run()
