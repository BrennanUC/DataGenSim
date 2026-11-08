"""Unreal editor script: export LOD0 render triangles; never auto-approve materials.

Requires PythonScriptPlugin + GeometryScripting. Run in the working project,
with CROPVOLUME_AUDIT_DIR set to a Windows folder on E:. Not executed in CPU tests.
"""
import json
import os
from pathlib import Path
import re
import hashlib
import unreal


def run():
    out = Path(os.environ["CROPVOLUME_AUDIT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    requested = os.environ.get("CROPVOLUME_ASSET_LIST")
    specifications = json.loads(Path(requested).read_text())["assets"] if requested else None
    paths = [s["unreal_asset"] for s in specifications] if specifications else unreal.EditorAssetLibrary.list_assets("/Game/CROPSIM/Meshes", recursive=False)
    for asset_path in paths:
        name = asset_path.rsplit("/", 1)[-1].split(".")[0]
        match = re.fullmatch(r"(basil|cilantro)_s\d+_v\d+", name)
        if not match and not specifications:
            continue
        mesh = unreal.load_asset(asset_path)
        dynamic = unreal.DynamicMesh()
        mode = os.environ.get("CROPVOLUME_AUDIT_LOD", "source")
        lod_type = unreal.GeometryScriptLODType.SOURCE_MODEL if mode == "source" else unreal.GeometryScriptLODType.RENDER_DATA
        lod = unreal.GeometryScriptMeshReadLOD(lod_type=lod_type, lod_index=0)
        dynamic, outcome = unreal.GeometryScript_AssetUtils.copy_mesh_from_static_mesh(
            mesh, dynamic, unreal.GeometryScriptCopyMeshFromAssetOptions(), lod)
        if outcome != unreal.GeometryScriptOutcomePins.SUCCESS:
            raise RuntimeError("Render geometry export failed: " + asset_path)
        _, positions, _ = dynamic.get_all_vertex_positions(False)
        vertices = positions.convert_vector_list_to_array()
        _, triangles, _ = dynamic.get_all_triangle_indices(False)
        faces = [(i, t) for i, t in enumerate(triangles.convert_triangle_list_to_array()) if min(t.x, t.y, t.z) >= 0]
        path = out / (name + ".ply")
        with path.open("w", encoding="ascii", newline="\n") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nproperty int material_id\nend_header\n")
            # UE render triangles use a left-handed winding convention. Swapping
            # X/Y yields right-handed ENU; retain indices (do not flip twice).
            for v in vertices:
                f.write(f"{v.y * .01:.12g} {v.x * .01:.12g} {v.z * .01:.12g}\n")
            for triangle_id, t in faces:
                material_id, valid = dynamic.get_triangle_material_id(triangle_id)
                f.write(f"3 {t.x} {t.y} {t.z} {material_id if valid else -1}\n")
        materials = [slot.material_interface.get_path_name() if slot.material_interface else None for slot in mesh.static_materials]
        uv_path = None
        if os.environ.get("CROPVOLUME_EXPORT_UVS") == "1":
            uv_rows = []
            color_rows = []
            query = next(getattr(unreal.GeometryScript_MeshQueries, n) for n in dir(unreal.GeometryScript_MeshQueries)
                         if n.replace("_", "").lower() == "gettriangleuvs")
            for triangle_id, _ in faces:
                a, b, c, valid = query(dynamic, 0, triangle_id)
                if not valid:
                    raise RuntimeError("Source UV0 is missing on a triangle")
                uv_rows.append([[v.x, v.y] for v in (a, b, c)])
                _, ca, cb, cc, colors_valid = unreal.GeometryScript_MeshQueries.get_triangle_vertex_colors(dynamic, triangle_id)
                color_rows.append([[v.r, v.g, v.b, v.a] for v in (ca, cb, cc)] if colors_valid else [[1.,1.,1.,1.]]*3)
            uv_path = out / (name + ".uv0.json")
            uv_path.write_text(json.dumps(uv_rows))
            (out / (name + ".color0.json")).write_text(json.dumps(color_rows))
        material_review = []
        for material_path in materials:
            entry = {"material": material_path}
            try:
                material = unreal.load_asset(material_path)
                overrides = []
                while isinstance(material, unreal.MaterialInstanceConstant):
                    overrides.append(material.get_editor_property("base_property_overrides"))
                    material = material.get_editor_property("parent")
                blend = material.get_editor_property("blend_mode")
                sided = material.get_editor_property("two_sided")
                for override in reversed(overrides):
                    if override.get_editor_property("override_blend_mode"):
                        blend = override.get_editor_property("blend_mode")
                    if override.get_editor_property("override_two_sided"):
                        sided = override.get_editor_property("two_sided")
                entry["blend_mode"] = str(blend)
                entry["two_sided"] = bool(sided)
                entry["base_material"] = material.get_path_name()
                entry["textures"] = [t.get_path_name() for t in unreal.MaterialEditingLibrary.get_used_textures(material)]
                for label, prop in (("opacity_mask_connected", unreal.MaterialProperty.MP_OPACITY_MASK),
                                    ("displacement_connected", unreal.MaterialProperty.MP_WORLD_POSITION_OFFSET)):
                    node = unreal.MaterialEditingLibrary.get_material_property_input_node(material, prop)
                    entry[label] = node is not None
                    if label == "displacement_connected":
                        entry["displacement_constant_zero"] = bool(node is not None and isinstance(node, unreal.MaterialExpressionConstant) and node.get_editor_property("r") == 0.)
            except Exception as error:
                entry["inspection_error"] = str(error)
            material_review.append(entry)
        package_file = Path(unreal.Paths.project_content_dir()) / (asset_path.split(".")[0].removeprefix("/Game/") + ".uasset")
        package_hash = hashlib.sha256(package_file.read_bytes()).hexdigest()
        try:
            nanite_enabled = bool(mesh.get_editor_property("nanite_settings").get_editor_property("enabled"))
        except Exception:
            nanite_enabled = None  # Commandlets may not expose the editor subsystem/property.
        rows.append({"id": name, "species": name.split("_")[0], "mesh": path.name, "units": "m", "unreal_asset": asset_path,
                     "geometry_version": mode + "-lod0-enu-v2", "geometry_matches_render": False,
                     "unreal_uasset_sha256": package_hash,
                     "material_geometry_reviewed": False, "materials": materials,
                     "material_inspection": material_review, "nanite_enabled": nanite_enabled,
                     "review_required": "Inspect opacity masks, displacement, closure, winding and leaf thickness. Approval is not automatic."})
        if specifications:
            original = next(s for s in specifications if s["unreal_asset"] == asset_path)
            rows[-1]["solid_recipe"] = original
            rows[-1]["species"] = original.get("species", rows[-1]["species"])
        if uv_path:
            rows[-1]["uv0"] = uv_path.name
    (out / "assets.review.json").write_text(json.dumps({"assets": rows}, indent=2), encoding="utf-8")
    unreal.log(f"Exported {len(rows)} plant candidates to {out}; all require review before CPU volume audit.")


run()
