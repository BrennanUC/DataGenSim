"""Explicit leaf-thickness augmentation; rendering must use the resulting mesh too."""
import numpy as np
import trimesh
from plyfile import PlyData

from .geometry import solid_union
from .io import atomic_json, identity, read_json, sha256


def cap_planar_stem_ends(surface):
    """Explicit repair of convex planar tube ends; reject irregular stem holes."""
    edges = surface.edges
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    boundary = edges[counts[inverse] == 1]
    if (counts > 2).any() or not surface.is_winding_consistent:
        raise ValueError("Nonmanifold stem")
    successors = {int(a): int(b) for a, b in boundary}
    if len(successors) != len(boundary) or set(successors) != set(successors.values()):
        raise ValueError("Stem boundaries are not simple closed loops")
    vertices, faces = surface.vertices.tolist(), surface.faces.tolist()
    while successors:
        start = next(iter(successors)); loop = [start]; current = successors.pop(start)
        while current != start:
            loop.append(current)
            if current not in successors:
                raise ValueError("Open stem boundary")
            current = successors.pop(current)
        points = surface.vertices[loop]
        center = points.mean(0)
        _, _, basis = np.linalg.svd(points-center, full_matrices=False)
        if len(loop) < 3 or np.abs((points-center) @ basis[-1]).max() > 1e-6:
            raise ValueError("Stem end is not planar")
        uv = (points-center) @ basis[:2].T
        turns = np.cross(np.roll(uv,-1,axis=0)-uv, np.roll(uv,-2,axis=0)-np.roll(uv,-1,axis=0))
        if not ((turns >= -1e-12).all() or (turns <= 1e-12).all()):
            raise ValueError("Stem end is not convex")
        middle = len(vertices); vertices.append(center.tolist())
        faces.extend([[loop[(i+1)%len(loop)], a, middle] for i,a in enumerate(loop)])
    solid = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not solid.is_volume:
        raise ValueError("Stem end capping did not create a positive closed solid")
    return solid


def thicken_surface(surface, thickness_m):
    if not np.isfinite(thickness_m) or thickness_m <= 0:
        raise ValueError("Leaf thickness must be finite and positive")
    if surface.is_watertight or not surface.is_winding_consistent:
        raise ValueError("Expected an orientable open leaf surface")
    edges = surface.edges
    unique, inverse, count = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    if (count > 2).any():
        raise ValueError("Nonmanifold leaf surface cannot be thickened automatically")
    boundary = edges[count[inverse] == 1]
    n = len(surface.vertices)
    normals = surface.vertex_normals
    vertices = np.vstack([surface.vertices + normals * thickness_m / 2,
                          surface.vertices - normals * thickness_m / 2])
    side = [[int(b), int(a), int(a + n)] for a, b in boundary]
    side += [[int(b), int(a + n), int(b + n)] for a, b in boundary]
    faces = np.vstack([surface.faces, surface.faces[:, ::-1] + n, np.asarray(side)])
    result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not result.is_volume:
        raise ValueError("Thickened leaf did not form a positive closed solid")
    return result


def _close_attributes(original, closed, values, leaf_shell):
    if values is None:
        return None
    vertex_values = np.zeros((len(original.vertices), values.shape[-1]))
    vertex_values[original.faces.reshape(-1)] = values.reshape(-1, values.shape[-1])
    if leaf_shell:
        walls = closed.faces[2*len(original.faces):] % len(original.vertices)
        return np.concatenate([values, values[:, ::-1], vertex_values[walls]])
    added = len(closed.vertices)-len(original.vertices)
    vertex_values = np.vstack([vertex_values, np.tile(vertex_values.mean(0), (added,1))])
    return np.concatenate([values, vertex_values[closed.faces[len(original.faces):]]])


def solidify_export(source_path, output, thickness_m=.0003, cap_stems=False):
    """Prepare explicit versioned input for a later Unreal import/round-trip audit.

    The thickness is a synthetic modeling choice, not a measured plant property.
    Approval happens only after the same geometry is imported and rendered.
    """
    from pathlib import Path
    source_path, output = Path(source_path), Path(output)
    data = PlyData.read(source_path)
    vertices = np.column_stack([data["vertex"][axis] for axis in ("x", "y", "z")])
    faces = np.vstack(data["face"]["vertex_indices"])
    materials = np.asarray(data["face"]["material_id"])
    uv_path = source_path.with_suffix(".uv0.json")
    source_uv = np.asarray(read_json(uv_path), float) if uv_path.exists() else None
    if source_uv is not None and (source_uv.shape != (len(faces), 3, 2) or not np.isfinite(source_uv).all()):
        raise ValueError("Invalid source triangle UVs")
    color_path = source_path.with_suffix(".color0.json")
    source_colors = np.asarray(read_json(color_path), float) if color_path.exists() else None
    if source_colors is not None and (source_colors.shape != (len(faces),3,4) or not np.isfinite(source_colors).all()):
        raise ValueError("Invalid source vertex colors")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.face_attributes["material_id"] = materials
    mesh.process(validate=False)
    groups = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), engine="scipy")
    solids, material_ids, triangle_uv, triangle_colors = [], [], [], []
    for group in groups:
        values = np.unique(mesh.face_attributes["material_id"][group])
        if len(values) != 1:
            raise ValueError("A connected component spans multiple materials; manual review required")
        material_id = int(values[0])
        part = mesh.submesh([group], append=True, repair=False)
        part_uv = source_uv[group] if source_uv is not None else None
        part_colors = source_colors[group] if source_colors is not None else None
        original = part
        if not part.is_watertight:
            if material_id != 0:
                if material_id != 1 or not cap_stems:
                    raise ValueError("An open nonleaf component requires manual repair")
                part = cap_planar_stem_ends(part)
            else:
                part = thicken_surface(part, thickness_m)
            part_uv = _close_attributes(original, part, part_uv, material_id == 0)
            part_colors = _close_attributes(original, part, part_colors, material_id == 0)
        elif not part.is_volume:
            raise ValueError("Closed source component has invalid orientation/volume")
        solids.append(part)
        material_ids.extend([material_id] * len(part.faces))
        if part_uv is not None:
            triangle_uv.extend(part_uv.tolist())
        if part_colors is not None:
            triangle_colors.extend(part_colors.tolist())
    merged = solid_union(solids)
    assembled = trimesh.util.concatenate(solids)
    result = {"schema": "cropvolume-solid-geometry-v1", "source_sha256": sha256(source_path),
              "leaf_thickness_m": thickness_m, "geometry_version": f"leaf-shell-{thickness_m:.9g}m-v1",
              "vertices_m": assembled.vertices.tolist(), "triangles": assembled.faces.tolist(),
              "material_ids": material_ids, "plant_solid_volume_m3": float(merged.volume),
              "notice": "Explicit synthetic leaf thickness. Requires Unreal round-trip validation before catalog approval."}
    if source_uv is not None:
        result.update(triangle_uv0=triangle_uv, vertex_normals=assembled.vertex_normals.tolist(),
                      source_uv0_sha256=sha256(uv_path))
    if cap_stems:
        result["stem_repair"] = "explicit_convex_planar_end_caps_v1"
    if source_colors is not None:
        result.update(triangle_colors=triangle_colors, source_colors_sha256=sha256(color_path))
    result["geometry_id"] = identity(result)
    atomic_json(output, result)
    return result
