"""Ground truth comes from validated final solids, independently of reconstruction."""
from pathlib import Path
import heapq
import math
import numpy as np
import trimesh

from .io import atomic_json, child, identity, read_json, sha256
from .profiling import profiled

METHOD = "manifold-solid-union-v1"
SCENE_METHOD = 'cached-solids-local-overlap-v3'
MAX_BOOLEAN_GROUP = 32


def load_mesh(path, unit="m", process=True):
    if unit not in ("m", "cm"):
        raise ValueError("Explicit geometry units must be m or cm")
    mesh = trimesh.load_mesh(path, process=process)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"Expected one triangle mesh: {path}")
    if unit == "cm":
        mesh.apply_scale(.01)
    return mesh


def solid_union(meshes, preserve_csg_boundaries=False):
    solids = []
    for mesh in meshes:
        if not np.isfinite(mesh.vertices).all():
            raise ValueError("Nonfinite mesh vertices")
        # Approved Boolean caches include closed internal cavity shells. Keep
        # their negative winding attached to the outer boundary. Raw/source
        # meshes still take the strict per-component audit below.
        parts = [mesh] if preserve_csg_boundaries else mesh.split(only_watertight=False, repair=False)
        if not len(parts):
            raise ValueError("No solid geometry")
        for part in parts:
            if not part.is_volume:
                raise ValueError("Open or inconsistently oriented geometry cannot provide solid volume")
            # Reconstruct through the Boolean backend even for a single part. This checks
            # backend acceptance; geometric/material review is additionally required.
            solids.append(part)
    if not solids:
        raise ValueError("No plant solids selected")
    result = trimesh.boolean.union(solids, engine="manifold", check_volume=True)
    if result.is_empty or not result.is_volume or not np.isfinite(result.volume) or result.volume <= 0:
        raise ValueError("Solid union failed")
    return result


@profiled('asset_audit', root_arg='output')
def audit_catalog(source, output):
    source, output = Path(source), Path(output)
    config = read_json(source)
    entries, rejected = [], []
    output.parent.mkdir(parents=True, exist_ok=True)
    for spec in config["assets"]:
        try:
            if spec.get("geometry_matches_render") is not True:
                raise ValueError("Render/measurement geometry equivalence has not been reviewed")
            if spec.get("material_geometry_reviewed") is not True:
                raise ValueError("Opacity cutouts/displacement require explicit geometry review")
            asset_id = spec["id"]
            if not asset_id.replace("_", "").isalnum():
                raise ValueError("Asset ID must be alphanumeric with underscores")
            mesh_path = child(source.parent, spec["mesh"])
            union = solid_union([load_mesh(mesh_path, spec["units"])])
            baked = output.parent / "solids" / (asset_id + ".ply")
            baked.parent.mkdir(exist_ok=True)
            union.export(baked)
            # Welding near-coincident CSG intersection vertices can destroy
            # valid topology. Preserve the backend's cached vertex indices.
            union = load_mesh(baked, process=False)
            if not union.is_volume:
                raise ValueError("Cached CSG boundary did not preserve a closed positive solid")
            entries.append({"id": asset_id, "species": spec["species"], "unreal_asset": spec.get("unreal_asset"),
                            "mesh": baked.relative_to(output.parent).as_posix(), "mesh_sha256": sha256(baked),
                            "source_sha256": sha256(mesh_path), "volume_m3": float(union.volume),
                            "bounds_m": union.bounds.tolist(), "approved": True,
                            "test_fixture": bool(spec.get("test_fixture", False)),
                            "geometry_version": spec["geometry_version"], "method": METHOD})
            entries[-1]["unreal_uasset_sha256"] = spec.get("unreal_uasset_sha256")
            entries[-1]["solid_recipe"] = spec.get("solid_recipe")
        except (ValueError, KeyError, OSError) as error:
            rejected.append({"id": spec.get("id"), "reason": str(error)})
    ids = [a["id"] for a in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate asset IDs")
    result = {"schema": "cropvolume-catalog-v1", "assets": entries, "rejected": rejected}
    result["catalog_id"] = identity(result)
    atomic_json(output, result)
    return result


def load_catalog(path):
    path = Path(path)
    catalog = read_json(path)
    expected = catalog.pop("catalog_id")
    if identity(catalog) != expected:
        raise ValueError("Catalog identity mismatch")
    catalog["catalog_id"] = expected
    for asset in catalog["assets"]:
        if not asset["approved"] or sha256(child(path.parent, asset["mesh"])) != asset["mesh_sha256"]:
            raise ValueError("Unapproved or modified geometry")
    return catalog


def instance_bounds(asset, transform):
    bounds = np.asarray(asset['bounds_m'], dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or (bounds[1] < bounds[0]).any():
        raise ValueError('Invalid cached asset bounds')
    center = (bounds[0] + bounds[1]) * .5
    half = (bounds[1] - bounds[0]) * .5
    world_center = transform[:3, :3] @ center + transform[:3, 3]
    world_half = np.abs(transform[:3, :3]) @ half
    return world_center-world_half, world_center+world_half


def overlap_groups(bounds):
    """Conservative AABB components; O(n) working memory, never an n×n matrix."""
    if not bounds:
        return []
    values = np.asarray(bounds)
    low, high = values[:, 0], values[:, 1]
    axis = int(np.argmax(np.ptp((low+high)*.5, axis=0)))
    parent = list(range(len(bounds)))
    sizes = [1] * len(bounds)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    active, ends = set(), []
    for i in np.argsort(low[:, axis], kind='stable'):
        i = int(i)
        while ends and ends[0][0] <= low[i, axis]:
            _, old = heapq.heappop(ends)
            active.remove(old)
        if active:
            candidates = np.fromiter(active, dtype=int)
            hits = candidates[np.all(high[candidates] > low[i], axis=1) & np.all(low[candidates] < high[i], axis=1)]
            for j in hits:
                a, b = find(i), find(int(j))
                if a != b:
                    if sizes[a] < sizes[b]:
                        a, b = b, a
                    parent[b] = a
                    sizes[a] += sizes[b]
        active.add(i)
        heapq.heappush(ends, (high[i, axis], i))
    groups = {}
    for i in range(len(bounds)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda group: group[0])


def measure_scene(scene, catalog_path, boolean_origin_m=None):
    catalog = load_catalog(catalog_path)
    if scene["catalog_id"] != catalog["catalog_id"]:
        raise ValueError("Scene/catalog mismatch")
    assets = {a["id"]: a for a in catalog["assets"]}
    rows, transforms, bounds, selected = [], [], [], []
    origin = np.zeros(3) if boolean_origin_m is None else np.asarray(boolean_origin_m, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('Boolean origin must have three finite metric coordinates')
    for instance in scene["instances"]:
        if instance["role"] != "plant":
            continue
        asset = assets[instance["asset_id"]]
        transform = np.asarray(instance["local_to_world_m"], dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError("Invalid instance transform")
        determinant = abs(np.linalg.det(transform[:3, :3]))
        if not np.isfinite(determinant) or determinant < 1e-12:
            raise ValueError("Degenerate transform")
        volume = asset['volume_m3'] * determinant
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError('Invalid cached solid volume')
        transforms.append(transform)
        bounds.append(instance_bounds(asset, transform))
        selected.append(asset)
        rows.append({"instance_id": instance["instance_id"], "asset_id": asset["id"],
                     "mesh_sha256": asset["mesh_sha256"], "volume_m3": volume,
                     "geometry_version": asset["geometry_version"],
                     "leaf_thickness_m": (asset.get("solid_recipe") or {}).get("leaf_thickness_m")})
    if not rows:
        raise ValueError('No plant solids selected')
    groups = overlap_groups(bounds)
    volumes, source_meshes = [], {}
    bounds_array = np.asarray(bounds)
    streamed_groups, largest_batch = 0, 0
    def transformed_mesh(i, group_origin):
        asset = selected[i]
        if asset['id'] not in source_meshes:
            source_meshes[asset['id']] = load_mesh(child(Path(catalog_path).parent, asset['mesh']), process=False)
        mesh = source_meshes[asset['id']].copy()
        local_transform = transforms[i].copy()
        local_transform[:3, 3] -= group_origin
        mesh.apply_transform(local_transform)
        return mesh
    boolean_groups = [group for group in groups if len(group) > 1]
    for group in groups:
        total = math.fsum(rows[i]['volume_m3'] for i in group)
        if len(group) == 1:
            volumes.append(total)
            continue
        if len(group) > MAX_BOOLEAN_GROUP:
            # Exact union-volume identity: add V(plant) minus its intersection
            # with the union of earlier neighbors. Distant earlier plants cannot
            # intersect, so no growing whole-component mesh is ever retained.
            streamed_groups += 1
            contributions = []
            for offset, i in enumerate(group):
                earlier = np.asarray(group[:offset], dtype=int)
                neighbors = earlier[np.all(bounds_array[earlier,1] > bounds_array[i,0], axis=1)
                                    & np.all(bounds_array[earlier,0] < bounds_array[i,1], axis=1)]
                if not len(neighbors):
                    contributions.append(rows[i]['volume_m3'])
                    continue
                group_origin = origin if boolean_origin_m is not None else transforms[i][:3,3]
                meshes = [transformed_mesh(int(j),group_origin) for j in neighbors]
                prior = meshes[0] if len(meshes)==1 else solid_union(meshes,preserve_csg_boundaries=True)
                current = transformed_mesh(i,group_origin)
                intersection = trimesh.boolean.intersection([current,prior],engine='manifold',check_volume=True)
                overlap = 0. if intersection.is_empty else float(intersection.volume)
                if not np.isfinite(overlap) or overlap < -1e-12:
                    raise ValueError('Invalid local solid intersection')
                contributions.append(max(0.,rows[i]['volume_m3']-max(0.,overlap)))
                largest_batch = max(largest_batch,len(neighbors)+1)
                del meshes,prior,current,intersection
            volumes.append(math.fsum(contributions))
            continue
        # Only possibly intersecting groups need geometry. All other plants use
        # their audited asset volume × absolute affine determinant directly.
        group_origin = origin if boolean_origin_m is not None else transforms[group[0]][:3, 3]
        meshes = [transformed_mesh(i,group_origin) for i in group]
        merged = solid_union(meshes, preserve_csg_boundaries=True)
        volumes.append(min(total, float(merged.volume)))
        largest_batch = max(largest_batch,len(group))
        del meshes, merged
    result = {"schema": "cropvolume-solid-label-v1", "scene_id": scene["scene_id"],
            "plant_solid_volume_m3": math.fsum(volumes), "method": SCENE_METHOD,
            "catalog_id": catalog["catalog_id"], "units": "m3", "instances": rows,
            "sum_instance_volume_m3": math.fsum(r["volume_m3"] for r in rows),
            'calculation': {'cached_instances': sum(len(group) == 1 for group in groups),
                'boolean_instances': sum(map(len, boolean_groups)), 'overlap_groups': len(boolean_groups),
                'largest_overlap_group': max(map(len, boolean_groups), default=0),
                'streamed_overlap_groups':streamed_groups,'largest_boolean_batch':largest_batch,
                'method': 'Cached asset volumes times affine determinants; isolated plants summed directly; local unions for small groups; larger groups use per-plant intersections with earlier overlapping neighbors'},
            "numerical_precision": "Float64 cached volumes/transforms/summation; Manifold float32 only for overlap groups; not a biological measurement"}
    if boolean_origin_m is not None:
        result['boolean_origin_m'] = origin.tolist()
        result['boolean_frame'] = 'ENU metres translated by the common origin before union; source transforms remain unchanged'
    return result
