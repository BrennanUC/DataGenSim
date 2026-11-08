import math
from pathlib import Path
import numpy as np

from .geometry import load_catalog
from .io import identity


def rng_for(seed, purpose):
    return np.random.default_rng(int(identity([seed, purpose])[:16], 16))


def family_split(family):
    n = int(identity(family)[:8], 16) % 100
    return "train" if n < 80 else "validation" if n < 90 else "test"


def generate(catalog_path, seed, config, family_seed=None):
    if config.get("geometry_augmentation", "transform_only") != "transform_only":
        raise ValueError("Nonlinear augmentation needs baked geometry and a new catalog; it is not implemented")
    catalog = load_catalog(catalog_path)
    rng = rng_for(seed, "geometry")
    family = identity(["family-v1", catalog["catalog_id"], seed if family_seed is None else family_seed])
    all_assets = catalog["assets"]
    if not all_assets:
        raise ValueError("No approved solid plant assets; run the mesh audit first")
    species = config.get("species", "basil")
    assets = [a for a in all_assets if species == "mixed" or a["species"] == species]
    if not assets:
        raise ValueError("No approved assets for requested species")
    lo, hi = config.get("plant_count", [4, 16])
    if lo < 1 or hi < lo or hi > 10000:
        raise ValueError("Invalid plant count range")
    width, length = config.get("patch_m", [2., 2.])
    if min(width, length) <= 0:
        raise ValueError("Patch dimensions must be positive")
    n = int(rng.integers(lo, hi + 1))
    field = config.get("field")
    basesim = field and field.get('backend') == 'unreal_basesim_editor'
    slots = None
    authored_rows = basesim and field.get('row_layout') == 'authored_basesim_anchors_v1' and config.get('layout') == 'rows'
    if authored_rows:
        from .basesim_rows import row_slots, anchors
        slots = row_slots(width, length, config.get('scale', [.8,1.2])[1], config.get('min_center_spacing_m', .08))
        if len(slots) < n:
            raise ValueError('Not enough BaseSim row anchors for requested plant count')
        slots = [slots[j] for j in rng.permutation(len(slots))[:n]]
    if config.get("layout") == "game_beds":
        if not field or field.get("backend") != "unreal_game":
            raise ValueError("game_beds requires the FlightSchool field profile")
        beds = int(field["bed_count"])
        spacing = float(field["bed_spacing_m"])
        step = float(field["plant_step_m"])
        if min(beds, spacing, step) <= 0:
            raise ValueError("Invalid bed layout")
        radius = max(np.linalg.norm(np.abs(np.asarray(a["bounds_m"])[:, :2]).max(axis=0)) for a in assets) * config.get("scale", [.8, 1.2])[1]
        half_steps = int((width / 2 - radius - .01) // step)
        slots = [(i*step, (b-(beds-1)/2)*spacing+offset) for b in range(beds)
                 for offset in field["lane_offsets_m"] for i in range(-half_steps, half_steps+1)]
        if half_steps < 0 or len(slots) < n:
            raise ValueError("Not enough authored bed-row planting positions")
        slots = [slots[i] for i in rng.permutation(len(slots))[:n]]
    instances = []
    for i in range(n):
        for attempt in range(1000):
            asset = assets[int(rng.integers(len(assets)))]
            a, b = config.get("scale", [.8, 1.2])
            if not (0 < a <= b):
                raise ValueError("Scale range must be positive and ordered")
            scale = rng.uniform(a, b, size=3) if config.get("anisotropic", False) else np.repeat(rng.uniform(a, b), 3)
            yaw = rng.uniform(0, 2 * math.pi)
            rot = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            bounds = np.asarray(asset["bounds_m"])
            corners = np.array([[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]])
            rotated = (corners * scale) @ rot.T
            low, high = rotated.min(0), rotated.max(0)
            ranges = np.array([[-width / 2 - low[0], width / 2 - high[0]], [-length / 2 - low[1], length / 2 - high[1]]])
            if (ranges[:, 0] > ranges[:, 1]).any():
                continue
            xy = rng.uniform(ranges[:, 0], ranges[:, 1])
            layout = config.get("layout", "scatter")
            if layout == "game_beds":
                xy = np.asarray(slots[i]) + rng.uniform(-.008, .008, 2)
                if (xy < ranges[:, 0]).any() or (xy > ranges[:, 1]).any():
                    continue
            elif authored_rows:
                xy = np.array(slots[i][:2])
                if (xy < ranges[:, 0]).any() or (xy > ranges[:, 1]).any():
                    continue
            elif layout == "rows":
                cols = math.ceil(math.sqrt(n))
                xy = np.array([((i % cols + .5) / cols - .5) * width, ((i // cols + .5) / cols - .5) * length])
                xy += rng.uniform(-.05, .05, 2)
                if (xy < ranges[:, 0]).any() or (xy > ranges[:, 1]).any():
                    continue
            elif layout != "scatter":
                raise ValueError("Layout must be rows or scatter")
            spacing = config.get("min_center_spacing_m", .08)
            if any(np.linalg.norm(xy - np.array(v["local_to_world_m"])[:2, 3]) < spacing for v in instances):
                continue
            transform = np.eye(4)
            transform[:3, :3] = rot @ np.diag(scale)
            transform[:3, 3] = [*xy, (field["bed_top_m"] if slots is not None or basesim else 0) - low[2]]
            instances.append({"instance_id": f"plant_{i:04d}", "role": "plant", "asset_id": asset["id"],
                              "local_to_world_m": transform.tolist(), "augmentation": {"scale": scale.tolist(), "yaw_rad": yaw}})
            if authored_rows:
                instances[-1]['source_anchor_index'] = slots[i][2]
            break
        else:
            raise ValueError("Cannot fit requested plants in patch; reduce count/size/spacing")
    appearance_rng = rng_for(seed, "appearance")
    if basesim:
        center = np.asarray(field['center_enu_m'], dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError('Invalid BaseSim field center')
        for instance in instances:
            instance['local_to_world_m'][0][3] += float(center[0])
            instance['local_to_world_m'][1][3] += float(center[1])
    scene = {"schema": "cropvolume-scene-v1", "catalog_id": catalog["catalog_id"], "seed": seed,
             "scene_family_id": family, "split": family_split(family), "coordinate_frame": "ENU_m_Z_up",
             "patch_m": [width, length], "instances": instances,
             "appearance": {"exposure_ev100": float(appearance_rng.uniform(*config.get("exposure_ev100", [10.5, 12.5]))), "seed": int(appearance_rng.integers(2**31))},
             "test_fixture": any(a.get("test_fixture", False) for a in assets), "resolved_config": config}
    scene["placement_version"] = "basesim-authored-rows-v1" if authored_rows else "upright-enu-v2"
    if slots is not None or basesim:
        scene["field"] = field
    if basesim:
        scene['center_enu_m'] = field['center_enu_m']
        scene['appearance'] = {'policy':field['appearance_policy'], 'seed':scene['appearance']['seed']}
        scene['label_scope'] = ('All rendered plants inside the bounded field; exterior is blank'
            if field.get('context_policy') == 'bounded_field_black_exterior_v1' else
            'Generated target plants only; preserved authored plants outside the target region are context and excluded from the volume label')
        if authored_rows:
            scene['row_anchor_provenance'] = anchors()[2]
    scene["scene_id"] = identity(scene)
    return scene
