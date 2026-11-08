import numpy as np
import pytest
import trimesh

from cropvolume.geometry import audit_catalog, load_catalog, measure_scene, solid_union
from cropvolume.io import atomic_json, identity, read_json
from cropvolume.scenes import generate


def test_known_cube_union_does_not_double_count():
    a = trimesh.creation.box()
    b = a.copy()
    b.apply_translation([.5, 0, 0])
    assert solid_union([a, b]).volume == pytest.approx(1.5, rel=1e-6)


def test_open_surface_is_rejected():
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    with pytest.raises(ValueError, match="Open"):
        solid_union([mesh])


def test_scale_rotation_and_nonplant_exclusion(catalog):
    scene = generate(catalog, 7, {"plant_count": [1, 1], "scale": [1, 1]})
    base = measure_scene(scene, catalog)["plant_solid_volume_m3"]
    transform = np.array(scene["instances"][0]["local_to_world_m"])
    transform[:3, :3] *= 2
    transform[:3, 3] += [10, -20, 3]
    scene["instances"][0]["local_to_world_m"] = transform.tolist()
    scene["instances"].append({"role": "soil", "asset_id": "must_not_be_loaded"})
    assert measure_scene(scene, catalog)["plant_solid_volume_m3"] == pytest.approx(base * 8, rel=2e-5)
    transform[:3, :3] /= 2
    transform[:3, :3] = transform[:3, :3] @ np.diag([2, 3, 4])
    scene["instances"][0]["local_to_world_m"] = transform.tolist()
    assert measure_scene(scene, catalog)["plant_solid_volume_m3"] == pytest.approx(base * 24, rel=2e-5)


def test_catalog_rejects_unreviewed_asset_and_cm_conversion(tmp_path):
    trimesh.creation.box(extents=[100, 100, 100]).export(tmp_path / "cube.ply")
    spec = {"id": "cube", "mesh": "cube.ply", "units": "cm", "species": "basil", "geometry_version": "v1"}
    atomic_json(tmp_path / "source.json", {"assets": [spec]})
    result = audit_catalog(tmp_path / "source.json", tmp_path / "catalog.json")
    assert not result["assets"] and len(result["rejected"]) == 1
    spec.update(geometry_matches_render=True, material_geometry_reviewed=True)
    atomic_json(tmp_path / "source.json", {"assets": [spec]})
    result = audit_catalog(tmp_path / "source.json", tmp_path / "catalog.json")
    assert result["assets"][0]["volume_m3"] == pytest.approx(1.)


def test_catalog_mutation_is_detected(catalog):
    value = read_json(catalog)
    value["assets"][0]["volume_m3"] = 42.
    atomic_json(catalog, value)
    with pytest.raises(ValueError, match="identity"):
        load_catalog(catalog)


def test_seed_reproducibility_and_family_grouping(catalog):
    config = {"plant_count": [4, 6], "anisotropic": True}
    a = generate(catalog, 8, config)
    assert a == generate(catalog, 8, config)
    assert a["scene_id"] != generate(catalog, 9, config)["scene_id"]
    b = generate(catalog, 99, config, family_seed=8)
    assert b["scene_family_id"] == a["scene_family_id"] and b["split"] == a["split"]
    with pytest.raises(ValueError, match="Nonlinear"):
        generate(catalog, 8, {"geometry_augmentation": "bend"})
