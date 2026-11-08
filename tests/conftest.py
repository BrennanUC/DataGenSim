from pathlib import Path
import numpy as np
import pytest
import trimesh
from PIL import Image

from cropvolume.cameras import capture_plan, export_cameras
from cropvolume.geometry import audit_catalog, measure_scene
from cropvolume.io import atomic_json
from cropvolume.prepare import seal_capture
from cropvolume.scenes import generate


@pytest.fixture
def catalog(tmp_path):
    mesh = trimesh.creation.box(extents=[.2, .2, .4])
    mesh.export(tmp_path / "plant.ply")
    atomic_json(tmp_path / "assets.json", {"assets": [{"id": "basil_fixture", "species": "basil", "mesh": "plant.ply",
                "units": "m", "geometry_matches_render": True, "material_geometry_reviewed": True,
                "geometry_version": "analytic-box-v1", "test_fixture": True}]})
    path = tmp_path / "catalog.json"
    audit_catalog(tmp_path / "assets.json", path)
    return path


@pytest.fixture
def sample(tmp_path, catalog):
    path = tmp_path / "dataset/samples/fixture_0001"
    path.mkdir(parents=True)
    scene = generate(catalog, 923, {"plant_count": [2, 2], "scale": [1., 1.], "species": "basil"})
    atomic_json(path / "scene.json", scene)
    atomic_json(path / "labels/volume.json", measure_scene(scene, catalog))
    capture = capture_plan(scene, views=24, width=32, height=24)
    atomic_json(path / "capture.json", capture)
    export_cameras(path, capture["frames"])
    for frame in capture["frames"]:
        image = path / frame["image"]
        image.parent.mkdir(exist_ok=True)
        # Explicit fixture imagery; does not pretend to be an Unreal rendering.
        Image.fromarray(np.full((24, 32, 3), 100, dtype=np.uint8)).save(image)
        depth = path / frame["depth"]
        depth.parent.mkdir(exist_ok=True)
        Image.fromarray(np.full((24, 32), 1000, dtype=np.uint16)).save(depth)
    seal_capture(path, "cpu_fixture")
    return path


@pytest.fixture(autouse=True)
def isolated_machine_configuration(tmp_path, monkeypatch):
    """Tests never read a developer's preferences or write to their runtime."""
    monkeypatch.setenv('CROPVOLUME_CONFIG', str(tmp_path/'machine.local.json'))
    monkeypatch.setenv('CROPVOLUME_RUNTIME_ROOT', str(tmp_path/'runtime'))
