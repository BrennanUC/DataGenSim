import numpy as np
from PIL import Image
import pytest

from cropvolume.cameras import capture_plan
from cropvolume.dataset import publish
from cropvolume.geometry import measure_scene
from cropvolume.io import atomic_json, read_json
from cropvolume.prepare import prepare, seal_capture
from cropvolume.scenes import generate


def test_instance_union_handles_duplicate_plant(catalog):
    scene = generate(catalog, 4, {"plant_count": [1, 1], "scale": [1, 1]})
    original = measure_scene(scene, catalog)["plant_solid_volume_m3"]
    duplicate = dict(scene["instances"][0], instance_id="second")
    scene["instances"].append(duplicate)
    result = measure_scene(scene, catalog)
    assert result["plant_solid_volume_m3"] == pytest.approx(original)
    assert result["sum_instance_volume_m3"] == pytest.approx(original * 2)


def test_invalid_depth_encoding_and_saturation(sample, tmp_path):
    capture = read_json(sample / "capture.json")
    for frame in capture["frames"]:
        Image.fromarray(np.full((24, 32), 65535, np.uint16)).save(sample / frame["depth"])
    (sample / "CAPTURE_COMPLETE.json").unlink()
    seal_capture(sample, "cpu_fixture")
    with pytest.raises(ValueError, match="Insufficient"):
        prepare(sample, tmp_path / "saturated", "depth_assisted")
    for frame in capture["frames"]:
        Image.fromarray(np.ones((24, 32), np.uint8)).save(sample / frame["depth"])
    (sample / "CAPTURE_COMPLETE.json").unlink()
    seal_capture(sample, "cpu_fixture")
    with pytest.raises(ValueError, match="16-bit"):
        prepare(sample, tmp_path / "uint8", "depth_assisted")


def test_nan_threshold_is_rejected_before_publication(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        publish(tmp_path, "anything", min_psnr=float("nan"))


def test_too_large_mesh_is_not_placed_outside_patch(catalog):
    with pytest.raises(ValueError, match="Cannot fit"):
        generate(catalog, 4, {"plant_count": [1, 1], "patch_m": [.01, .01], "scale": [1, 1]})
