import numpy as np
import pytest
from PIL import Image

from cropvolume.cameras import import_colmap
from cropvolume.io import atomic_json, read_json
from cropvolume.reference import plan_reference, verify_reference


def test_reference_cameras_use_original_field_coordinates_without_labels(tmp_path):
    plan_reference(tmp_path, center_enu_m=(50., 2.), views=24, width=32, height=24)
    frames = import_colmap(tmp_path)
    centers = np.array([f['c2w_cv'] for f in frames])[:, :3, 3]
    assert np.allclose(centers.mean(0)[:2], [50., 2.])
    assert not (tmp_path / 'labels').exists()
    assert not (tmp_path / 'scene.json').exists()
    assert all('depth' not in f for f in read_json(tmp_path / 'capture.json')['frames'])
    with pytest.raises(ValueError, match='empty'):
        plan_reference(tmp_path)


@pytest.mark.parametrize('center,patch', [([np.nan, 0.], [2., 2.]), ([101., 0.], [2., 2.]),
                                        ([50., 0.], [-2., 2.]), ([50., 0.], [np.inf, 2.])])
def test_invalid_reference_region_is_rejected(tmp_path, center, patch):
    with pytest.raises(ValueError):
        plan_reference(tmp_path, center_enu_m=center, patch_m=patch)
    assert not (tmp_path / 'reference.json').exists()


def reference_fixture(root):
    source = plan_reference(root, views=24, width=32, height=24)
    plan = read_json(root / 'capture.json')
    for frame in plan['frames']:
        path = root / frame['image']
        path.parent.mkdir(exist_ok=True)
        pixels = np.indices((24, 32)).sum(0).astype(np.uint8) * 4
        Image.fromarray(pixels).convert('RGB').save(path)
    atomic_json(root / 'REFERENCE_RENDER_FINISHED.json', {
        'scene_id': source['scene_id'], 'source_map': source['source_map'], 'frames': 24,
        'field_unchanged': True, 'field_before_sha256': 'original', 'field_after_sha256': 'original',
        'maximum_camera_error': 1e-7})
    atomic_json(root / 'source_content.json', {key: {'field.umap': 'original'}
        for key in ('files', 'files_after', 'staged_files', 'staged_files_after')})


def test_reference_seals_only_after_images_and_preservation_checks(tmp_path):
    reference_fixture(tmp_path)
    receipt = verify_reference(tmp_path)
    assert receipt['frames'] == 24 and receipt['volume_labels'] is False
    assert (tmp_path / 'preview.jpg').exists()
    assert not (tmp_path / 'COMPLETE.json').exists()


@pytest.mark.parametrize('target', ['field', 'asset', 'camera', 'image'])
def test_reference_does_not_seal_changed_field_assets_cameras_or_bad_images(tmp_path, target):
    reference_fixture(tmp_path)
    if target == 'field':
        path = tmp_path / 'REFERENCE_RENDER_FINISHED.json'
        data = read_json(path)
        data['field_after_sha256'] = 'changed'
        atomic_json(path, data)
    elif target == 'asset':
        path = tmp_path / 'source_content.json'
        data = read_json(path)
        data['staged_files_after']['field.umap'] = 'changed'
        atomic_json(path, data)
    elif target == 'camera':
        path = tmp_path / 'REFERENCE_RENDER_FINISHED.json'
        data = read_json(path)
        data['maximum_camera_error'] = .01
        atomic_json(path, data)
    else:
        Image.new('RGB', (32, 24)).save(tmp_path / 'images/IMG_0001.png')
    with pytest.raises(ValueError):
        verify_reference(tmp_path)
    assert not (tmp_path / 'REFERENCE_COMPLETE.json').exists()
