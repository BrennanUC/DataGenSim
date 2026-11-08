from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from cropvolume.authored_capture import DEFAULT_CONFIG, capture_metadata, stage_volume_capture
from cropvolume.dataset import publish
from cropvolume.io import atomic_json, identity, read_json, sha256
from cropvolume.prepare import prepare, verify_capture


@pytest.fixture
def authored_source(sample, tmp_path):
    source = tmp_path / 'authored_source'
    source.mkdir()
    for name in ('capture.json', 'transforms.json', 'images', 'depth', 'sparse', 'labels'):
        original, destination = sample / name, source / name
        if original.is_dir():
            shutil.copytree(original, destination)
        else:
            shutil.copyfile(original, destination)
    scene = read_json(sample / 'scene.json')
    scene.pop('scene_id')
    scene.update(schema='cropvolume-authored-solid-scene-v1', source_map_sha256='analytic-fixture-map',
                 source_region_id='fixture-region', center_enu_m=[0,0], label_scope='Selected fixture plants only')
    for index, row in enumerate(scene['instances']):
        row.update(source_component_path='Field.Foliage.Fixture', source_instance_index=index, source_mesh='fixture')
    scene['scene_id'] = identity(scene)
    atomic_json(source / 'solid_scene.json', scene)
    plan = read_json(source / 'capture.json')
    plan['scene_id'] = scene['scene_id']
    atomic_json(source / 'capture.json', plan)
    label = read_json(source / 'labels/volume.json')
    label.update(scene_id=scene['scene_id'], label_scope=scene['label_scope'])
    atomic_json(source / 'labels/volume.json', label)
    proof = {'scene_id': scene['scene_id'], 'solid_scene_id': scene['scene_id'], 'field_unchanged': True,
        'field_before_sha256': 'fixture-frozen', 'field_after_sha256': 'fixture-frozen',
        'binding': {'all_authored_transforms_preserved': True, 'non_target_before_sha256': 'context',
                    'non_target_after_sha256': 'context', 'target_count': len(scene['instances'])},
        'target_instances': [{**r, 'forced_lod': 0} for r in scene['instances']]}
    atomic_json(source / 'SOLID_RENDER_FINISHED.json', proof)
    # Explicit analytic test imagery; not a real Unreal capture or production pair.
    for frame in plan['frames']:
        pixels = (np.indices((24,32)).sum(0)*4).astype('uint8')
        Image.fromarray(pixels).convert('RGB').save(source / frame['image'])
    files = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*') if p.is_file()}
    atomic_json(source / 'VOLUME_COMPLETE.json', {'scene_id': scene['scene_id'], 'files': files})
    return source


def config():
    result = deepcopy(DEFAULT_CONFIG)
    result['reserve_bytes'] = 0
    return result


def test_staging_preserves_scene_label_and_cameras_and_keeps_truth_out_of_training(authored_source, tmp_path):
    root = tmp_path / 'dataset'
    sample_id = stage_volume_capture(authored_source, root, config())
    sample = root / 'samples' / sample_id
    assert sha256(sample / 'scene.json') == sha256(authored_source / 'solid_scene.json')
    for name in ('labels/volume.json', 'capture.json', 'transforms.json'):
        assert sha256(sample / name) == sha256(authored_source / name)
    verify_capture(sample)
    prepared = tmp_path / 'prepared'
    data = prepare(sample, prepared, 'depth_assisted', max_points=100)
    assert data['scene_id'] == read_json(sample / 'scene.json')['scene_id']
    assert not (prepared / 'labels').exists() and not (prepared / 'geometry').exists()
    assert 'plant_solid_volume_m3' not in data


def test_staging_is_restartable_and_recovers_the_plan_marker(authored_source, tmp_path):
    root = tmp_path / 'dataset'
    sample_id = stage_volume_capture(authored_source, root, config())
    sample = root / 'samples' / sample_id
    before = sha256(sample / 'CAPTURE_COMPLETE.json')
    (sample / 'PLAN_COMPLETE.json').unlink()
    assert stage_volume_capture(authored_source, root, config()) == sample_id
    assert (sample / 'PLAN_COMPLETE.json').exists()
    assert sha256(sample / 'CAPTURE_COMPLETE.json') == before
    assert read_json(root / 'reservations.json') == {}


def test_modified_target_scope_is_rejected(authored_source, tmp_path):
    root = tmp_path / 'dataset'
    sample = root / 'samples' / stage_volume_capture(authored_source, root, config())
    metadata = read_json(sample / 'sample.json')
    metadata['label_scope'] = 'All visible plants'
    atomic_json(sample / 'sample.json', metadata)
    with pytest.raises(ValueError, match='scope'):
        verify_capture(sample)


def test_authored_publication_records_selected_targets_and_full_scene_scope(authored_source, tmp_path):
    from test_lifecycle import fixture_splat
    root = tmp_path / 'dataset'
    sample = root / 'samples' / stage_volume_capture(authored_source, root, config())
    fixture_splat(sample)
    row = publish(root, sample.name, allow_fixture=True)
    assert row['splat_scope'] == 'full_scene'
    assert row['label_scope'] == 'Selected fixture plants only'
    assert len(row['target_instance_ids']) == 2
    assert row['target_region']['source_region_id'] == 'fixture-region'
    assert read_json(sample / 'splat.manifest.json')['label_scope'] == row['label_scope']
    assert row == publish(root, sample.name, allow_fixture=True)


def test_overlapping_regions_of_the_same_map_share_one_learning_family():
    scene = {'scene_id': 'a', 'source_map_sha256': 'same-map', 'source_region_id': 'region-a',
             'label_scope': 'targets', 'center_enu_m': [50,0], 'patch_m': [2,2], 'instances': []}
    a = capture_metadata(scene, 'receipt-a')
    b = capture_metadata({**scene, 'scene_id': 'b', 'source_region_id': 'region-b', 'center_enu_m': [50.5,0]}, 'receipt-b')
    assert a['scene_family_id'] == b['scene_family_id'] and a['split'] == b['split']
