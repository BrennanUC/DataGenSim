"""Bridge immutable authored-field captures into the existing reconstruction worker."""
from pathlib import Path
import shutil

from .field_volume import finalize_volume_region, validate_render_proof
from .io import atomic_json, child, identity, lease, read_json, sha256
from .scenes import family_split
from .storage import Storage
from .profiling import profiled, annotate

AUTHORED_SCHEMA = 'cropvolume-authored-solid-scene-v1'
DEFAULT_CONFIG = {'budget_bytes': 800_000_000_000, 'reserve_bytes': 100_000_000_000,
    'reconstruction_recipe': 'rgb_known_pose',
    'training': {'steps': 4000, 'init_points': 30000, 'max_gaussians': 100000, 'preserve_green_init': True}}


def capture_metadata(scene, volume_receipt_sha256):
    # Conservatively group this entire source map together; overlapping field
    # regions must never leak across downstream learning splits.
    family = identity(['authored-field-map-v1', scene['source_map_sha256']])
    data = {'schema': 'cropvolume-authored-sample-v1', 'scene_id': scene['scene_id'],
        'scene_family_id': family, 'split': family_split(family),
        'source_volume_receipt_sha256': volume_receipt_sha256,
        'source_region_id': scene['source_region_id'], 'source_map_sha256': scene['source_map_sha256'],
        'splat_scope': 'full_scene', 'label_scope': scene['label_scope'],
        'target_region': {'center_enu_m': scene['center_enu_m'], 'patch_m': scene['patch_m'],
                          'source_region_id': scene['source_region_id']},
        'target_instance_ids': [r['instance_id'] for r in scene['instances']],
        'grouping_policy': 'All regions and capture variants of this source map share one learning split'}
    data['metadata_id'] = identity(data)
    return data


@profiled('authored_capture_validation', root_arg='sample')
def verify_authored_capture(sample):
    sample = Path(sample)
    receipt = finalize_volume_region(sample)  # Existing receipt: checksums only; never remeasure.
    scene = read_json(sample / 'scene.json')
    if (scene.get('schema') != AUTHORED_SCHEMA or sha256(sample / 'scene.json') != receipt['files']['solid_scene.json']
            or receipt['scene_id'] != scene['scene_id']):
        raise ValueError('Authored capture changed the immutable volume scene')
    validate_render_proof(scene, read_json(sample / 'SOLID_RENDER_FINISHED.json'))
    metadata = read_json(sample / 'sample.json')
    if metadata != capture_metadata(scene, sha256(sample / 'VOLUME_COMPLETE.json')):
        raise ValueError('Authored capture target scope, provenance or split metadata changed')
    label = read_json(sample / 'labels/volume.json')
    if (label['scene_id'] != scene['scene_id'] or label['label_scope'] != metadata['label_scope']
            or len(label['instances']) != len(metadata['target_instance_ids'])
            or {r['instance_id'] for r in label['instances']} != set(metadata['target_instance_ids'])):
        raise ValueError('Authored capture label does not match the selected targets')
    return metadata


@profiled('staging', root_arg='root')
def stage_volume_capture(source, root, config=None):
    source, root = Path(source), Path(root)
    config = DEFAULT_CONFIG if config is None else config
    receipt = finalize_volume_region(source)
    scene = read_json(source / 'solid_scene.json')
    if scene.get('schema') != AUTHORED_SCHEMA:
        raise ValueError('Expected an audited authored-field solid scene')
    sample_id = 'authored_' + scene['scene_id'][:12]
    annotate(sample_id=sample_id)
    sample = root / 'samples' / sample_id
    storage = Storage(root, config['budget_bytes'], config['reserve_bytes'])
    metadata = {'schema': 'cropvolume-dataset-v1', 'source_type': 'authored_field',
                'root_is_configurable': True, 'volume_units': 'm3', 'config': config}
    with lease(root / 'locks/metadata'):
        if (root / 'dataset.json').exists():
            if read_json(root / 'dataset.json') != metadata:
                raise ValueError('Dataset settings are immutable; use another root for another recipe')
        else:
            atomic_json(root / 'dataset.json', metadata)
    with lease(root / 'locks' / (sample_id + '_stage')):
        if (sample / 'CAPTURE_COMPLETE.json').exists():
            annotate(reused=True)
            from .prepare import verify_capture
            verify_capture(sample)
            if sha256(source / 'VOLUME_COMPLETE.json') != sha256(sample / 'VOLUME_COMPLETE.json'):
                raise ValueError('Existing sample came from another volume capture')
            if not (sample / 'PLAN_COMPLETE.json').exists():
                atomic_json(sample / 'PLAN_COMPLETE.json', {'scene_id': scene['scene_id'], 'sample_id': sample_id,
                    'source_volume_receipt_sha256': sha256(sample / 'VOLUME_COMPLETE.json'), 'state': 'capture_complete'})
            return sample_id
        job = 'stage_' + sample_id
        storage.reserve_job(job, sum(child(source, name).stat().st_size for name in receipt['files']) + 100_000_000)
        try:
            sample.mkdir(parents=True, exist_ok=True)
            for name in [*receipt['files'], 'VOLUME_COMPLETE.json']:
                target = child(sample, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.staging')
                shutil.copyfile(child(source, name), temporary)
                temporary.replace(target)
            # Byte-for-byte copies retain the exact scene and label identities.
            shutil.copyfile(sample / 'solid_scene.json', sample / 'scene.json')
            shutil.copyfile(sample / 'SOLID_RENDER_FINISHED.json', sample / 'RENDER_FINISHED.json')
            atomic_json(sample / 'sample.json', capture_metadata(scene, sha256(sample / 'VOLUME_COMPLETE.json')))
            from .prepare import seal_capture
            seal_capture(sample)
            atomic_json(sample / 'PLAN_COMPLETE.json', {'scene_id': scene['scene_id'], 'sample_id': sample_id,
                        'source_volume_receipt_sha256': sha256(sample / 'VOLUME_COMPLETE.json'), 'state': 'capture_complete'})
        finally:
            storage.release_job(job)
    return sample_id
