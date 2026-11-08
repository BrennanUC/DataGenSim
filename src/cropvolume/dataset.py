from pathlib import Path
import math
import numpy as np
from plyfile import PlyData

from .io import atomic_json, canonical, child, identity, lease, read_json, sha256
from .prepare import verify_capture
from .profiling import profiled


def validate_ply(path):
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3"}
    if not len(vertex) or not required.issubset(vertex.dtype.names):
        raise ValueError("PLY is not a nonempty Gaussian splat")
    for name in vertex.dtype.names:
        if vertex[name].dtype.kind in "fiu" and not np.isfinite(vertex[name]).all():
            raise ValueError("Nonfinite Gaussian field: " + name)
    q = np.column_stack([vertex[f"rot_{i}"] for i in range(4)])
    if not np.allclose(np.linalg.norm(q, axis=1), 1., atol=.002):
        raise ValueError("Unnormalized Gaussian rotation")
    return {"gaussians": len(vertex), "bytes": Path(path).stat().st_size, "sha256": sha256(path)}


@profiled('publication', root_arg='root', sample_arg='sample_id')
def publish(root, sample_id, min_psnr=15., min_ssim=.3, allow_fixture=False, quality_policy='thresholds'):
    if quality_policy not in ('record_only', 'thresholds'):
        raise ValueError('Unknown reconstruction quality policy')
    if not math.isfinite(min_psnr) or not math.isfinite(min_ssim) or not 0 <= min_ssim <= 1:
        raise ValueError("Quality thresholds must be finite; SSIM must be within [0,1]")
    root = Path(root)
    sample = child(root / "samples", sample_id)
    with lease(root / "locks" / sample_id):
        if (sample / "RETIRED.json").exists():
            raise ValueError("Retired sample cannot be published")
        scene = read_json(sample / "scene.json")
        actual_id = scene.pop("scene_id")
        if identity(scene) != actual_id:
            raise ValueError("Scene identity mismatch")
        scene["scene_id"] = actual_id
        authored = scene.get('schema') == 'cropvolume-authored-solid-scene-v1'
        if authored:
            from .authored_capture import verify_authored_capture
            sample_metadata = verify_authored_capture(sample)
        else:
            sample_metadata = scene
        if scene.get("test_fixture") and not allow_fixture:
            raise ValueError("Test fixtures cannot be published as training data")
        marker, _ = verify_capture(sample)
        label = read_json(sample / "labels/volume.json")
        if marker["scene_id"] != label["scene_id"] or label["scene_id"] != actual_id:
            raise ValueError("Splat/volume scene mismatch")
        volume = label["plant_solid_volume_m3"]
        if not math.isfinite(volume) or volume <= 0 or label["units"] != "m3":
            raise ValueError("Invalid solid volume label")
        manifest = read_json(sample / "splat.manifest.json")
        metrics = read_json(sample / "metrics.json")
        if manifest["scene_id"] != actual_id or metrics["scene_id"] != actual_id:
            raise ValueError("Trained artifacts belong to another scene")
        if manifest["coordinate_frame"] != "ENU_m_Z_up" or manifest["scope"] != "full_scene":
            raise ValueError("Unexpected PLY units or scope")
        if manifest["capture_marker_sha256"] != sha256(sample / "CAPTURE_COMPLETE.json"):
            raise ValueError("Training was based on a different capture")
        if metrics.get("state") != "complete" or metrics.get("evaluation_is_withheld_from_splat_loss") is not True:
            raise ValueError("A completed fit and held-out evaluation are required")
        if not manifest.get("trained") and not (allow_fixture and scene.get("test_fixture")):
            raise ValueError("Untrained artifact cannot enter the dataset")
        if manifest["reconstruction_recipe"] != metrics["reconstruction_recipe"]:
            raise ValueError("Recipe mismatch")
        psnr, ssim = metrics["test"]["mean_psnr_db"], metrics["test"]["mean_ssim"]
        if not math.isfinite(psnr) or not math.isfinite(ssim):
            raise ValueError('Nonfinite reconstruction metrics')
        if quality_policy == 'thresholds' and (psnr < min_psnr or ssim < min_ssim):
            raise ValueError("Reconstruction below configured held-out quality thresholds")
        ply = validate_ply(sample / "splat.ply")
        if ply["sha256"] != manifest["sha256"]:
            raise ValueError("PLY checksum mismatch")
        row = {"sample_id": sample_id, "scene_id": actual_id, "scene_family_id": sample_metadata["scene_family_id"],
               "split": sample_metadata["split"], "splat_path": f"samples/{sample_id}/splat.ply", "splat_sha256": ply["sha256"],
               "label_path": f"samples/{sample_id}/labels/volume.json", "label_sha256": sha256(sample / "labels/volume.json"),
               "plant_solid_volume_m3": volume, "volume_method_version": label["method"],
               "reconstruction_recipe": manifest["reconstruction_recipe"], "test_fixture": scene.get("test_fixture", False),
               "quality_thresholds": {"min_psnr_db": min_psnr, "min_ssim": min_ssim}}
        if quality_policy == 'record_only':
            # Scores describe the pair; they never select which fields enter the dataset.
            metadata = {'schema': 'cropvolume-pair-metadata-v1', 'sample_id': sample_id,
                'scene_id': actual_id, 'seed': scene.get('seed'), 'scene_family_id': row['scene_family_id'],
                'split': row['split'], 'field': scene.get('resolved_config', {}),
                'plant_count': len(scene.get('instances', [])), 'instances': scene.get('instances', []),
                'coordinate_frame': scene['coordinate_frame'], 'volume_units': 'm3',
                'plant_solid_volume_m3': volume, 'label_method': label['method'],
                'capture_settings': read_json(root / 'dataset.json').get('config', {}).get('capture', {}),
                'training_settings': read_json(root / 'dataset.json').get('config', {}).get('training', {}),
                'quality_policy': 'record_only', 'metrics': metrics,
                'scene_path': 'scene.json', 'capture_path': 'capture.json', 'label_path': 'labels/volume.json',
                'splat_path': 'splat.ply', 'splat_sha256': ply['sha256'], 'label_sha256': row['label_sha256']}
            metadata_path = sample / 'metadata.json'
            if scene.get('field', {}).get('backend') == 'unreal_basesim_editor':
                metadata.update(label_scope=scene['label_scope'],
                    target_region={'center_enu_m':scene['center_enu_m'],'patch_m':scene['patch_m']},
                    render_environment=scene['field']['source_map'])
            if (sample / 'COMPLETE.json').exists() and (not metadata_path.exists() or read_json(metadata_path) != metadata):
                raise ValueError('Accepted pair metadata is immutable')
            atomic_json(metadata_path, metadata)
            row.update(quality_policy='record_only', quality_thresholds=None,
                metadata_path=f'samples/{sample_id}/metadata.json', metadata_sha256=sha256(metadata_path))
            if scene.get('field', {}).get('backend') == 'unreal_basesim_editor':
                pairing = {'label_scope':scene['label_scope'],
                    'target_region':{'center_enu_m':scene['center_enu_m'],'patch_m':scene['patch_m']},
                    'target_instance_ids':[i['instance_id'] for i in scene['instances']],
                    'render_environment':scene['field']['source_map']}
                manifest.update(pairing)
                atomic_json(sample/'splat.manifest.json',manifest)
                row.update(pairing,splat_scope=manifest['scope'])
        if authored:
            pairing = {key: sample_metadata[key] for key in ('label_scope', 'target_region', 'source_region_id',
                                                             'target_instance_ids', 'source_volume_receipt_sha256')}
            for key, value in pairing.items():
                if key in manifest and manifest[key] != value:
                    raise ValueError('Splat pairing scope differs from the authored target region')
            manifest.update(pairing)
            atomic_json(sample / 'splat.manifest.json', manifest)
            row.update(pairing, splat_scope=manifest['scope'])
        complete_path = sample / "COMPLETE.json"
        if complete_path.exists() and read_json(complete_path) != row:
            raise ValueError("Accepted sample is immutable; create a new sample for a new fit")
        atomic_json(complete_path, row)
    rebuild_index(root, allow_fixture=allow_fixture)
    return row


@profiled('indexing', root_arg='root')
def rebuild_index(root, allow_fixture=False):
    root = Path(root)
    with lease(root / "locks" / "index"):
        rows, families = [], {}
        for path in sorted((root / "samples").glob("*/COMPLETE.json")):
            if (path.parent / "RETIRED.json").exists():
                continue
            row = read_json(path)
            if row.get("test_fixture") and not allow_fixture:
                continue
            for key, digest_key in (("splat_path", "splat_sha256"), ("label_path", "label_sha256")):
                if sha256(child(root, row[key])) != row[digest_key]:
                    raise ValueError("Accepted artifact was modified: " + row[key])
            if row.get('metadata_path') and sha256(child(root, row['metadata_path'])) != row['metadata_sha256']:
                raise ValueError('Accepted metadata was modified: ' + row['metadata_path'])
            family = row["scene_family_id"]
            if family in families and families[family] != row["split"]:
                raise ValueError("Scene family leaked between learning splits")
            families[family] = row["split"]
            rows.append(row)
        # Index is a rebuildable view; immutable COMPLETE receipts are the source of truth.
        temp = root / "index.jsonl.tmp"
        with temp.open("wb") as stream:
            for row in rows:
                stream.write(canonical(row) + b"\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())
        temp.replace(root / "index.jsonl")
        atomic_json(root / "splits.json", {split: [r["sample_id"] for r in rows if r["split"] == split]
                                         for split in ("train", "validation", "test")})
        return rows
