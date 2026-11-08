import argparse
import importlib.util
import json
from pathlib import Path
import sys

from .cameras import capture_plan, export_cameras
from .geometry import audit_catalog, measure_scene
from .inventory import inventory
from .io import atomic_json, identity, lease, read_json
from .scenes import generate
from .storage import DEFAULT_ROOT, Storage
from .configuration import configuration
from .profiling import profiled, span


@profiled('batch_planning', root_arg='root')
def plan_batch(root, catalog, config, count, seed, peak_bytes):
    root = Path(root)
    if (root / "DATASET_RETIRED.json").exists():
        raise ValueError("Dataset root retired after placement validation; use the corrected game dataset")
    storage = Storage(root, config.get("budget_bytes", 800_000_000_000), config.get("reserve_bytes", 100_000_000_000))
    catalog_id = read_json(catalog)["catalog_id"]
    batch_identity = identity({"catalog_sha": catalog_id, "config": config, "seed": seed})
    with lease(root / "locks" / "metadata"):
        metadata = {"schema": "cropvolume-dataset-v1", "root_is_configurable": True,
                    "catalog_id": catalog_id, "volume_units": "m3", "config": config}
        if (root / "dataset.json").exists() and read_json(root / "dataset.json") != metadata:
            raise ValueError("Dataset configuration is immutable; use a new dataset root")
        atomic_json(root / "dataset.json", metadata)
    sample_ids = []
    for i in range(count):
        sample_id = f"{batch_identity[:12]}_{i:06d}"
        sample_ids.append(sample_id)
        sample = root / "samples" / sample_id
        if (sample / "PLAN_COMPLETE.json").exists():
            marker = read_json(sample / "PLAN_COMPLETE.json")
            if marker["batch_id"] != batch_identity:
                raise ValueError("Existing sample belongs to another batch")
            continue
        storage.reserve_job(sample_id, peak_bytes)
        try:
            with lease(root / "locks" / sample_id):
                with span('scene_generation', sample_id=sample_id):
                    scene = generate(catalog, seed + i, config["scene"])
                with span('volume_measurement', sample_id=sample_id):
                    label = measure_scene(scene, catalog)
                sample.mkdir(parents=True, exist_ok=True)
                atomic_json(sample / "scene.json", scene)
                atomic_json(sample / "labels/volume.json", label)
                capture = capture_plan(scene, **config.get("capture", {}))
                atomic_json(sample / "capture.json", capture)
                export_cameras(sample, capture["frames"])
                atomic_json(sample / "PLAN_COMPLETE.json", {"batch_id": batch_identity, "scene_id": scene["scene_id"],
                                                           "sample_id": sample_id, "state": "awaiting_capture"})
        finally:
            # No capture/training is started by planning; workers acquire their own peak reservation.
            storage.release_job(sample_id)
    return sample_ids


def main():
    parser = argparse.ArgumentParser(description="CropVolume calibrated plant-volume dataset pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory")
    inv.add_argument("meshes", type=Path)
    inv.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("source", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    batch = commands.add_parser("plan-batch")
    batch.add_argument("--catalog", type=Path, required=True)
    batch.add_argument("--config", type=Path, required=True)
    batch.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    batch.add_argument("--count", type=int, default=1)
    batch.add_argument("--seed", type=int, default=923)
    batch.add_argument("--peak-bytes", type=int, default=2_000_000_000)
    run_batch = commands.add_parser("batch", help="Plan, capture, prepare and optionally train a local batch")
    run_batch.add_argument("--catalog", type=Path, required=True)
    run_batch.add_argument("--config", type=Path, required=True)
    run_batch.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    run_batch.add_argument("--runtime-root", type=Path, default=configuration()['runtime_root'])
    run_batch.add_argument("--engine-root", default=configuration()['engine_root'])
    run_batch.add_argument("--count", type=int, default=1)
    run_batch.add_argument("--seed", type=int, default=923)
    run_batch.add_argument("--peak-bytes", type=int, default=2_000_000_000)
    run_batch.add_argument("--capture-timeout", type=int, default=900)
    run_batch.add_argument("--train", action="store_true")
    run_batch.add_argument("--min-psnr", type=float, default=15.)
    run_batch.add_argument("--min-ssim", type=float, default=.3)
    prep = commands.add_parser("prepare")
    prep.add_argument("sample", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--recipe", choices=["rgb_known_pose", "depth_assisted"], default="rgb_known_pose")
    prep.add_argument("--max-width", type=int, default=1024)
    prep.add_argument("--max-points", type=int, default=30000)
    seal = commands.add_parser("seal-capture")
    seal.add_argument("sample", type=Path)
    train = commands.add_parser("train")
    train.add_argument("prepared", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int, default=4000)
    train.add_argument("--init-points", type=int, default=30000)
    train.add_argument("--max-gaussians", type=int, default=100000)
    work = commands.add_parser("work")
    work.add_argument("sample_id")
    work.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    work.add_argument("--recipe", choices=["rgb_known_pose", "depth_assisted"], default="rgb_known_pose")
    work.add_argument("--train", action="store_true", help="Run GPU training after CPU preparation")
    work.add_argument("--min-psnr", type=float, default=15.)
    work.add_argument("--min-ssim", type=float, default=.3)
    pub = commands.add_parser("publish")
    pub.add_argument("sample_id")
    pub.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    pub.add_argument("--min-psnr", type=float, required=True)
    pub.add_argument("--min-ssim", type=float, required=True)
    index = commands.add_parser("reindex")
    index.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    commands.add_parser("doctor")
    args = parser.parse_args()
    if args.command == "inventory":
        result = inventory(args.meshes)
        atomic_json(args.output, result)
        print(json.dumps({"candidates": len(result["assets"]), "approved": 0, "output": str(args.output)}))
    elif args.command == "audit":
        result = audit_catalog(args.source, args.output)
        print(json.dumps({"approved": len(result["assets"]), "rejected": result["rejected"]}))
    elif args.command == "plan-batch":
        if args.count < 1:
            parser.error("count must be positive")
        plan_batch(args.root, args.catalog, read_json(args.config), args.count, args.seed, args.peak_bytes)
    elif args.command == "batch":
        from .batch import run_batch
        if args.count < 1 or not 1 <= args.capture_timeout <= 7200:
            parser.error("count must be positive and capture timeout must be 1–7200 seconds")
        result = run_batch(args.root, args.catalog, read_json(args.config), args.count, args.seed,
                           args.runtime_root, args.engine_root, args.train, args.peak_bytes,
                           args.capture_timeout, args.min_psnr, args.min_ssim)
        print(json.dumps(result))
        if result["failed"] or result["paused"]:
            raise SystemExit(2)
    elif args.command == "prepare":
        from .prepare import prepare
        data = prepare(args.sample, args.output, args.recipe, args.max_width, args.max_points)
        print(json.dumps({"prepared": str(args.output), "frames": len(data["frames"]), "points": data["point_count"]}))
    elif args.command == "seal-capture":
        from .prepare import seal_capture
        seal_capture(args.sample)
    elif args.command == "train":
        from .training import train
        print(json.dumps(train(args.prepared, args.output, args.steps, args.init_points, args.max_gaussians)))
    elif args.command == "work":
        from .worker import work_sample
        print(json.dumps(work_sample(args.root, args.sample_id, args.recipe, not args.train,
                                    min_psnr=args.min_psnr, min_ssim=args.min_ssim)))
    elif args.command == "publish":
        from .dataset import publish
        print(json.dumps(publish(args.root, args.sample_id, args.min_psnr, args.min_ssim)))
    elif args.command == "reindex":
        from .dataset import rebuild_index
        print(json.dumps({"accepted": len(rebuild_index(args.root))}))
    else:
        root = DEFAULT_ROOT
        result = {"python": sys.version, "dataset_root": str(root), "dataset_parent_exists": root.parent.exists(),
                  "modules": {name: importlib.util.find_spec(name) is not None for name in
                              ["numpy", "trimesh", "manifold3d", "PIL", "pycolmap", "torch", "gsplat"]}}
        if result["modules"]["torch"]:
            import torch
            result["gpu_available"] = torch.cuda.is_available()
            result["torch_hip"] = torch.version.hip
            result["torch_cuda"] = torch.version.cuda
            result["backend"] = "rocm" if torch.version.hip else "cuda" if torch.version.cuda else "cpu"
        print(json.dumps(result, indent=2))
