"""GPU adapter around supplied trainer utilities; importable on a CPU-only machine."""
import importlib.util
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree

from .io import atomic_json, child, lease, read_json, sha256
from .prepare import verify_prepared
from .runtime import gpu_backend, validate_gsplat
from .profiling import profiled, span


def _initialization_indices(colors, count, rng, preserve_green):
    """Optional RGB-only stratification; no masks, geometry or volume labels."""
    if not preserve_green:
        return rng.choice(len(colors), count, replace=False)
    green = np.flatnonzero((colors[:, 1] > colors[:, 0] * 1.15) &
                           (colors[:, 1] > colors[:, 2] * 1.15) & (colors[:, 1] > .1))
    selected = rng.choice(green, min(len(green), count // 2), replace=False)
    remaining = np.setdiff1d(np.arange(len(colors)), selected)
    return np.concatenate([selected, rng.choice(remaining, count - len(selected), replace=False)])


def _capacity_strategy(cap, steps):
    import torch
    from gsplat import DefaultStrategy
    from gsplat.strategy.ops import duplicate, split

    class CappedStrategy(DefaultStrategy):
        # Adapted from the supplied regional trainer: each binary split/clone
        # increases count by one, so allocate only the available growth budget.
        @torch.no_grad()
        def _grow_gs(self, params, optimizers, state, step):
            count = len(params["means"])
            available = max(0, cap - count)
            gradients = state["grad2d"] / state["count"].clamp_min(1)
            candidates = torch.where(torch.isfinite(gradients) & (gradients > self.grow_grad2d))[0]
            if len(candidates) > available:
                candidates = candidates[torch.topk(gradients[candidates], available).indices] if available else candidates[:0]
            chosen = torch.zeros(count, dtype=torch.bool, device=gradients.device)
            chosen[candidates] = True
            small = params["scales"].exp().max(dim=-1).values <= self.grow_scale3d * state["scene_scale"]
            clones, splits = chosen & small, chosen & ~small
            nc, ns = int(clones.sum()), int(splits.sum())
            if count + nc + ns > cap:
                raise RuntimeError("Gaussian growth exceeds allocation budget")
            if nc:
                duplicate(params, optimizers, state, clones)
                splits = torch.cat([splits, torch.zeros(nc, dtype=torch.bool, device=gradients.device)])
            if ns:
                split(params, optimizers, state, splits, revised_opacity=False)
            return nc, ns

        @torch.no_grad()
        def step_post_backward(self, params, optimizers, state, step, info, packed=False):
            # Explicit schedule avoids the supplied base's noted 1.5.3 reset condition bug.
            if step >= self.refine_stop_iter:
                return
            self._update_state(params, state, info, packed=packed)
            if step > self.refine_start_iter and step % self.refine_every == 0:
                self._prune_gs(params, optimizers, state, step)
                if not len(params["means"]):
                    raise RuntimeError("All Gaussians pruned")
                self._grow_gs(params, optimizers, state, step)
                state["grad2d"].zero_()
                state["count"].zero_()
                self.check_sanity(params, optimizers)
                if len(params["means"]) > cap:
                    raise RuntimeError("Gaussian budget violated")

    return CappedStrategy(refine_start_iter=min(500, steps // 8), refine_stop_iter=int(steps * .7),
                          refine_every=100, grow_grad2d=.0008, grow_scale3d=.002,
                          prune_scale3d=.1, absgrad=True, refine_scale2d_stop_iter=0)


@profiled('training', root_arg='prepared')
def train(prepared, output, steps=4000, init_points=30000, max_gaussians=100000, seed=923, preserve_green_init=False):
    # Do not read scene meshes or labels in this module.
    try:
        import torch
        import gsplat
        from torch import nn
        from gsplat.strategy.ops import reset_opa
        from pytorch_msssim import ssim
    except ImportError as error:
        raise RuntimeError(f"Training dependency is unavailable: {error}. Install the matching pipeline/requirements-training*.txt in the training environment and run tools/doctor.py --gpu") from error
    backend = gpu_backend(torch)
    runtime = validate_gsplat(gsplat, backend)
    if steps < 100 or min(init_points, max_gaussians) < 4 or init_points > max_gaussians:
        raise ValueError("Invalid training schedule or capacity")
    prepared, output = Path(prepared), Path(output)
    data = verify_prepared(prepared)
    if data["schema"] != "cropvolume-sim-known-pose-v1":
        raise ValueError("Unexpected prepared schema")
    if sha256(child(prepared, data["initialization"])) != data["initialization_sha256"]:
        raise ValueError("Initialization checksum changed")
    for relative, digest in data["prepared_images_sha256"].items():
        if sha256(child(prepared, relative)) != digest:
            raise ValueError("Prepared RGB changed")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists() or (output / "checkpoint.pt").exists():
        raise ValueError("Use a new training attempt; existing artifacts are immutable")
    vendor_path = Path(__file__).with_name("vendor") / "train_real_splat.py"
    spec = importlib.util.spec_from_file_location("cropvolume_legacy_trainer", vendor_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    with lease(output / ".training"):
        try:
            atomic_json(output / "status.json", {"state": "training"})
            with span('training_initialization', sync=torch.cuda.synchronize):
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                xyz = np.load(prepared / data["initialization"])
                points, colors = xyz["points"], xyz["colors"]
                count = min(len(points), init_points)
                colors = colors.astype(np.float32) / 255
                chosen = _initialization_indices(colors, count, rng, preserve_green_init)
                points, colors = points[chosen], colors[chosen]
                d = cKDTree(points).query(points, k=4)[0][:, 1:]
                scale = np.sqrt((d ** 2).mean(1)).clip(.0001, .05)
                quats = rng.normal(size=(count, 4)).astype(np.float32)
                quats /= np.linalg.norm(quats, axis=1, keepdims=True)
                values = {"means": points, "scales": np.log(np.repeat(scale[:, None], 3, 1)), "quats": quats,
                          "opacities": np.full(count, np.log(.1 / .9)), "sh0": ((colors - .5) / base.C0)[:, None],
                          "shN": np.zeros((count, 15, 3))}
                params = nn.ParameterDict({k: nn.Parameter(torch.as_tensor(v, device="cuda", dtype=torch.float32)) for k, v in values.items()})
                rates = {"means": 1.6e-4 * data["scene_scale_m"], "scales": .005, "quats": .001,
                         "opacities": .05, "sh0": .0025, "shN": .000125}
                opts = {k: torch.optim.Adam([params[k]], lr=v, eps=1e-15) for k, v in rates.items()}
                strategy = _capacity_strategy(max_gaussians, steps)
                strategy.check_sanity(params, opts)
                state = strategy.initialize_state(scene_scale=data["scene_scale_m"])
                width = max(f["width"] for f in data["frames"])
                images = base.Images(prepared, data["frames"], width, cache_limit_gb=1)
                train_ids = np.array([i for i, f in enumerate(data["frames"]) if f["split"] == "train"])
                validation = [i for i, f in enumerate(data["frames"]) if f["split"] == "validation"]
                test_ids = [i for i, f in enumerate(data["frames"]) if f["split"] == "test"]
                if len(train_ids) < 4 or not validation or not test_ids:
                    raise ValueError("Not enough split views")
                views = torch.tensor(np.linalg.inv(np.array([f["c2w_cv"] for f in data["frames"]], np.float32)), device="cuda")
                order, cursor = rng.permutation(train_ids), 0
                history = []
                started = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                reset_every = max(1000, steps // 3)
                atomic_json(output / "config.json", {"steps": steps, "seed": seed, "init_points": init_points,
                            "max_gaussians": max_gaussians, "preserve_green_init": preserve_green_init,
                            "prepared_sha256": sha256(prepared / "dataset.json"),
                            "adapter_sha256": sha256(__file__), "vendor_sha256": sha256(vendor_path), "runtime": runtime})
            with span('optimization', total=steps, unit='steps', sync=torch.cuda.synchronize) as optimization:
                for step in range(1, steps + 1):
                    if cursor == len(order):
                        order, cursor = rng.permutation(train_ids), 0
                    i = int(order[cursor])
                    cursor += 1
                    w, h, K, _ = base.camera(data["frames"][i], width)
                    truth = torch.tensor(images.get(i), device="cuda", dtype=torch.float32)[None] / 255
                    color, _, info = base.render(params, views[i:i + 1], torch.tensor(K, device="cuda")[None], w, h,
                                                 degree=min(3, step // max(1, steps // 4)), absgrad=True)
                    loss = .8 * (color - truth).abs().mean() + .2 * (1 - ssim(color.permute(0, 3, 1, 2), truth.permute(0, 3, 1, 2), data_range=1))
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite optimization loss")
                    strategy.step_pre_backward(params, opts, state, step, info)
                    loss.backward()
                    for opt in opts.values():
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    opts["means"].param_groups[0]["lr"] = rates["means"] * .01 ** (step / steps)
                    strategy.step_post_backward(params, opts, state, step, info)
                    if step % reset_every == 0 and step < strategy.refine_stop_iter:
                        reset_opa(params, opts, state, value=.01)
                    if step == 1 or step % 100 == 0:
                        history.append({"step": step, "loss": float(loss.detach()), "gaussians": len(params["means"])})
                        atomic_json(output / "history.json", history)
                        optimization.update(current=step, loss=history[-1]["loss"], gaussians=len(params["means"]))
            with span("validation_evaluation", total=len(validation), unit="views", sync=torch.cuda.synchronize):
                base.evaluate(params, data, images, validation, output, steps, width, label="validation")
            with span("test_evaluation", total=len(test_ids), unit="views", sync=torch.cuda.synchronize):
                final = base.evaluate(params, data, images, test_ids, output, steps, width, label="test")
            with span('splat_export', sync=torch.cuda.synchronize):
                manifest = base.export_ply(output / "splat.ply", params)
                manifest.update({"trained": True, "scene_id": data["scene_id"], "coordinate_frame": "ENU_m_Z_up",
                                 "world_coordinates": "Exact simulator ENU meters, Z up", "scope": "full_scene",
                                 "reconstruction_recipe": data["reconstruction_recipe"],
                                 "capture_marker_sha256": data["capture_marker_sha256"]})
                manifest["runtime"] = runtime
                atomic_json(output / "splat.manifest.json", manifest)
                torch.save({"params": {k: v.detach().cpu() for k, v in params.items()}, "step": steps}, output / "checkpoint.pt")
            result = {"state": "complete", "scene_id": data["scene_id"], "test": final,
                      "reconstruction_recipe": data["reconstruction_recipe"], "evaluation_is_withheld_from_splat_loss": True,
                      "elapsed_s": time.monotonic() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                      "gaussians": len(params["means"]), "gsplat": gsplat.__version__, "torch": torch.__version__,
                      "runtime": runtime, "gpu": torch.cuda.get_device_name(0)}
            atomic_json(output / "metrics.json", result)
            atomic_json(output / "status.json", {"state": "complete"})
            return result
        except BaseException as error:
            atomic_json(output / "status.json", {"state": "failed", "error": str(error)})
            raise
