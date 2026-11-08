"""Exercise the installed gsplat rasterizer and autograd; records actual GPU backend."""
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    start = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch has no accessible CUDA/HIP GPU")
    # Check lightweight training helpers before paying the first JIT build cost.
    from pytorch_msssim import ssim
    import gsplat
    from gsplat.strategy.ops import reset_opa
    from cropvolume.runtime import gpu_backend, validate_gsplat
    runtime = validate_gsplat(gsplat, gpu_backend(torch))
    from gsplat import rasterization
    means = torch.tensor([[0., 0., 2.], [.15, .1, 2.3]], device="cuda", requires_grad=True)
    quats = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]], device="cuda", requires_grad=True)
    scales = torch.full((2, 3), .08, device="cuda", requires_grad=True)
    opacity = torch.full((2,), .8, device="cuda", requires_grad=True)
    colors = torch.tensor([[.2, .8, .1], [.8, .3, .2]], device="cuda", requires_grad=True)
    view = torch.eye(4, device="cuda")[None]
    K = torch.tensor([[[55., 0., 32.], [0., 55., 32.], [0., 0., 1.]]], device="cuda")
    rendered, alpha, info = rasterization(means, quats, scales, opacity, colors, view, K, 64, 64,
                                         packed=False, absgrad=True)
    loss = rendered.square().mean() + alpha.mean()
    loss.backward()
    torch.cuda.synchronize()
    for name, tensor in {"means": means, "quats": quats, "scales": scales, "opacity": opacity, "colors": colors}.items():
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise RuntimeError("Missing or nonfinite gradient: " + name)
    if not torch.isfinite(rendered).all() or alpha.max() <= 0:
        raise RuntimeError("Invalid render")
    result = {"state": "passed", "torch": torch.__version__, "gsplat": gsplat.__version__,
              "gsplat_source": str(Path(gsplat.__file__).resolve()), "hip": torch.version.hip, "cuda": torch.version.cuda,
              "backend": runtime["backend"], "runtime": runtime, "gpu": torch.cuda.get_device_name(0),
              "render_shape": list(rendered.shape), "loss": float(loss.detach()),
              "means_grad_norm": float(means.grad.norm()), "absgrad_available": hasattr(info["means2d"], "absgrad"),
              "training_dependencies_verified": True, "elapsed_s": time.monotonic() - start}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
