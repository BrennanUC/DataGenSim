"""Distinguish PyTorch's CUDA-shaped device API from its actual CUDA/HIP backend."""
from pathlib import Path

from .io import identity, read_json, sha256


def gpu_backend(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access a GPU; check the CUDA/ROCm runtime and device permissions")
    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    raise RuntimeError("Unrecognized PyTorch GPU backend")


def validate_gsplat(gsplat, backend, profile_path=None):
    if backend == "cuda":
        if gsplat.__version__ != "1.5.3":
            raise RuntimeError("CUDA capacity adapter requires the supplied gsplat==1.5.3 profile")
        return {"backend": backend, "gsplat": gsplat.__version__}
    if backend != "rocm":
        raise RuntimeError("Unsupported GPU backend: " + backend)
    profile_path = Path(profile_path) if profile_path else Path(__file__).with_name("profiles") / "rocm.json"
    profile = read_json(profile_path)
    if gsplat.__version__ != profile["gsplat_version"]:
        raise RuntimeError("ROCm gsplat version does not match the pinned Radeon profile")
    root = Path(gsplat.__file__).resolve().parent
    for relative, digest in profile["python_files_sha256"].items():
        if sha256(root / relative) != digest:
            raise RuntimeError("ROCm source differs from pinned Radeon profile: " + relative)
    return {"backend": backend, "gsplat": gsplat.__version__, "source_commit": profile["source_commit"],
            "profile_sha256": identity(profile), "profile": profile, "source": str(root)}
