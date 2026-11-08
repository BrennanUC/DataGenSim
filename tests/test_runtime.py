from types import SimpleNamespace
import pytest

from cropvolume.runtime import gpu_backend, validate_gsplat
from cropvolume.io import atomic_json, sha256


def fake_torch(hip=None, cuda=None, available=True):
    return SimpleNamespace(version=SimpleNamespace(hip=hip, cuda=cuda),
                           cuda=SimpleNamespace(is_available=lambda: available))


def test_rocm_uses_cuda_device_api_without_cuda_runtime():
    assert gpu_backend(fake_torch(hip="7.2", cuda=None)) == "rocm"
    assert gpu_backend(fake_torch(cuda="12.8")) == "cuda"
    with pytest.raises(RuntimeError, match="cannot access"):
        gpu_backend(fake_torch(hip="7.2", available=False))


def test_cuda_profile_remains_pinned():
    assert validate_gsplat(SimpleNamespace(__version__="1.5.3"), "cuda")["gsplat"] == "1.5.3"
    with pytest.raises(RuntimeError, match="profile"):
        validate_gsplat(SimpleNamespace(__version__="1.6.0"), "cuda")


def test_rocm_source_changes_invalidate_the_profile(tmp_path):
    source = tmp_path / "gsplat"
    source.mkdir()
    module = source / "__init__.py"
    module.write_text("# pinned source\n")
    profile = {"gsplat_version": "1.6.0", "source_commit": "test-source-commit",
               "python_files_sha256": {"__init__.py": sha256(module)}}
    atomic_json(tmp_path / "profile.json", profile)
    gsplat = SimpleNamespace(__version__="1.6.0", __file__=str(module))
    assert validate_gsplat(gsplat, "rocm", tmp_path / "profile.json")["profile"] == profile
    module.write_text("# different source\n")
    with pytest.raises(RuntimeError, match="source differs"):
        validate_gsplat(gsplat, "rocm", tmp_path / "profile.json")
