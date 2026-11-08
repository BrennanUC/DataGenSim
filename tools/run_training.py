"""Launch the configured CUDA or ROCm Python with isolated build caches."""
import os
from pathlib import Path
import runpy
import site
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cropvolume.configuration import configuration


def main():
    config=configuration()
    source=config['gsplat_source']
    python=config['training_python']
    if not os.environ.get('CROPVOLUME_TRAINING_CHILD'):
        env=dict(os.environ)
        env.update(CROPVOLUME_TRAINING_CHILD='1', PYTHONDONTWRITEBYTECODE='1',
            TORCH_EXTENSIONS_DIR=str(config['cache_root']/'torch_extensions'))
        paths=[str(ROOT/'src'),*[p for p in site.getsitepackages() if Path(p).is_dir()]]
        paths += [p for p in os.environ.get('PYTHONPATH','').split(os.pathsep) if p]
        if config['backend']=='rocm':
            paths.insert(1,str(source))
            for key,value in {'HSA_ENABLE_DXG_DETECTION':'1','ROCM_HOME':'/opt/rocm',
                'MAX_JOBS':'2','BUILD_3DGUT':'0','BUILD_2DGS':'0','BUILD_LOSSES':'0',
                'BUILD_ADAM':'0','BUILD_RELOC':'1','BUILD_3DGS':'1'}.items():
                env.setdefault(key,value)
            env['PATH']=str(python.parent)+os.pathsep+str(Path(env['ROCM_HOME'])/'bin')+os.pathsep+env.get('PATH','')
            env['LD_LIBRARY_PATH']=str(Path(env['ROCM_HOME'])/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
        env['PYTHONPATH']=os.pathsep.join(dict.fromkeys(paths))
        raise SystemExit(subprocess.call([str(python),__file__,*sys.argv[1:]],env=env))
    if config['backend']=='rocm':
        # PyTorch's JIT hipify scans extra header directories even when kernels are
        # already HIP. Do not rewrite ROCm/system/third-party headers in place.
        import torch
        if not os.environ.get('PYTORCH_ROCM_ARCH') and torch.cuda.is_available():
            architectures={torch.cuda.get_device_properties(i).gcnArchName.split(':')[0]
                           for i in range(torch.cuda.device_count())}
            os.environ['PYTORCH_ROCM_ARCH']=';'.join(sorted(architectures))
        from torch.utils.hipify import hipify_python
        original = hipify_python.hipify

        def local_hipify(*args, **kwargs):
            files = kwargs.get("extra_files", [])
            native_root = source / "gsplat/hip"
            if files and all(Path(f).resolve().is_relative_to(native_root) for f in files):
                # This pinned fork ships native HIP sources. Running CUDA-to-HIP
                # translation again creates duplicate *_hip headers and definitions.
                from types import SimpleNamespace
                return {str(Path(f).resolve()): SimpleNamespace(hipified_path=str(Path(f).resolve())) for f in files}
            include = kwargs.get("header_include_dirs", [])
            keep, ignored = [], list(kwargs.get("ignores", []))
            for folder in include:
                path = Path(folder).resolve()
                if not path.is_relative_to(source) or "third_party" in path.parts:
                    ignored.extend([str(path), str(path) + "/*"])
                else:
                    keep.append(folder)
            kwargs["header_include_dirs"], kwargs["ignores"] = keep, ignored
            result = original(*args, **kwargs)
            # PyTorch 2.9's JIT caller differs from its wheel-build caller: it fails
            # to fall back to the original path for already-HIP/ignored inputs.
            for name, item in result.items():
                if item.hipified_path is None:
                    item.hipified_path = name
            return result

        hipify_python.hipify = local_hipify

    args=sys.argv[1:]
    if args[:1]==['--script']:
        script,rest=Path(args[1]).resolve(),args[2:]
    else:
        script,rest=ROOT/'run.py',args
    sys.argv=[str(script),*rest]
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':main()
