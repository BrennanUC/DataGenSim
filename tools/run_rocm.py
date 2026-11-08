"""Compatibility entry point for the ROCm backend."""
import os
from pathlib import Path
import runpy
os.environ.setdefault('CROPVOLUME_GPU_BACKEND','rocm')
runpy.run_path(str(Path(__file__).with_name('run_training.py')),run_name='__main__')
