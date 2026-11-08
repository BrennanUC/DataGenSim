# Configuration

Run `python tools/configure.py` to print resolved machine settings. Pass any of these flags to save them in `cropvolume.local.json`. Paths in a local configuration file may be absolute or relative to the project root. The dashboard's Output folder requires an absolute path (or `~`) to avoid ambiguity.

| Local key | CLI configuration flag | Environment override | Default |
| --- | --- | --- | --- |
| `output_root` | `--output-root` | `CROPVOLUME_OUTPUT_ROOT` | `<project>/data` |
| `runtime_root` | `--runtime-root` | `CROPVOLUME_RUNTIME_ROOT` | Output root; choose a Windows drive for WSL capture |
| `engine_root` | `--engine-root` | `CROPVOLUME_ENGINE_ROOT` | Must be configured |
| `training_python` | `--training-python` | `CROPVOLUME_TRAINING_PYTHON` | Current Python interpreter |
| `backend` | `--backend` | `CROPVOLUME_GPU_BACKEND` | `rocm`; alternative `cuda` |
| `gsplat_source` | `--gsplat-source` | `CROPVOLUME_GSPLAT_SOURCE` | Bundled `third_party/gsplat-rocm` |
| `cache_root` | `--cache-root` | `CROPVOLUME_CACHE_ROOT` | `~/.cache/cropvolume` |

`CROPVOLUME_CONFIG` selects a different local settings file. `CROPVOLUME_ROCM_PYTHON` remains an alias for the older launcher configuration; `CROPVOLUME_TRAINING_PYTHON` takes precedence. `PYTORCH_ROCM_ARCH` can explicitly select an AMD target; otherwise the launcher detects the installed GPU rather than compiling all architectures in the PyTorch wheel. `MAX_JOBS` controls compilation concurrency and defaults to two.

The dashboard's `--generation-root`, `--runtime-root`, `--engine-root` and `--catalog` arguments override their settings for that server. `--root` may be repeated to monitor other datasets. Changing the Output folder in the form records it and the previous roots so old runs remain manageable after restarting the dashboard. Explicit environment overrides still take precedence on restart.

On WSL, Windows drive paths such as `D:\datasets` are normalized to `/mnt/d/datasets`. Use the Windows installation path for Unreal, for example `C:/Program Files/Epic Games/UE_5.8`. Python/training and extension compilation run in WSL. Runtime project staging runs on a Windows drive; captured photographs can be written back to a WSL output folder.

The default catalog lives in `assets/plants/catalog.json`, independently of output. A custom `--catalog` must contain audited approved meshes and matching Unreal assets. Editing an existing catalog or generated scene invalidates its hashes. Historical asset rebuilding/field inspection helpers require their explicit source exports; the release's default generator needs none of those old exports.

For an unattended low-level batch, `configs/batch-example.json` contains a resolved example with a 5 GB free-space reserve:

```bash
python tools/run_training.py batch \
  --catalog assets/plants/catalog.json \
  --config configs/batch-example.json \
  --root /mnt/d/MyDataset/example-batch --count 1 --train
```

The runtime and engine default to the machine settings above. This direct batch command creates files at the exact `--root`; the dashboard launcher instead creates a new managed `random-field-...` directory under its Output folder.

The HTTP server is local-only: it binds to `127.0.0.1` and exposes no remote-host binding option. Run controls also require a local Host/Origin and the per-session token. No hosting, publication or deployment step is part of setup.
