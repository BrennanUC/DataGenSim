# Third-party provenance

The original `Basesim` and `Basegsplat` directories are not runtime dependencies.

- The field map, plants, soil and materials in `simulator/FlightSchool/Content` come from the supplied CROPSIM Flight School project. Only the Unreal dependency closure for the basil field and the four audited solid variants is retained. Original package hashes, mesh hashes and geometry/volume IDs are recorded in `assets/provenance.json` and `assets/plants/catalog.json`. The supplied notice is retained in `docs/upstream/CROPSIM-NOTICES.txt`.
- `src/cropvolume/vendor/train_real_splat.py` is copied unchanged from the supplied Basegsplat trainer. The Gaussian capacity adapter is derived from its optional region refit code. See `pipeline/UPSTREAM.md`.
- `third_party/gsplat-rocm` contains HPC-Ken's gsplat fork at commit `2a907639c1777c68f48c4a792232714d8e8cb854`, including its Apache-2.0 [license](third_party/gsplat-rocm/LICENSE) and bundled third-party notices. Build caches and compiled GPU binaries are excluded. The Python source hashes are checked at runtime against `src/cropvolume/profiles/rocm.json`.

The supplied CROPSIM/Basegsplat components retain their authors' terms. No project-wide license has been selected for CropVolume.
