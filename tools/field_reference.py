"""Plan, capture, or verify the original BaseSim field without creating a dataset pair."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from cropvolume.configuration import configuration

from cropvolume.batch import windows_path
from cropvolume.reference import plan_reference, verify_reference
from cropvolume.profiling import run_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "capture", "verify"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--runtime-root", type=Path, default=configuration()['runtime_root']/'runtime/BaseSimReference')
    parser.add_argument("--engine-root", default=configuration()['engine_root'])
    parser.add_argument("--center-enu-m", type=float, nargs=2, default=[50., 0.])
    parser.add_argument("--patch-m", type=float, nargs=2, default=[2., 2.])
    parser.add_argument("--views", type=int, default=24)
    args = parser.parse_args()
    if args.action in ("plan", "capture"):
        plan_reference(args.output, args.center_enu_m, args.patch_m, args.views)
    if args.action == "capture":
        run_process(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                        "-File", windows_path(ROOT / "tools/run_field_reference.ps1"),
                        "-Output", windows_path(args.output), "-Runtime", windows_path(args.runtime_root),
                        "-EngineRoot", args.engine_root], root=args.output, check=True)
    if args.action in ("capture", "verify"):
        receipt = verify_reference(args.output)
        print(f"Verified {receipt['frames']} original-field reference images: {args.output}")
    else:
        print(f"Reference camera plan: {args.output}")


if __name__ == "__main__":
    main()
