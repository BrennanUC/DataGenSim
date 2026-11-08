"""Bind, render, and label the selected authored plants using audited solid versions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]

from cropvolume.configuration import configuration

from cropvolume.batch import windows_path
from cropvolume.field_volume import finalize_volume_region, plan_volume_region
from cropvolume.profiling import run_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'capture', 'finalize', 'verify'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--region', type=Path)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--views', type=int, help='Capture a denser balanced set instead of reusing reference camera poses')
    parser.add_argument('--catalog', type=Path, default=ROOT/'assets/plants/catalog.json')
    parser.add_argument('--recipes', type=Path, default=None)
    parser.add_argument('--source-export', type=Path, default=configuration()['runtime_root']/'assets/game_source')
    parser.add_argument('--runtime-root', type=Path, default=configuration()['runtime_root']/'runtime/BaseSimSolidRegion')
    parser.add_argument('--engine-root', default=configuration()['engine_root'])
    args = parser.parse_args()
    if args.action in ('plan', 'capture'):
        if not all((args.region, args.inventory, args.reference, args.recipes)):
            parser.error('plan/capture requires --region, --inventory, --reference and --recipes')
        scene = plan_volume_region(args.output, args.region, args.inventory, args.reference,
                                  args.catalog, args.recipes, args.source_export, args.views)
        result = {'state': 'planned', 'scene_id': scene['scene_id'], 'target_plants': len(scene['instances'])}
        if args.action == 'capture':
            run_process(['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                '-File', windows_path(ROOT / 'tools/run_field_reference.ps1'), '-Action', 'solid',
                '-Output', windows_path(args.output), '-Runtime', windows_path(args.runtime_root),
                '-EngineRoot', args.engine_root], root=args.output, check=True)
            result = finalize_volume_region(args.output)
    else:
        if args.action == 'verify' and not (args.output / 'VOLUME_COMPLETE.json').exists():
            parser.error('verify requires a completed volume region; use finalize for an unsealed render')
        result = finalize_volume_region(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))


if __name__ == '__main__':
    main()
