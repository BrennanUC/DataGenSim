"""Export or select the authored field while preserving every source transform."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]

from cropvolume.configuration import configuration

from cropvolume.batch import windows_path
from cropvolume.field import prepare_inventory_export, seal_inventory, select_region, verify_region
from cropvolume.profiling import run_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('plan-export', 'export'):
        p = commands.add_parser(action)
        p.add_argument('output', type=Path)
        p.add_argument('--reference', type=Path, required=True)
        p.add_argument('--runtime-root', type=Path, default=configuration()['runtime_root']/'runtime/BaseSimReference')
        p.add_argument('--engine-root', default=configuration()['engine_root'])
    commands.add_parser('verify-inventory').add_argument('inventory', type=Path)
    for action in ('select', 'verify-region'):
        p = commands.add_parser(action)
        p.add_argument('inventory', type=Path)
        p.add_argument('output', type=Path)
        if action == 'select':
            p.add_argument('--center-enu-m', type=float, nargs=2, default=[50., 0.])
            p.add_argument('--patch-m', type=float, nargs=2, default=[2., 2.])
            p.add_argument('--context-margin-m', type=float, default=1.)
    args = parser.parse_args()
    if args.action in ('plan-export', 'export'):
        prepare_inventory_export(args.output, args.reference)
        if args.action == 'export':
            run_process(['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                '-File', windows_path(ROOT / 'tools/run_field_reference.ps1'), '-Action', 'export',
                '-Output', windows_path(args.output), '-Runtime', windows_path(args.runtime_root),
                '-EngineRoot', args.engine_root], root=args.output, check=True)
            result = seal_inventory(args.output)
        else:
            result = {'state': 'export_planned', 'output': str(args.output)}
    elif args.action == 'verify-inventory':
        result = seal_inventory(args.inventory)
    else:
        if args.action == 'select':
            select_region(args.inventory, args.output, args.center_enu_m, args.patch_m, args.context_margin_m)
        result = verify_region(args.output, args.inventory)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))


if __name__ == '__main__':
    main()
