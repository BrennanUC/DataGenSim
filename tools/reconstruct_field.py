"""Use the existing preparation, fitting and publication pipeline on an authored field capture."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]

from cropvolume.configuration import configuration

from cropvolume.authored_capture import stage_volume_capture
from cropvolume.worker import work_sample
from cropvolume.profiling import span, annotate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'prepare', 'fit', 'verify'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=configuration()['output_root']/'authored-field-pilot')
    args = parser.parse_args()
    with span('reconstruction_run', root=args.root, action=args.action):
        sample_id = stage_volume_capture(args.source, args.root)
        annotate(sample_id=sample_id)
        if args.action == 'stage':
            result = {'state': 'capture_complete', 'sample_id': sample_id}
        else:
            if args.action == 'verify' and not (args.root / 'samples' / sample_id / 'COMPLETE.json').exists():
                parser.error('verify requires a published pair; use prepare or fit first')
            result = work_sample(args.root, sample_id, 'rgb_known_pose', prepare_only=args.action == 'prepare')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
