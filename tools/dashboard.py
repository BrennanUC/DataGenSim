"""Start the local dataset generator and progress dashboard, or export its state."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from cropvolume.dashboard import Monitor, serve
from cropvolume.generation import GenerationManager
from cropvolume.storage import DEFAULT_ROOT
from cropvolume.configuration import configuration,local_config_path


def main():
    config=configuration()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', type=Path, help='Dataset root or parent containing dataset roots; repeatable')
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--generation-root', type=Path, default=config['output_root'])
    parser.add_argument('--catalog', type=Path, default=ROOT / 'assets/plants/catalog.json')
    parser.add_argument('--runtime-root', type=Path, default=None)
    parser.add_argument('--engine-root', default=None)
    parser.add_argument('--export', type=Path, help='Write a JSON status/timing report instead of starting the server')
    args = parser.parse_args()
    roots = args.root or [p for p in [config['output_root'], *[Path(p) for p in config.get('output_roots',[])]] if p.exists()]
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(Monitor(roots, args.artifacts).snapshot(), indent=2), encoding='utf-8')
        print(args.export)
    else:
        if not 0 <= args.port <= 65535:
            parser.error('port must be between 0 and 65535')
        generation = GenerationManager(args.generation_root, args.catalog, args.runtime_root, args.engine_root,
            preferences=local_config_path(),output_parents=config.get('output_roots',[]))
        roots = list(dict.fromkeys([*roots, *[p for p in generation.output_parents if p.exists()]]))
        serve(roots, args.artifacts, args.port, generation)


if __name__ == '__main__':
    main()
