"""Check a source checkout/release and optionally exercise its GPU runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cropvolume.configuration import configuration
from cropvolume.generation import GenerationManager
from cropvolume.geometry import load_catalog
from cropvolume.io import read_json,lease


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-only',action='store_true',help='Check the bundled assets without requiring Unreal/GPU setup')
    parser.add_argument('--gpu',action='store_true',help='Run actual GPU rasterization/autograd (first run compiles the extension)')
    args=parser.parse_args()
    catalog=load_catalog(ROOT/'assets/plants/catalog.json')
    for relative,digest in read_json(ROOT/'assets/provenance.json')['unreal_files'].items():
        path=ROOT/'simulator/FlightSchool/Content'/relative
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=digest:raise ValueError('Bundled Unreal asset changed: '+relative)
    print(f'PASS: {len(catalog["assets"])} approved solid meshes and all bundled Unreal packages verified',flush=True)
    if not args.assets_only:
        GenerationManager().preflight()
        print('PASS: Unreal, training interpreter and runtime paths configured',flush=True)
    if args.gpu:
        output=configuration()['cache_root']/'gpu-smoke.json'
        output.parent.mkdir(parents=True,exist_ok=True)
        with lease(configuration()['runtime_root']/'locks/dashboard_generation_worker'):
            subprocess.run([sys.executable,str(ROOT/'tools/run_training.py'),'--script',str(ROOT/'tools/gpu_smoke.py'),'--output',str(output)],check=True)
        print('PASS: GPU forward/backward rendering; details in '+str(output),flush=True)


if __name__=='__main__':main()
