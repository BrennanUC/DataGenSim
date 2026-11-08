"""Save machine-specific paths without putting them in the source release."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src')]
from cropvolume.configuration import configuration,local_config_path,save_preferences


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output-root','runtime-root','engine-root','training-python','gsplat-source','cache-root'):
        parser.add_argument('--'+key)
    parser.add_argument('--backend',choices=['rocm','cuda'])
    args=parser.parse_args()
    changes={k:v for k,v in vars(args).items() if v is not None}
    if changes:save_preferences(changes)
    config=configuration()
    print('Configuration:',local_config_path())
    for key in ('output_root','runtime_root','engine_root','training_python','backend'):
        print(f'{key}: {config[key]}')


if __name__=='__main__':main()
