"""Build the small row-anchor cache from a verified original-field inventory."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src')]
import numpy as np
from cropvolume.field import load_inventory
from cropvolume.io import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'configs/basesim_row_anchors.npz')
    args = parser.parse_args()
    data, rows = load_inventory(args.inventory)
    components = {c['component_id']:c for c in data['components']}
    selected = [(index,row) for index,row in enumerate(rows)
        if data['assets'][components[row['component_id']]['mesh']]['role']=='plant']
    radius = max(np.linalg.norm(np.abs(np.asarray(a['bounds_local_ue_cm'])[:,:2]).max(axis=0))*.01
                 for a in data['assets'].values() if a['role']=='plant')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,
        xy=np.array([[r['translation_ue_cm'][1]*.01,r['translation_ue_cm'][0]*.01] for _,r in selected]),
        source_indices=np.array([i for i,_ in selected]))
    atomic_json(args.output.with_suffix('.json'),{'schema':'cropvolume-basesim-row-anchors-v1',
        'source_map':data['source_map'],'source_map_sha256':data['source_map_sha256'],
        'inventory_id':data['inventory_id'],'source_instances_sha256':data['instances_sha256'],
        'anchors':len(selected),'npz_sha256':sha256(args.output),
        'maximum_clump_horizontal_radius_m':float(np.ceil(radius*100)/100)})
    print(args.output)


if __name__=='__main__':
    main()
