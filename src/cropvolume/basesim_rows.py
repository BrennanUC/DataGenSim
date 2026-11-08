"""Planting anchors exported from the supplied, verified BaseSim field."""
from functools import lru_cache
from pathlib import Path
import math
import numpy as np
from .io import read_json, sha256

ROOT = Path(__file__).resolve().parents[2] / 'configs'


@lru_cache(maxsize=1)
def anchors():
    path = ROOT / 'basesim_row_anchors.npz'
    proof = read_json(ROOT / 'basesim_row_anchors.json')
    if sha256(path) != proof['npz_sha256']:
        raise ValueError('BaseSim row anchor checksum mismatch')
    with np.load(path, allow_pickle=False) as data:
        xy, indices = data['xy'].copy(), data['source_indices'].copy()
    if xy.shape != (proof['anchors'], 2) or not np.isfinite(xy).all():
        raise ValueError('Invalid BaseSim row anchors')
    return xy, indices, proof


@lru_cache(maxsize=32)
def row_slots(width, length, scale_max=1.2, spacing=.08):
    xy, indices, proof = anchors()
    # Reserve enough space for every yaw of every approved field clump, so no
    # labeled plant is cut by the field boundary. Positions retain authored gaps.
    margin = proof['maximum_clump_horizontal_radius_m'] * scale_max
    local = xy - [50., 0.]
    allowed = (np.abs(local) <= np.array([width, length])*.5-margin-1e-8).all(axis=1)
    local, indices = local[allowed], indices[allowed]
    # Honor the dashboard spacing with a deterministic spatial hash, also when
    # the requested region contains thousands of anchors.
    result, cells = [], {}
    for p, index in zip(local, indices):
        if spacing:
            key = tuple(np.floor(p / spacing).astype(int))
            near = [q for x in range(key[0]-1,key[0]+2) for y in range(key[1]-1,key[1]+2)
                    for q in cells.get((x,y), ())]
            if any(np.linalg.norm(p-q) < spacing for q in near):
                continue
            cells.setdefault(key, []).append(p)
        result.append((float(p[0]),float(p[1]),int(index)))
        if len(result) > 10000:
            raise ValueError('Field exceeds the 10,000-plant limit')
    return tuple(result)


def row_count_range(width, length, scale_max=1.2, spacing=.08):
    count = len(row_slots(width, length, scale_max, spacing))
    if not count:
        raise ValueError('No BaseSim row plants fit this field size, mesh scale and spacing; enlarge the field or use scatter')
    if count > 10000:
        raise ValueError('Field exceeds the 10,000-plant limit')
    return [max(1, math.floor(count*.9)), count]
