"""Inventory binary Unreal assets without pretending to inspect their triangle topology."""
from pathlib import Path
import re

from .io import sha256


def inventory(base):
    root = Path(base)
    result = []
    for path in sorted(root.glob("*.uasset")):
        match = re.fullmatch(r"(basil|cilantro)_s(\d+)_v(\d+)", path.stem)
        if not match:
            continue
        result.append({"id": path.stem, "species": match[1], "stage": int(match[2]), "variant": int(match[3]),
                       "unreal_asset": "/Game/CROPSIM/Meshes/" + path.stem, "source_sha256": sha256(path),
                       "status": "requires_editor_export_and_geometry_review", "approved": False})
    return {"schema": "cropvolume-unreal-inventory-v1", "assets": result,
            "notice": "Binary asset inventory only; no enclosed volumes or material/topology approvals inferred."}
