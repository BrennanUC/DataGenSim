"""Build solid recipes from the supplied simulator's full source-mesh export."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from cropvolume.io import atomic_json, child, read_json
from cropvolume.solidify import solidify_export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Unreal SOURCE_MODEL assets.review.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--leaf-thickness-m", type=float, default=.0003)
    parser.add_argument("--cap-stems", action="store_true", help="Explicitly cap planar convex stem-tube ends")
    parser.add_argument("--asset-suffix", default="_solid")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use an empty output directory to preserve existing geometry versions")
    rows = []
    for spec in read_json(args.source)["assets"]:
        if not spec["geometry_version"].startswith("source-") or spec["units"] != "m":
            raise ValueError("Use the full source mesh in meters, not a Nanite fallback")
        supported = [["/Game/CROPSIM/Materials/M_Leaf.M_Leaf", "/Game/CROPSIM/Materials/M_Stem.M_Stem"],
                     ["/Game/FlightSchool/Materials/MI_BasilLeaf.MI_BasilLeaf", "/Game/FlightSchool/Materials/M_BasilStem.M_BasilStem"]]
        if spec["materials"] not in supported:
            raise ValueError("Review new material slots before assigning leaf thickness")
        relative = spec["id"] + ".json"
        geometry = solidify_export(child(args.source.parent, spec["mesh"]), child(args.output, relative),
                                   args.leaf_thickness_m, args.cap_stems)
        rows.append({"id": spec["id"], "species": spec["species"], "asset_suffix": args.asset_suffix,
                     "geometry": relative, "geometry_id": geometry["geometry_id"],
                     "source_asset": spec["unreal_asset"], "source_materials": spec["materials"],
                     "leaf_thickness_m": geometry["leaf_thickness_m"],
                     "volume_m3": geometry["plant_solid_volume_m3"]})
    atomic_json(args.output / "manifest.json", {"assets": rows,
                "status": "requires_unreal_import_and_final_render_mesh_audit"})
    print("Prepared", len(rows), "solid recipes; Unreal round-trip approval is still required")


if __name__ == "__main__":
    main()
