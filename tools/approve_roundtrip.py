"""Approve only imported static solids whose final render geometry matches its recipe."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from cropvolume.geometry import audit_catalog, load_mesh, solid_union
from cropvolume.io import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    source = read_json(args.export / "assets.review.json")
    checks = []
    for spec in source["assets"]:
        recipe = spec["solid_recipe"]
        if spec["nanite_enabled"] is not False:
            raise ValueError("Final geometry must use the audited non-Nanite render mesh")
        for mat in spec["material_inspection"]:
            if "inspection_error" in mat or mat["opacity_mask_connected"] or "BLEND_OPAQUE" not in mat["blend_mode"]:
                raise ValueError("Unreviewed or masked material")
            if mat["displacement_connected"] and not mat["displacement_constant_zero"]:
                raise ValueError("Geometry is moved by a shader")
        mesh = load_mesh(args.export / spec["mesh"])
        volume = float(solid_union([mesh]).volume)
        expected = recipe["expected_volume_m3"]
        relative = abs(volume - expected) / expected
        if relative > .0001:
            raise ValueError(f"Volume changed during Unreal import for {spec['id']}: {relative}")
        spec.update(geometry_matches_render=True, material_geometry_reviewed=True,
                    geometry_version="solid-shell-v1-" + recipe["geometry_id"])
        checks.append({"asset_id": spec["id"], "volume_m3": volume, "relative_roundtrip_error": relative,
                       "leaf_thickness_m": recipe["leaf_thickness_m"]})
    reviewed = args.export / "assets.approved.json"
    atomic_json(reviewed, source)
    catalog = audit_catalog(reviewed, args.catalog)
    if catalog["rejected"] or len(catalog["assets"]) != len(source["assets"]):
        raise ValueError("Catalog audit rejected one or more imported solids")
    atomic_json(args.catalog.parent / "roundtrip_report.json", {"assets": checks, "approved": len(checks),
                "max_relative_error": max(r["relative_roundtrip_error"] for r in checks),
                "notice": "Solid volumes use a configurable synthetic leaf thickness of 0.3 mm; no biological thickness measurement supplied."})
    print("Approved", len(checks), "rendered solid assets; maximum relative volume error", max(r["relative_roundtrip_error"] for r in checks))


if __name__ == "__main__":
    main()
