"""Generate labeled procedural test scenes without Unreal, CUDA, or fabricated splats."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
import numpy as np
import trimesh
from PIL import Image, ImageDraw

from cropvolume.cli import plan_batch
from cropvolume.geometry import audit_catalog, solid_union, load_catalog, load_mesh
from cropvolume.io import atomic_json, child, read_json
from cropvolume.cameras import look_at


def preview(scene, catalog_path, output):
    assets = {a["id"]: a for a in load_catalog(catalog_path)["assets"]}
    view = np.linalg.inv(look_at([2.5, -3., 3.4], [0., 0., .1]))
    image = Image.new("RGB", (1000, 800), (229, 224, 211))
    draw = ImageDraw.Draw(image)
    triangles = []
    for instance in scene["instances"]:
        asset = assets[instance["asset_id"]]
        mesh = load_mesh(child(catalog_path.parent, asset["mesh"]))
        mesh.apply_transform(np.array(instance["local_to_world_m"]))
        camera = mesh.vertices @ view[:3, :3].T + view[:3, 3]
        projected = camera[:, :2] / camera[:, 2, None] * 900 + [500, 400]
        for face_index, face in enumerate(mesh.faces):
            triangles.append((camera[face, 2].mean(), projected[face], mesh.face_normals[face_index][2]))
    all_points = np.concatenate([t[1] for t in triangles])
    low, high = all_points.min(0), all_points.max(0)
    zoom = min(900 / max(high[0] - low[0], 1), 650 / max(high[1] - low[1], 1))
    middle = (low + high) / 2
    for depth, points, lighting in sorted(triangles, key=lambda x: -x[0]):
        points = (points - middle) * zoom + [500, 450]
        green = int(95 + 45 * abs(lighting))
        draw.polygon([tuple(p) for p in points], fill=(45, green, 38))
    draw.text((18, 18), "CPU procedural geometry fixture - not a reconstructed Gaussian splat", fill=(20, 35, 20))
    image.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/cpu_demo")
    parser.add_argument("--count", type=int, default=25)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "assets.json").exists():
        raise SystemExit("Demo output already exists; choose a new --output")
    specs = []
    for i in range(3):
        stem = trimesh.creation.cylinder(radius=.006, height=.22, sections=12)
        stem.apply_translation([0, 0, .11])
        parts = [stem]
        for j in range(6 + i):
            leaf = trimesh.creation.icosphere(subdivisions=1)
            leaf.apply_scale([.055 + i * .007, .018, .003])
            angle = j * 2.4
            transform = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
            transform[:3, 3] = [np.cos(angle) * .035, np.sin(angle) * .035, .04 + j * .022]
            leaf.apply_transform(transform)
            parts.append(leaf)
        plant = solid_union(parts)
        mesh_name = f"procedural_plant_{i}.ply"
        plant.export(args.output / mesh_name)
        specs.append({"id": f"basil_demo_{i}", "species": "basil", "mesh": mesh_name, "units": "m",
                      "geometry_version": "procedural-test-v1", "geometry_matches_render": True,
                      "material_geometry_reviewed": True, "test_fixture": True})
    atomic_json(args.output / "assets.json", {"assets": specs})
    catalog = args.output / "catalog.json"
    audit_catalog(args.output / "assets.json", catalog)
    config = {"reserve_bytes": 0, "scene": {"plant_count": [4, 10], "patch_m": [1., 1.],
              "scale": [.7, 1.4], "anisotropic": True}, "capture": {"views": 64, "width": 1024, "height": 768}}
    plan_batch(args.output / "dataset", catalog, config, args.count, 923, peak_bytes=1000000)
    samples = sorted((args.output / "dataset/samples").iterdir())
    rows = [{"sample_id": p.name, "plants": len(read_json(p / "scene.json")["instances"]),
             "plant_solid_volume_m3": read_json(p / "labels/volume.json")["plant_solid_volume_m3"]} for p in samples]
    preview(read_json(samples[0] / "scene.json"), catalog, args.output / "preview.png")
    report = {"test_fixture": True, "scenes": len(rows), "approved_procedural_assets": 3,
              "real_unreal_assets_audited": 0, "splats_trained": 0,
              "volume_range_m3": [min(r["plant_solid_volume_m3"] for r in rows), max(r["plant_solid_volume_m3"] for r in rows)],
              "samples": rows, "notice": "CPU geometry demonstration only. No Unreal rendering or Gaussian optimization performed."}
    atomic_json(args.output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
