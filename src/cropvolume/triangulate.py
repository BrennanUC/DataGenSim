"""CPU COLMAP triangulation with authoritative simulator poses and intrinsics."""
from pathlib import Path
import numpy as np

from .cameras import export_cameras
from .io import atomic_json, child
from .profiling import profiled, span


@profiled('triangulation', root_arg='sample')
def triangulate(sample, output, frames):
    try:
        import pycolmap
    except ImportError as error:
        raise RuntimeError("rgb_known_pose requires pinned pycolmap==4.2.0 (CPU supported)") from error
    if pycolmap.__version__ != "4.2.0":
        raise RuntimeError("This fixed-pose adapter is validated against pycolmap==4.2.0")
    sample, output = Path(sample), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    database = output / "database.db"
    if database.exists():
        raise ValueError("Use a new triangulation attempt directory; do not reuse a partial database")
    training = [dict(f) for f in frames if f["split"] == "train"]
    names = [f["name"] for f in training]
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.use_gpu = False
    extraction.num_threads = 4
    extraction.max_image_size = 1600
    extraction.sift.max_num_features = 4096
    with span('feature_extraction', total=len(training), unit='views'):
        pycolmap.extract_features(database, sample / "images", image_names=names,
                                  camera_mode=pycolmap.CameraMode.PER_IMAGE,
                                  reader_options=reader, extraction_options=extraction, device=pycolmap.Device.cpu)
    with pycolmap.Database.open(database) as db:
        images = {im.name: im for im in db.read_all_images()}
        if set(images) != set(names):
            raise ValueError("Feature database contains missing or held-out views")
        for frame in training:
            image = images[frame["name"]]
            K = np.asarray(frame["K"])
            cam = db.read_camera(image.camera_id)
            cam.model = pycolmap.CameraModelId.PINHOLE
            cam.params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
            cam.width, cam.height = frame["width"], frame["height"]
            db.update_camera(cam)
            frame["image_id"], frame["camera_id"] = image.image_id, image.camera_id
    matching = pycolmap.FeatureMatchingOptions()
    matching.use_gpu = False
    matching.num_threads = 4
    with span('feature_matching', training_views=len(training)):
        pycolmap.match_exhaustive(database, matching_options=matching, device=pycolmap.Device.cpu)
    known = output / "known"
    export_cameras(known, training)
    model = pycolmap.Reconstruction(known / "sparse/0")
    before = {im.name: (im.cam_from_world().matrix().copy(), model.camera(im.camera_id).params.copy())
              for im in model.images.values()}
    options = pycolmap.IncrementalPipelineOptions()
    options.fix_existing_frames = True
    options.mapper.fix_existing_frames = True
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.ba_refine_sensor_from_rig = False
    options.constant_cameras = set(model.cameras)
    options.mapper.constant_cameras = set(model.cameras)
    options.num_threads = 4
    options.triangulation.ignore_two_view_tracks = True
    options.triangulation.min_angle = .5
    with span('fixed_pose_triangulation'):
        model = pycolmap.triangulate_points(model, database, sample / "images", output / "sparse",
                                           options=options, refine_intrinsics=False)
    worst = 0.
    for im in model.images.values():
        pose, intrinsic = before[im.name]
        worst = max(worst, float(np.max(np.abs(im.cam_from_world().matrix() - pose))))
        if not np.allclose(model.camera(im.camera_id).params, intrinsic, atol=1e-10, rtol=0):
            raise ValueError("COLMAP changed authoritative intrinsics")
    if worst > 1e-9:
        raise ValueError("COLMAP changed authoritative simulator poses")
    points = [p for p in model.points3D.values() if p.track.length() >= 3 and p.error <= 2.5 and np.isfinite(p.xyz).all()]
    if len(points) < 4:
        raise ValueError("Insufficient RGB triangulation; do not silently switch to simulator depth")
    atomic_json(output / "report.json", {"pose_source": "simulator_exact_fixed", "max_pose_change": worst,
                "training_views": names, "heldout_used": False, "points": len(points), "pycolmap": pycolmap.__version__})
    return np.asarray([p.xyz for p in points], np.float32), np.asarray([p.color for p in points], np.uint8)
