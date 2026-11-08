import pytest
import trimesh

from cropvolume.geometry import solid_union, load_mesh


def test_cached_boolean_cavity_is_preserved_on_reuse(tmp_path):
    outer=trimesh.creation.box(extents=[2,2,2])
    inner=trimesh.creation.box(extents=[1,1,1])
    hollow=trimesh.boolean.difference([outer,inner],engine='manifold')
    assert hollow.is_volume
    hollow.export(tmp_path/'cached.ply')
    cached=load_mesh(tmp_path/'cached.ply',process=False)
    result=solid_union([cached,cached.copy()],preserve_csg_boundaries=True)
    assert result.volume==pytest.approx(7)
    # Arbitrary inverted raw components remain rejected by the source audit.
    inner.invert()
    with pytest.raises(ValueError):
        solid_union([inner])
