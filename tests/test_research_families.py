import numpy as np
import pytest

from oxygen_shape.geometry.constraints import ShapeConstraints
from oxygen_shape.geometry.research_families import FAMILIES, make_research_geometry, quadratic_koch
from oxygen_shape.optimization.guided_search import validate_candidate
from oxygen_shape.optimization.trajectory_search import TrajectoryConfig


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", [0, 310001])
def test_family_area_source_connectivity_and_multiscale_features(family, seed):
    geometry = make_research_geometry(family, seed=seed).geometry
    config = TrajectoryConfig()
    assert abs(geometry.area - 0.24) < 1e-12
    assert len(geometry.source_faces) * geometry.h == pytest.approx(0.1)
    assert validate_candidate(geometry, geometry, config) == ""
    assert ShapeConstraints(0.24, 8).forbid_holes


def test_source_collar_is_identical_across_families_and_seeds():
    masks = [
        make_research_geometry(f, seed=s).geometry.inside[:, :12]
        for f in FAMILIES
        for s in (0, 310001)
    ]
    assert all(np.array_equal(masks[0], mask) for mask in masks[1:])


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_koch_generator_has_declared_finite_length(level):
    curve = quadratic_koch(level)
    assert len(curve) == 5**level + 1
    assert np.linalg.norm(np.diff(curve, axis=0), axis=1).sum() == pytest.approx((5 / 3) ** level)
    np.testing.assert_array_equal(curve[[0, -1]], [[0, 0], [1, 0]])


def test_family_variants_reproduce_and_change_distal_geometry():
    first = make_research_geometry("hexagon", seed=310001).geometry
    again = make_research_geometry("hexagon", seed=310001).geometry
    other = make_research_geometry("hexagon", seed=310002).geometry
    np.testing.assert_array_equal(first.inside, again.inside)
    assert not np.array_equal(first.inside, other.inside)
