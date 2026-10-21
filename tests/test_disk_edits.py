import numpy as np
import pytest

from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.boundary import BoundaryKind, extract_boundary
from oxygen_shape.geometry.constraints import ShapeConstraints, validate_shape
from oxygen_shape.geometry.disk_edits import apply_disk, boundary_samples, propose_paired_disk


def test_paired_disks_preserve_area_and_are_actual_boolean_operations():
    g = matched_area("deep_cavity", 100).geometry
    rng = np.random.default_rng(42)
    valid = 0
    constraints = ShapeConstraints(0.24, 4.0)
    for _ in range(150):
        proposal = propose_paired_disk(g, rng, 0.06, 0.12)
        if proposal.geometry is None:
            continue
        assert proposal.geometry.inside.sum() == g.inside.sum()
        p = proposal.parameters
        addition = apply_disk(g, np.array([p["add_x"], p["add_y"]]), p["radius"], add=True)
        expected = apply_disk(
            addition, np.array([p["remove_x"], p["remove_y"]]), p["remove_radius"], add=False
        )
        np.testing.assert_array_equal(expected.inside, proposal.geometry.inside)
        valid += validate_shape(proposal.geometry, constraints, g).valid
    assert valid > 0


def test_boundary_sampling_covers_unique_equal_measure_faces():
    g = matched_area("deep_cavity", 100).geometry
    points, normals = boundary_samples(g, 0.12)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
    assert points[:, 0].min() >= 0.12
    eligible = [
        face
        for face in extract_boundary(g)
        if face.kind == BoundaryKind.ROBIN and (face.key.col + 0.5) * g.h >= 0.12 + 2 * g.h
    ]
    assert len(points) == len(eligible)
    assert len(np.unique(points, axis=0)) == len(points)


def test_invalid_disks_are_rejected_without_clipping():
    g = matched_area("deep_cavity", 100).geometry
    with pytest.raises(ValueError, match="3h"):
        apply_disk(g, np.array([0.8, 0.5]), 0.01, add=True)
    with pytest.raises(ValueError, match="box"):
        apply_disk(g, np.array([0.99, 0.5]), 0.1, add=True)
