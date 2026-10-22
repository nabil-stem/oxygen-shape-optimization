import numpy as np
import pytest

from oxygen_shape.geometry.square_holes import SquareHole, hole_boundary, maze_chamber, perforate
from oxygen_shape.physics.model import DiffusionModel
from oxygen_shape.solvers.finite_volume import solve
from scripts.audit_square_holes import face_replay, verify_fem_boundary


@pytest.mark.parametrize("reflecting", [False, True])
def test_independent_replay_checks_saved_flux_and_boundary_labels(reflecting):
    pytest.importorskip("triangle")
    from oxygen_shape.solvers.fem_reference import solve_raster_fem

    geometry = perforate(maze_chamber(), (SquareHole(32, 28, 8),))
    if reflecting:
        geometry = hole_boundary(geometry, "neumann")
    fvm = solve(geometry, DiffusionModel(Bi=1))
    q, source, perimeter, residual = face_replay(geometry.inside, fvm.concentration, 1, reflecting)
    assert q == pytest.approx(fvm.metrics.physical_flux_Q, abs=1e-12)
    assert source == pytest.approx(q, abs=1e-11)
    assert residual < 1e-11
    fem = solve_raster_fem(geometry, 1, 0.001)
    replay_perimeter = verify_fem_boundary(
        fem.vertices, fem.boundary_segments, fem.boundary_markers, geometry.inside, reflecting
    )
    assert replay_perimeter == pytest.approx(perimeter, abs=1e-12)
    changed = fem.boundary_markers.copy()
    changed[np.flatnonzero(changed == 2)[0]] = 3
    with pytest.raises(AssertionError):
        verify_fem_boundary(
            fem.vertices, fem.boundary_segments, changed, geometry.inside, reflecting
        )


def test_independent_face_residual_detects_changed_interior_concentration():
    geometry = maze_chamber()
    result = solve(geometry, DiffusionModel(Bi=1))
    changed = result.concentration.copy()
    changed[30, 40] += 0.01
    _, _, _, residual = face_replay(geometry.inside, changed, 1, False)
    assert residual > 0.01
