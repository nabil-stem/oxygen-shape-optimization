import numpy as np
import pytest

pytest.importorskip("triangle")

from oxygen_shape.geometry.generators import rectangle
from oxygen_shape.solvers.fem_reference import solve_pslg, solve_raster_fem


@pytest.mark.parametrize("Bi", [0.0, 0.01, 1.0, 100.0])
def test_independent_fem_slab_concentration_flux_and_energy(Bi):
    g = rectangle(10, 3, 0.1, slab=True)
    result = solve_raster_fem(g, Bi, 0.001)
    exact = 1 - Bi / (1 + Bi) * result.vertices[:, 0]
    np.testing.assert_allclose(result.concentration, exact, atol=1e-11)
    assert result.Q == pytest.approx(0.3 * Bi / (1 + Bi), abs=2e-11)
    assert result.area == pytest.approx(0.3)
    assert result.robin_perimeter == pytest.approx(0.3)
    assert result.energy == pytest.approx(result.Q, abs=2e-11)
    assert result.conservation_error < 1e-9


def test_polygon_fem_is_consistent_under_rigid_rotation():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.2, 0.5], [0.8, 1.0], [0.0, 0.8]])
    segments = np.c_[np.arange(5), np.roll(np.arange(5), -1)]
    markers = np.array([2, 2, 2, 2, 1])[:, None]
    first = solve_pslg(
        dict(vertices=points, segments=segments, segment_markers=markers), 1.0, 0.0005
    )
    angle = 0.371
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    second = solve_pslg(
        dict(vertices=points @ rotation.T + 2, segments=segments, segment_markers=markers),
        1.0,
        0.0005,
    )
    assert first.Q == pytest.approx(second.Q, rel=3e-4)
    assert first.area == pytest.approx(second.area, abs=1e-10)
    assert first.energy == pytest.approx(first.Q, abs=1e-10)


def test_fem_rejects_coincident_boundary_vertices_before_assembly():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="Coincident"):
        solve_pslg(
            dict(
                vertices=vertices,
                segments=np.array([[0, 1], [1, 2], [2, 3]]),
                segment_markers=np.array([[1], [2], [2]]),
            ),
            1.0,
        )
