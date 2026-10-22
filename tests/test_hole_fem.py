import numpy as np
import pytest

pytest.importorskip("triangle")

from oxygen_shape.experiments.morphology_validation import cached_fem
from oxygen_shape.geometry.generators import rectangle
from oxygen_shape.geometry.raster import refine_geometry
from oxygen_shape.geometry.square_holes import SquareHole, hole_boundary, perforate
from oxygen_shape.physics.model import DiffusionModel
from oxygen_shape.solvers.fem_reference import raster_pslg, solve_pslg, solve_raster_fem
from oxygen_shape.solvers.finite_volume import solve


def annulus_pslg(segments):
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    unit = np.c_[np.cos(theta), np.sin(theta)]
    outer = np.c_[np.arange(segments), np.roll(np.arange(segments), -1)]
    return dict(
        vertices=np.vstack([unit, 0.3 * unit]),
        segments=np.vstack([outer, outer + segments]),
        segment_markers=np.r_[np.ones(segments), 2 * np.ones(segments)].astype(int)[:, None],
        holes=np.array([[0.0, 0.0]]),
    )


@pytest.mark.parametrize("Bi", [0.0, 1.0, 10.0])
def test_analytical_annulus_validates_inner_normal_and_reactive_flux(Bi):
    exact_Q = 2 * np.pi * 0.3 * Bi / (1 + Bi * 0.3 * np.log(1 / 0.3))
    errors = []
    for segments, area in [(32, 0.003), (128, 0.0003)]:
        result = solve_pslg(annulus_pslg(segments), Bi, area)
        radius = np.linalg.norm(result.vertices, axis=1)
        exact = (1 + Bi * 0.3 * np.log(radius / 0.3)) / (1 + Bi * 0.3 * np.log(1 / 0.3))
        errors.append(abs(result.Q - exact_Q))
        assert result.energy == pytest.approx(result.Q, abs=1e-10)
        assert np.max(np.abs(result.concentration - exact)) < 0.005
    assert errors[-1] <= errors[0] + 1e-12
    assert errors[-1] < max(1e-12, 0.001 * exact_Q)


def test_square_hole_fem_does_not_fill_the_island_and_checks_reflecting_control():
    outer = rectangle(30, 20, 0.025, slab=True)
    g = perforate(outer, (SquareHole(8, 12, 4),), gap=0.05)
    assert raster_pslg(g)["holes"].shape == (1, 2)
    fem = solve_raster_fem(g, 1, 0.0002)
    assert fem.area == pytest.approx(g.area, abs=1e-12)
    assert fem.robin_perimeter == pytest.approx(0.9)
    errors = [
        abs(solve(x, DiffusionModel(Bi=1)).metrics.physical_flux_Q - fem.Q)
        for x in (g, refine_geometry(g, 4))
    ]
    assert errors[1] < errors[0]
    assert errors[1] / fem.Q < 0.01
    reflected = solve_raster_fem(hole_boundary(g, "neumann"), 1, 0.0002)
    assert reflected.area == pytest.approx(g.area)
    assert reflected.robin_perimeter == pytest.approx(0.5)
    assert reflected.Q < solve_raster_fem(outer, 1, 0.0002).Q
    assert reflected.Q < fem.Q


def test_fem_cache_identity_includes_hole_seeds(tmp_path):
    pslg = annulus_pslg(24)
    path = tmp_path / "annulus.npz"
    cached_fem(path, pslg, 1.0, 0.005)
    changed = {**pslg, "holes": np.array([[0.01, 0.01]])}
    with pytest.raises(ValueError, match="identity mismatch"):
        cached_fem(path, changed, 1.0, 0.005)
