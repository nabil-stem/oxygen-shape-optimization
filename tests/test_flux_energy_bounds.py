import numpy as np
import pytest

from oxygen_shape.geometry.generators import rectangle
from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.physics.model import DiffusionModel
from oxygen_shape.solvers.fem_reference import solve_raster_fem
from oxygen_shape.solvers.finite_volume import solve
from scripts.flux_energy_bounds import lower_flux_estimate, primal_upper_estimate


@pytest.mark.parametrize("Bi", [0.01, 1.0, 100.0])
def test_complementary_estimate_recovers_exact_slab(Bi):
    geometry = rectangle(20, 4, 0.05, slab=True)
    result = solve(geometry, DiffusionModel(Bi=Bi))
    bound = lower_flux_estimate(geometry, result.concentration, Bi)
    exact = 0.2 * Bi / (1 + Bi)
    assert bound.complementary_value == pytest.approx(exact, abs=2e-12)
    assert bound.lower <= exact + 1e-12
    assert bound.divergence_residual_max < 1e-12


@pytest.mark.parametrize("level", [-0.5, 0.0, 0.3, 0.9, 1.0, 1.5])
def test_residual_corrected_bound_holds_for_unsolved_slab_fields(level):
    geometry = rectangle(10, 3, 0.1, slab=True)
    field = np.full(geometry.inside.shape, level)
    bound = lower_flux_estimate(geometry, field, 1.0)
    assert bound.lower <= 0.15 + 1e-12


@pytest.mark.parametrize("family", ["rectangle", "hexagon", "maze"])
def test_dual_lower_and_independent_primal_fem_upper_are_consistent(family):
    pytest.importorskip("triangle")
    geometry = make_research_geometry(family).geometry
    fv = solve(geometry, DiffusionModel())
    bound = lower_flux_estimate(geometry, fv.concentration, 1.0)
    fem = solve_raster_fem(geometry, 1.0, 5e-5)
    assert bound.lower <= fem.energy + 1e-10
    assert bound.complementary_value >= fv.metrics.physical_flux_Q - 1e-10
    assert fem.energy == pytest.approx(fem.Q, abs=1e-10)
    upper, energy, guard = primal_upper_estimate(
        fem.vertices,
        fem.triangles,
        fem.concentration,
        fem.boundary_segments,
        fem.boundary_markers,
        1.0,
    )
    assert energy == pytest.approx(fem.energy, abs=1e-10)
    assert upper >= energy and guard > 0
    assert bound.lower <= upper
