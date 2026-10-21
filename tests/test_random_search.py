import numpy as np

from oxygen_shape.geometry.constraints import validate_shape
from oxygen_shape.optimization.config import SearchConfig
from oxygen_shape.optimization.random_search import constraints_for, run_search
from oxygen_shape.optimization.refinement import refine_geometry
from oxygen_shape.solvers.finite_volume import solve


def test_seeded_search_improves_and_reproduces_under_constraints():
    config = SearchConfig(solve_budget=11, seed=42)
    a = run_search(config)
    b = run_search(config)
    assert a.number_of_pde_solves == b.number_of_pde_solves == 11
    np.testing.assert_array_equal(a.final_geometry.inside, b.final_geometry.inside)
    assert a.proposals == b.proposals
    assert (
        a.final_solution.metrics.physical_flux_Q > 1.02 * a.initial_solution.metrics.physical_flux_Q
    )
    assert validate_shape(a.final_geometry, constraints_for(config), a.initial_geometry).valid
    assert np.all(np.diff([r["physical_flux_Q"] for r in a.history]) >= 0)
    assert {r["area"] for r in a.history} == {a.initial_geometry.area}
    assert max(r["robin_perimeter"] for r in a.history) <= config.perimeter_limit
    # Gain must persist when the same physical cell union is refined.
    fine_initial = refine_geometry(a.initial_geometry, 2)
    fine_final = refine_geometry(a.final_geometry, 2)
    validate_shape(fine_final, constraints_for(config), fine_initial).require_valid()
    assert (
        solve(fine_final, config.model).metrics.physical_flux_Q
        > solve(fine_initial, config.model).metrics.physical_flux_Q
    )
