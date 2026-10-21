import pytest

from oxygen_shape.physics.model import DiffusionModel
from oxygen_shape.solvers.finite_volume import solve
from scripts.source_corner_validation import strip_flux, strip_geometry


def test_spectral_tail_contains_a_more_complete_series_and_membrane_limit():
    short, long = strip_flux(100, terms=128), strip_flux(100, terms=2048)
    assert short["lower"] <= long["partial_Q"] <= short["upper"]
    small = strip_flux(1e-5, terms=128)
    assert small["partial_Q"] / 1e-5 == pytest.approx(1, abs=5e-4)


@pytest.mark.parametrize("Bi", [0.1, 1.0, 10.0, 100.0])
def test_independent_analytical_2d_flux_and_corner_refinement(Bi):
    exact = strip_flux(Bi, terms=2048)["partial_Q"]
    errors = []
    for n in (100, 200, 400):
        field = solve(strip_geometry(n), DiffusionModel(Bi=Bi))
        errors.append(abs(field.metrics.physical_flux_Q - exact))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] / exact < 0.025
