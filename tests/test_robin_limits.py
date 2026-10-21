import numpy as np
import pytest

from oxygen_shape.geometry.generators import rectangle
from oxygen_shape.physics.model import DiffusionModel
from oxygen_shape.solvers.finite_volume import robin_conductance, solve


def test_zero_permeability_is_exact_neumann_limit():
    result = solve(rectangle(15, 12, 0.05), DiffusionModel(Bi=0))
    np.testing.assert_array_equal(result.concentration, 1)
    assert result.metrics.physical_flux_Q == result.metrics.source_influx == 0
    assert result.metrics.conservation_error == 0
    assert result.metrics.efficiency_eta is None


def test_flux_monotone_and_transport_limit_for_separated_boundaries():
    # Slab avoids source/absorber corner singularities in the Bi->infinity limit.
    geometry = rectangle(32, 4, 1 / 32, slab=True)
    values = [
        solve(geometry, DiffusionModel(Bi=b)).metrics.physical_flux_Q / (4 / 32)
        for b in [0.01, 0.1, 1, 10, 100, 1e6]
    ]
    assert np.all(np.diff(values) > 0)
    assert values[-1] == pytest.approx(1, abs=2e-6)


def test_conductance_limits_and_resistance():
    assert robin_conductance(0, 0.1) == 0
    assert robin_conductance(2, 0.1) == pytest.approx(0.2 / 1.1)
    assert robin_conductance(1e300, 0.1) == pytest.approx(2)


@pytest.mark.parametrize("Bi", [-1, np.nan, np.inf])
def test_invalid_permeability_rejected(Bi):
    with pytest.raises(ValueError):
        DiffusionModel(Bi=Bi)
