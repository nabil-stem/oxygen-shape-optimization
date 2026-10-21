import numpy as np
import pytest
from matplotlib.colors import PowerNorm

from oxygen_shape.experiments.source_amplitude import solve_source_amplitude
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.generators import rectangle
from oxygen_shape.physics.model import DiffusionModel
from oxygen_shape.solvers.analytical_1d import slab_flux, slab_solution
from oxygen_shape.solvers.finite_volume import solve
from oxygen_shape.viz.concentration import (
    ConcentrationPanel,
    concentration_norm,
    plot_contrast_comparison,
)


@pytest.mark.parametrize("Bi", [0, 0.1, 1, 10])
def test_independently_solved_slab_source_scaling_matches_analytical_solution(Bi):
    geometry = rectangle(64, 3, 1 / 64, slab=True)
    x = (np.arange(64) + 0.5) / 64
    for amplitude in (1, 2, 5):
        result = solve_source_amplitude(geometry, DiffusionModel(Bi=Bi), amplitude)
        exact = np.broadcast_to(amplitude * slab_solution(x, Bi), (3, 64))
        np.testing.assert_allclose(result.v, exact, atol=2e-11)
        assert result.flux / (3 / 64) == pytest.approx(amplitude * slab_flux(Bi), abs=2e-11)
        assert result.normalized_residual < 1e-10
        assert result.conservation_error < 1e-9


@pytest.mark.parametrize("Bi", [0.1, 1, 10])
def test_irregular_shape_normalized_fields_efficiency_and_ranking_are_invariant(Bi):
    rankings = []
    for amplitude in (1, 2, 5):
        values = []
        for family in ("deep_cavity", "rectangle"):
            geometry = matched_area(family, 100, 0.24).geometry
            model = DiffusionModel(Bi=Bi)
            reference = solve(geometry, model)
            result = solve_source_amplitude(geometry, model, amplitude)
            np.testing.assert_allclose(
                result.normalized_u, reference.concentration, atol=2e-12, equal_nan=True
            )
            assert result.normalized_flux == pytest.approx(
                reference.metrics.physical_flux_Q, abs=2e-12
            )
            assert result.eta == pytest.approx(reference.metrics.efficiency_eta, abs=2e-12)
            assert result.conservation_error < 1e-9
            values.append(result.flux)
        rankings.append(np.argsort(values).tolist())
    assert rankings[0] == rankings[1] == rankings[2]


def test_nonzero_blood_concentration_scales_the_driving_difference():
    u = np.array([1.0, 0.2, 0.01])
    blood, source = 0.3, 1.3
    old = blood + (source - blood) * u
    new_source = blood + 2 * (source - blood)
    new = blood + (new_source - blood) * u
    np.testing.assert_allclose((new - blood) / (new_source - blood), u)
    np.testing.assert_allclose(new - blood, 2 * (old - blood))


def test_contrast_has_fixed_shared_bounds_and_does_not_modify_fields(tmp_path):
    geometry = rectangle(8, 3, 1 / 8, slab=True)
    field = solve(geometry, DiffusionModel()).concentration
    before = field.copy()
    field.flags.writeable = False
    norm = concentration_norm("power", 0.35)
    assert isinstance(norm, PowerNorm) and norm.vmin == 0 and norm.vmax == 1
    assert norm(0.01) == pytest.approx(0.01**0.35)
    assert norm(0.01) > concentration_norm("linear")(0.01)
    plot_contrast_comparison(
        [ConcentrationPanel(geometry, field, "Same physical field")], tmp_path / "contrast.png"
    )
    np.testing.assert_array_equal(field, before)
    assert (tmp_path / "contrast.png").is_file()
    for gamma in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError):
            concentration_norm("power", gamma)
    for amplitude in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError):
            solve_source_amplitude(geometry, DiffusionModel(), amplitude)
