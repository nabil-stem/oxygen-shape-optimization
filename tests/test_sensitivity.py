import numpy as np
import pytest

from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.constraints import validate_shape
from oxygen_shape.geometry.disk_edits import DiskEdit, prepare_boundary, propose_paired_disk
from oxygen_shape.optimization.guided_config import GuidedConfig, SensitivityBenchmarkConfig
from oxygen_shape.optimization.guided_search import run_guided_search
from oxygen_shape.optimization.random_search import constraints_for
from oxygen_shape.optimization.sensitivity import (
    PairResponse,
    flux_probabilities,
    mixture_probabilities,
    sensitivity_probabilities,
)
from oxygen_shape.solvers.finite_volume import solve


def test_signed_pair_response_matches_known_two_point_formula():
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    responses = [PairResponse((0, 0), (1, 0), 2, 2), PairResponse((1, 0), (0, 0), -4, 2)]
    assert responses[0].sensitivity == 1
    p = sensitivity_probabilities(points, responses, 0.1, 0.25)
    np.testing.assert_allclose(p, [[0.0625, 0.8125], [0.0625, 0.0625]], atol=1e-10)
    scaled = [
        PairResponse(r.addition, r.donor, 7 * r.delta_Q, r.reallocated_area) for r in responses
    ]
    np.testing.assert_allclose(p, sensitivity_probabilities(points, scaled, 0.1, 0.25))
    assert p.min() >= 0.25 / p.size
    assert p.sum() == pytest.approx(1)


def test_joint_map_does_not_assign_independent_credit_to_pair_members():
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    responses = [PairResponse((0, 0), (1, 0), 1, 1), PairResponse((1, 0), (0, 0), 1, 1)]
    p = sensitivity_probabilities(points, responses, 0.1, 0.25)
    assert p[0, 1] > 5 * p[0, 0]
    assert p[1, 0] > 5 * p[1, 1]


def test_uniform_limits_and_invalid_response_inputs():
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    negative = [PairResponse((0, 0), (1, 0), -1, 1)]
    for responses in ([], negative):
        np.testing.assert_allclose(sensitivity_probabilities(points, responses, 0.1, 0.25), 0.25)
    for flux in (np.ones(2), np.zeros(2)):
        np.testing.assert_allclose(flux_probabilities(flux, 0.25), 0.25)
    np.testing.assert_allclose(
        flux_probabilities(np.array([1.0, 0.0]), 0.25), [[0.0625, 0.8125], [0.0625, 0.0625]]
    )
    for bad in (np.array([-1]), np.array([np.nan])):
        with pytest.raises(ValueError):
            flux_probabilities(bad, 0.25)
    with pytest.raises(ValueError):
        PairResponse((0, 0), (1, 0), 1, 0)
    with pytest.raises(ValueError):
        sensitivity_probabilities(points, [], 0, 0.25)
    with pytest.raises(ValueError):
        mixture_probabilities(np.zeros((2, 2)), 0)


def test_cached_boundary_preserves_proposals_and_rejects_stale_geometry():
    geometry = matched_area("deep_cavity", 100).geometry
    prepared = prepare_boundary(geometry, 0.12)
    a, b = np.random.default_rng(9000), np.random.default_rng(9000)
    for _ in range(25):
        uncached = propose_paired_disk(geometry, a, 0.06, 0.12)
        cached = propose_paired_disk(geometry, b, 0.06, 0.12, prepared=prepared)
        assert uncached.parameters == cached.parameters
        assert uncached.rejection == cached.rejection
        if uncached.geometry is not None:
            np.testing.assert_array_equal(uncached.geometry.inside, cached.geometry.inside)
    with pytest.raises(ValueError, match="different geometry"):
        propose_paired_disk(geometry.with_mask(geometry.inside), a, 0.06, 0.12, prepared=prepared)
    with pytest.raises(ValueError, match="Face pair"):
        propose_paired_disk(geometry, a, 0.06, 0.12, face_pair=(-1, 0))


@pytest.mark.parametrize("method", ["uniform", "flux", "sensitivity"])
def test_equal_total_budgets_and_real_pde_objectives(method, monkeypatch):
    actual_solves = []

    def counted_solve(*args):
        result = solve(*args)
        actual_solves.append(result)
        return result

    monkeypatch.setattr("oxygen_shape.optimization.guided_search.solve", counted_solve)
    config = GuidedConfig(
        seed=9000,
        solve_budget=13,
        batch_size=6,
        probe_count=3,
        max_attempts_per_solve=100,
        refinement_factors=[2],
    )
    maps = []
    result = run_guided_search(config, method, on_map=lambda *args: maps.append(args))
    search = result.search
    assert result.status == "complete"
    assert search.number_of_pde_solves == len(actual_solves) == len(search.history) == 13
    assert [row["candidate_pde_solves"] for row in result.rounds] == [6, 6]
    assert (
        search.final_solution.metrics.physical_flux_Q
        >= search.initial_solution.metrics.physical_flux_Q
    )
    assert validate_shape(
        search.final_geometry, constraints_for(config), search.initial_geometry
    ).valid
    assert np.all(np.diff([r["physical_flux_Q"] for r in search.history]) >= 0)
    assert result.maximum_candidate_conservation_error < 1e-9
    if method == "sensitivity":
        assert result.probe_pde_solves == len(result.probes) == 6
        assert len(maps) == 2
        for round_index, geometry, points, probabilities, responses in maps:
            assert len(responses) == config.probe_count
            assert probabilities.shape == (len(points), len(points))
            assert probabilities.min() >= config.epsilon_uniform / probabilities.size
            base = result.rounds[round_index - 1]["base_Q"]
            assert {r["base_Q"] for r in result.probes if r["round"] == round_index} == {base}
            # Re-solve the actual fixed base used by the map, independently.
            assert solve(geometry, config.model).metrics.physical_flux_Q == base
    else:
        assert result.probe_pde_solves == 0
        assert not maps


def test_reproducibility_partial_final_round_and_attempt_exhaustion(monkeypatch):
    config = GuidedConfig(
        seed=9001, solve_budget=11, max_attempts_per_solve=100, refinement_factors=[2]
    )
    a, b = run_guided_search(config, "sensitivity"), run_guided_search(config, "sensitivity")
    assert a.search.proposals == b.search.proposals
    assert a.search.history == b.search.history
    assert [r["candidate_pde_solves"] for r in a.rounds] == [8, 2]
    np.testing.assert_array_equal(a.search.final_geometry.inside, b.search.final_geometry.inside)
    monkeypatch.setattr(
        "oxygen_shape.optimization.guided_search.propose_paired_disk",
        lambda *args, **kwargs: DiskEdit(None, {}, "No feasible pair"),
    )
    partial = run_guided_search(
        config.model_copy(update={"max_attempts_per_solve": 1}), "sensitivity"
    )
    assert partial.status == "incomplete_attempt_budget"
    assert partial.search.number_of_pde_solves == 1
    assert partial.search.attempts == 11
    assert partial.probe_pde_solves == 0


@pytest.mark.parametrize(
    "fields",
    [
        {"probe_count": 8},
        {"kernel_bandwidth": 0},
        {"epsilon_uniform": 0},
        {"model": {"Bi": 0}},
        {"evaluation_seeds": []},
        {"evaluation_seeds": [8000, 8000]},
        {"shapes": []},
        {"shapes": ["rectangle", "rectangle"]},
    ],
)
def test_invalid_guidance_configuration(fields):
    with pytest.raises(ValueError):
        SensitivityBenchmarkConfig(**fields)
