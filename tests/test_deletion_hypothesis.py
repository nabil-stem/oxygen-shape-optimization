import csv
import json

import numpy as np
import pytest

from oxygen_shape.experiments.deletion_config import DeletionConfig
from oxygen_shape.experiments.deletion_hypothesis import candidate_violations, run_deletion_study
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.constraints import validate_geometry
from oxygen_shape.geometry.disk_edits import (
    DiskEdit,
    apply_disk,
    boundary_samples,
    propose_paired_disk,
)
from oxygen_shape.geometry.raster import refine_geometry
from oxygen_shape.optimization.donor_sampling import eligible_wall_flux, low_flux_weights
from oxygen_shape.physics.model import DiffusionModel
from oxygen_shape.solvers.finite_volume import solve


def test_flux_weights_use_equal_face_measure_and_include_quantile_ties():
    flux = np.array([0.1, 0.1, 0.5, 0.9])
    np.testing.assert_array_equal(low_flux_weights(flux, 0.2), [0.5, 0.5, 0, 0])
    np.testing.assert_array_equal(low_flux_weights(np.ones(4), 0.2), np.full(4, 0.25))
    for bad in ([], [np.nan], [[1, 2]]):
        with pytest.raises(ValueError):
            low_flux_weights(bad, 0.2)
    with pytest.raises(ValueError):
        low_flux_weights(flux, 1)


def test_weighted_donor_uses_selected_face_and_keeps_exact_area():
    geometry = matched_area("deep_cavity", 100).geometry
    reference = solve(geometry, DiffusionModel(Bi=1))
    flux = eligible_wall_flux(geometry, reference, 0.12)
    points, _ = boundary_samples(geometry, 0.12)
    assert len(points) == len(flux)
    weights = low_flux_weights(flux, 0.2)
    rng = np.random.default_rng(7300)
    valid = 0
    for _ in range(80):
        proposal = propose_paired_disk(geometry, rng, 0.06, 0.12, donor_weights=weights)
        if proposal.geometry is not None:
            donor = proposal.parameters["donor_index"]
            assert weights[donor] > 0
            assert flux[donor] <= np.quantile(flux, 0.2)
            assert proposal.geometry.area == geometry.area
            valid += 1
    assert valid > 0
    for invalid in (np.zeros(len(flux)), [-1], np.full(len(flux), np.nan)):
        with pytest.raises(ValueError, match="Donor weights"):
            propose_paired_disk(geometry, rng, 0.06, 0.12, donor_weights=invalid)


def test_deletion_study_global_solves_artifacts_constraints_and_reproducibility(tmp_path):
    config = DeletionConfig(
        output_root=tmp_path,
        seed=7300,
        shapes=["deep_cavity"],
        permeabilities=[1.0],
        samples_per_group=2,
        max_attempts_per_group=500,
        refinement_factors=[2],
    )
    first, summary = run_deletion_study(config)
    second, repeat = run_deletion_study(config)
    assert summary["status"] == "complete"
    assert summary["evaluated_candidates"] == 8
    assert summary["number_of_pde_solves"] == 9 + 1 + summary["refined_candidates"]
    assert summary["max_conservation_error"] < 1e-9
    assert (first / "figures/deletion_effects.png").is_file()
    for name in (
        "source_manifest.json",
        "config.yaml",
        "summary.json",
        "proposals.csv",
        "quotas.csv",
        "refinement.csv",
        "report.md",
        "initial_geometry.npy",
    ):
        assert (first / name).is_file()
    with (first / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert (first / "metrics.csv").read_text() == (second / "metrics.csv").read_text()
    assert summary["proposal_attempts"] == repeat["proposal_attempts"]
    baseline = matched_area("deep_cavity", 100).geometry
    for row in rows:
        mask = np.load(first / "cases" / row["case"] / row["sample_id"] / "final_geometry.npy")
        geometry = baseline.with_mask(mask)
        validate_geometry(geometry, reference=baseline).require_valid()
        deletion = row["mode"] == "deletion"
        assert not candidate_violations(geometry, baseline, config, deletion=deletion)
        if deletion:
            assert (
                0
                < baseline.area - geometry.area
                <= config.max_removed_area_fraction * baseline.area
            )
        else:
            assert abs(geometry.area - config.target_area) < 1e-12
    # Independent PDE re-evaluation, rather than a reward inferred from local flux.
    check = rows[0]
    mask = np.load(first / "cases" / check["case"] / check["sample_id"] / "final_geometry.npy")
    actual = solve(baseline.with_mask(mask), DiffusionModel(Bi=1))
    assert actual.metrics.physical_flux_Q == float(check["physical_flux_Q"])


def test_attempt_exhaustion_preserves_incomplete_results(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "oxygen_shape.experiments.deletion_hypothesis.propose_paired_disk",
        lambda *args, **kwargs: DiskEdit(None, {}, "Deliberate infeasibility"),
    )
    directory, summary = run_deletion_study(
        DeletionConfig(
            output_root=tmp_path,
            shapes=["deep_cavity"],
            permeabilities=[1.0],
            samples_per_group=1,
            max_attempts_per_group=1,
            refinement_factors=[2],
        )
    )
    assert summary["status"] == "incomplete_quotas"
    assert summary["evaluated_candidates"] == 0
    assert summary["number_of_pde_solves"] == 1
    assert summary["proposal_attempts"] == 2
    assert json.loads((directory / "summary.json").read_text())["status"] == "incomplete_quotas"
    assert "0/1" in (directory / "report.md").read_text()


def test_invalid_initial_shape_rejected_before_any_pde_solve(tmp_path, monkeypatch):
    def forbidden_solve(*args, **kwargs):
        raise AssertionError("Preflight must precede every PDE solve")

    monkeypatch.setattr("oxygen_shape.experiments.deletion_hypothesis.solve", forbidden_solve)
    with pytest.raises(ValueError, match="Thin appendage"):
        run_deletion_study(
            DeletionConfig(
                output_root=tmp_path,
                shapes=["deep_cavity", "sinusoidal"],
                permeabilities=[1.0],
            )
        )
    output = next(tmp_path.iterdir())
    assert "FAILED" in (output / "run.log").read_text()


def test_numerical_failure_propagates_and_is_logged(tmp_path, monkeypatch):
    def failed_solve(*args, **kwargs):
        raise ArithmeticError("Conservation check failed")

    monkeypatch.setattr("oxygen_shape.experiments.deletion_hypothesis.solve", failed_solve)
    with pytest.raises(ArithmeticError, match="Conservation"):
        run_deletion_study(
            DeletionConfig(
                output_root=tmp_path,
                shapes=["deep_cavity"],
                permeabilities=[1.0],
            )
        )
    output = next(tmp_path.iterdir())
    assert "Conservation check failed" in (output / "run.log").read_text()


@pytest.mark.parametrize("factor", [1, 2])
@pytest.mark.parametrize(
    "center,radius,donor,sign",
    [
        ((0.6715429118, 0.2144664505), 0.0890139501, 2, -1),
        ((0.8168965197, 0.2294639895), 0.0664227707, 19, 1),
    ],
)
def test_low_flux_cuts_can_both_increase_and_decrease_global_flux(
    factor,
    center,
    radius,
    donor,
    sign,
):
    # Descriptive examples from the prespecified seed-7300 audit at Bi=0.1.
    # Preserve the physical cut under grid refinement, rather than rerasterizing.
    initial = matched_area("deep_cavity", 100).geometry
    model = DiffusionModel(Bi=0.1)
    reference = solve(initial, model)
    flux = eligible_wall_flux(initial, reference, 0.12)
    assert flux[donor] <= np.quantile(flux, 0.2)
    points, _ = boundary_samples(initial, 0.12)
    assert np.linalg.norm(points[donor] - center) < radius
    deletion = apply_disk(initial, np.array(center), radius, add=False)
    assert not candidate_violations(deletion, initial, DeletionConfig(), deletion=True)
    if factor > 1:
        initial, deletion = refine_geometry(initial, factor), refine_geometry(deletion, factor)
        reference = solve(initial, model)
    candidate = solve(deletion, model)
    delta = candidate.metrics.physical_flux_Q - reference.metrics.physical_flux_Q
    assert sign * delta > 1e-4


@pytest.mark.parametrize(
    "update",
    [
        {"shapes": []},
        {"shapes": ["rectangle", "rectangle"]},
        {"permeabilities": [0]},
        {"permeabilities": [1, 0.1]},
        {"n": 99},
        {"low_flux_fraction": 1},
        {"max_removed_area_fraction": 0},
        {"refinement_factors": []},
        {"samples_per_group": 3, "max_attempts_per_group": 2},
    ],
)
def test_invalid_deletion_configs(update):
    with pytest.raises(ValueError):
        DeletionConfig(**update)
