"""Audit saved masks, selection accounting and FV fields without solving a PDE."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from oxygen_shape.experiments.config import load_config
from oxygen_shape.experiments.feasible_pool_study import PoolStudyConfig
from oxygen_shape.experiments.pioneer_training import assert_source_unchanged
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.boundary import BoundaryKind, extract_boundary
from oxygen_shape.geometry.disk_edits import prepare_boundary, propose_paired_disk
from oxygen_shape.geometry.raster import refine_geometry
from oxygen_shape.optimization.guided_search import validate_candidate


def check_field(geometry, field, Bi):
    """Independent vectorized cell balance and series-resistance face fluxes."""
    inside = geometry.inside
    assert field.shape == inside.shape
    assert np.isnan(field[~inside]).all()
    u = np.nan_to_num(field)
    assert np.isfinite(field[inside]).all()
    assert u[inside].min() >= -1e-10 and u[inside].max() <= 1 + 1e-10
    padded, padded_u = np.pad(inside, 1), np.pad(u, 1)
    ny, nx = inside.shape
    residual = np.zeros_like(u)
    for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        neighbor = padded[1 + dy : 1 + dy + ny, 1 + dx : 1 + dx + nx]
        neighbor_u = padded_u[1 + dy : 1 + dy + ny, 1 + dx : 1 + dx + nx]
        common = inside & neighbor
        residual[common] += u[common] - neighbor_u[common]
    Q, influx = 0.0, 0.0
    conductance = Bi * geometry.h / (1 + Bi * geometry.h / 2)
    for face in extract_boundary(geometry):
        r, c = face.key.row, face.key.col
        if face.kind == BoundaryKind.SOURCE_DIRICHLET:
            flux = 2 * (u[r, c] - 1)
            influx -= flux
        elif face.kind == BoundaryKind.ROBIN:
            flux = conductance * u[r, c]
            Q += flux
        else:
            flux = 0
        residual[r, c] += flux
    error = abs(Q - influx) / max(Q, influx, 1e-15)
    assert np.max(np.abs(residual[inside])) < 1e-10
    assert error < 1e-9
    return float(Q), float(error)


def audit(root):
    config = load_config(root / "config.yaml", PoolStudyConfig)
    assert_source_unchanged(root)
    summary = json.loads((root / "summary.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    manifest = json.loads((root / "bank/manifest.json").read_text())
    rows = pd.read_csv(root / "evaluations.csv")
    assert len(rows) == protocol["expected_runs"] == summary["runs"]
    assert not rows.duplicated(["start", "seed", "method"]).any()
    expected = {
        (s["name"], seed, method)
        for s in manifest
        for seed in config.evaluation_seeds
        for method in protocol["methods"]
    }
    assert set(zip(rows.start, rows.seed, rows.method, strict=True)) == expected
    bank = {}
    for item in manifest:
        file = root / "bank" / item["mask_file"]
        assert hashlib.sha256(file.read_bytes()).hexdigest() == item["mask_sha256"]
        canonical = matched_area(item["family"], config.n, config.target_area).geometry
        geometry = canonical.with_mask(np.load(file, allow_pickle=False))
        assert not validate_candidate(geometry, canonical, config)
        bank[item["name"]] = geometry
    checked_fields, candidate_replays, maximum_error = 0, 0, 0.0
    raw_total, full_total, accepted_total, total_solves = 0, 0, 0, 0
    for job_index, row in enumerate(rows.to_dict("records"), start=1):
        name, seed, method = row["start"], row["seed"], row["method"]
        path = root / "runs" / f"{name}__{seed}__{method}"
        initial = bank[name]
        assert np.array_equal(np.load(path / "initial_geometry.npy"), initial.inside)
        Q, error = check_field(
            initial, np.load(path / "initial_concentration.npy"), config.model.Bi
        )
        checked_fields += 1
        maximum_error = max(maximum_error, error)
        assert abs(Q - row["initial_Q"]) < 1e-12
        pools = pd.read_csv(path / "pools.csv")
        proposals = pd.read_csv(path / "proposals.csv")
        assert len(proposals) == row["proposal_attempts"] <= config.raw_draw_budget
        assert proposals.raw_draw.tolist() == list(range(1, len(proposals) + 1))
        assert int(proposals.full_checked.sum()) == row["full_geometry_checks"]
        assert int(pools.selected_valid.sum()) + 1 == row["number_of_pde_solves"]
        assert row["number_of_pde_solves"] <= config.solve_budget
        assert len(list((path / "candidates").glob("*.npz"))) == int(pools.selected_valid.sum())
        assert np.isclose((pools.eligible >= 2).mean(), row["multi_choice_fraction"])
        grouped = proposals.groupby("pool", sort=True)
        assert np.allclose(grouped.selection_probability.sum(), 1, atol=1e-14)
        assert (grouped.selected.sum() == 1).all()
        assert np.array_equal(grouped.eligible.sum().to_numpy(), pools.eligible.to_numpy())
        assert (proposals.selection_probability >= 0).all()
        assert (proposals[proposals.slot == 0].selection_probability >= 0.25 - 1e-14).all()
        assert not (
            (~proposals.eligible) & (proposals.slot != 0) & (proposals.selection_probability > 0)
        ).any()
        current, current_Q, solve_index = initial, Q, 1
        prepared = prepare_boundary(current, config.protected_length)
        for pool in pools.to_dict("records"):
            choices = grouped.get_group(pool["pool"])
            selected = choices[choices.selected].iloc[0]
            assert bool(selected.eligible) == bool(pool["selected_valid"])
            if not pool["selected_valid"]:
                assert pd.isna(pool["candidate_Q"])
                continue
            # Recreate the actual geometry edit from its saved local random seed.
            local = np.random.default_rng(int(selected.proposal_seed))
            count = len(prepared.points)
            pair = divmod(int(local.integers(count**2)), count)
            radius = float(local.uniform(*config.radii))
            edit = propose_paired_disk(
                current, local, radius, config.protected_length, face_pair=pair, prepared=prepared
            )
            assert edit.geometry is not None and not validate_candidate(
                edit.geometry, initial, config
            )
            solve_index += 1
            with np.load(path / "candidates" / f"solve_{solve_index:03d}.npz") as saved:
                assert np.array_equal(saved["mask"], edit.geometry.inside)
                Q, error = check_field(edit.geometry, saved["concentration"], config.model.Bi)
                assert bool(saved["accepted"]) == bool(pool["accepted"])
            maximum_error = max(maximum_error, error)
            assert abs(Q - pool["candidate_Q"]) < 1e-12
            accepted = Q > current_Q + config.min_improvement
            assert accepted == bool(pool["accepted"])
            if accepted:
                current, current_Q = edit.geometry, Q
                prepared = prepare_boundary(current, config.protected_length)
                accepted_total += 1
            candidate_replays += 1
            checked_fields += 1
        assert np.array_equal(np.load(path / "final_geometry.npy"), current.inside)
        assert abs(current_Q - row["final_Q"]) < 1e-12
        refinement = pd.read_csv(path / "refinement.csv")
        for factor in config.refinement_factors:
            for label, geometry in (("initial", initial), ("final", current)):
                refined = refine_geometry(geometry, factor)
                field = np.load(path / f"refinement_{factor}" / label / "concentration.npy")
                Q, error = check_field(refined, field, config.model.Bi)
                expected_Q = float(
                    refinement.loc[refinement.factor == factor, f"{label}_Q"].iloc[0]
                )
                assert abs(Q - expected_Q) < 1e-12
                maximum_error = max(maximum_error, error)
                checked_fields += 1
        raw_total += len(proposals)
        full_total += int(proposals.full_checked.sum())
        total_solves += int(row["number_of_pde_solves"] + row["refinement_pde_solves"])
        if job_index % 30 == 0:
            print(f"Verified {job_index}/{len(rows)} complete jobs", flush=True)
    assert total_solves == summary["number_of_pde_solves"]
    interrupted_fields = 0
    interrupted_damaged = []
    if (root / "recovery.json").exists():
        recovery = json.loads((root / "recovery.json").read_text())
        assert recovery["status"] == "complete" and recovery["completed_files_unchanged"]
        for file, expected in recovery["preserved_completed_files"].items():
            assert hashlib.sha256((root / file).read_bytes()).hexdigest() == expected
        for attempt in recovery["interrupted_attempts"]:
            initial = bank[attempt["job"].split("__")[0]]
            for file in (root / attempt["archive"] / "candidates").glob("*.npz"):
                try:
                    with np.load(file) as saved:
                        geometry = initial.with_mask(saved["mask"])
                        assert not validate_candidate(geometry, initial, config)
                        _, error = check_field(geometry, saved["concentration"], config.model.Bi)
                        maximum_error = max(maximum_error, error)
                except (zipfile.BadZipFile, EOFError, ValueError) as exc:
                    # Only discarded interrupted attempts may be incomplete.
                    # Every field in the completed matrix remains mandatory.
                    interrupted_damaged.append(
                        {
                            "file": file.relative_to(root).as_posix(),
                            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                            "error": str(exc),
                            "used_in_results": False,
                        }
                    )
                    continue
                interrupted_fields += 1
    result = {
        "passed": True,
        "runs": len(rows),
        "raw_draws": raw_total,
        "full_geometry_checks": full_total,
        "candidate_edits_replayed": candidate_replays,
        "accepted_edits": accepted_total,
        "stored_fields_rechecked": checked_fields,
        "interrupted_fields_rechecked": interrupted_fields,
        "discarded_interrupted_damaged_files": interrupted_damaged,
        "maximum_recomputed_conservation_error": maximum_error,
        "experimental_pde_solves": total_solves,
        "audit_new_pde_solves": 0,
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (root / "artifact_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    audit(parser.parse_args().run_directory.resolve())
