"""Reconstruct recorded fluxes, search states and regression updates without solving."""

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import yaml

from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.optimization.trajectory_ranker import ResponseRanker
from oxygen_shape.optimization.trajectory_search import annealing_probability


def close(actual, expected, label, atol=1e-10):
    if not np.isclose(actual, expected, atol=atol, rtol=0):
        raise ArithmeticError(f"{label}: {actual} != {expected}")


def field_diagnostics(geometry, field, Bi):
    """Independent vectorized face balance, including all four exposed directions."""
    mask = geometry.inside
    if field.shape != mask.shape or not np.isfinite(field[mask]).all():
        raise ValueError("Nonfinite or mismatched field")
    if field[mask].min() < -1e-10 or field[mask].max() > 1 + 1e-10:
        raise ValueError("Maximum principle violated")
    padded = np.pad(mask, 1)
    values = np.where(mask, field, 0)
    padded_values = np.pad(values, 1)
    ny, nx = mask.shape
    robin = np.zeros(mask.shape, dtype=int)
    balance = np.zeros(mask.shape)
    for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        neighbor = padded[1 + dr : 1 + dr + ny, 1 + dc : 1 + dc + nx]
        other = padded_values[1 + dr : 1 + dr + ny, 1 + dc : 1 + dc + nx]
        robin += mask & ~neighbor
        balance += np.where(mask & neighbor, values - other, 0)
    source = np.zeros(mask.shape, dtype=int)
    for face in geometry.source_faces:
        if not mask[face.row, face.col]:
            raise ValueError("Source was removed")
        source[face.row, face.col] += 1
        robin[face.row, face.col] -= 1
    for face in geometry.neumann_faces:
        robin[face.row, face.col] -= 1
    g = Bi * geometry.h / (1 + Bi * geometry.h / 2)
    wall = g * robin * values
    inlet = 2 * source * (1 - values)
    balance += wall - inlet
    Q, Q_source = float(wall.sum()), float(inlet.sum())
    residual = float(np.abs(balance[mask]).max())
    conservation = abs(Q - Q_source) / max(abs(Q), abs(Q_source), 1e-14)
    if residual > 1e-9 or conservation > 1e-9:
        raise ArithmeticError(f"Field balance failed: {residual=}, {conservation=}")
    return Q, float(robin.sum() * geometry.h), conservation, residual


def audit_job(directory, initial, warm=None):
    identity = json.loads((directory / "identity.json").read_text())
    payload = {key: value for key, value in identity.items() if key != "identity"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if digest != identity["identity"]:
        raise ValueError("Trajectory identity digest mismatch")
    if hashlib.sha256(initial.inside.tobytes()).hexdigest() != identity["mask"]:
        raise ValueError("Initial mask identity mismatch")
    if (hashlib.sha256(warm.read_bytes()).hexdigest() if warm else None) != identity["warm_start"]:
        raise ValueError("Warm-start identity mismatch")
    ranker = ResponseRanker()
    if warm:
        with np.load(warm, allow_pickle=False) as model:
            ranker.weights, ranker.inverse = model["weights"].copy(), model["inverse"].copy()
            ranker.updates = int(model["updates"])
    config, method = identity["config"], identity["method"]
    summary = json.loads((directory / "summary.json").read_text())
    recorded = sorted((directory / "states").glob("state_*.npz"))
    if len(recorded) != summary["pde_solves"]:
        raise ValueError("Journal count differs from reported PDE cost")
    initial_updates = ranker.updates
    hashes, current, best = {}, None, None
    maximum_conservation = maximum_residual = 0.0
    protected = np.arange(initial.inside.shape[1]) * initial.h < config["protected_length"]
    for step, file in enumerate(recorded, 1):
        content = file.read_bytes()
        hashes[file.name] = hashlib.sha256(content).hexdigest()
        with np.load(BytesIO(content), allow_pickle=False) as state:
            meta = json.loads(str(state["metadata"]))
            row = meta["row"]
            if meta["identity"] != digest or row["pde_solves"] != step:
                raise ValueError("Journal sequence or identity mismatch")
            mask, field = state["candidate_mask"], state["candidate_field"]
            if not np.array_equal(mask[:, protected], initial.inside[:, protected]):
                raise ValueError("Protected collar was changed")
            if mask.sum() != initial.inside.sum():
                raise ValueError("Exact area was changed")
            q, perimeter, conservation, residual = field_diagnostics(
                initial.with_mask(mask), field, config["model"]["Bi"]
            )
            maximum_conservation = max(maximum_conservation, conservation)
            maximum_residual = max(maximum_residual, residual)
            close(q, row["candidate_Q"], "candidate Q")
            if perimeter > config["perimeter_limit"] + 1e-12:
                raise ValueError("Perimeter cap violated")
            if config["perimeter_band"] is not None:
                close(perimeter, summary["initial_perimeter"], "matched perimeter", atol=1e-12)
            if step == 1:
                reference_Q = row["candidate_Q"]
                close(q, meta["reference_Q"], "reference Q")
                current = best = (mask.copy(), field.copy(), row["candidate_Q"])
                if state["selected_features"].size:
                    raise ValueError("Initial solve must not invent a selected feature")
            else:
                close(reference_Q, meta["reference_Q"], "fixed reference Q", atol=1e-14)
                restart = bool(
                    config["restart_interval"] and (step - 1) % config["restart_interval"] == 0
                )
                if restart != row["restart"]:
                    raise ValueError("Restart schedule mismatch")
                if restart:
                    current = best
                delta = row["candidate_Q"] - current[2]
                close(delta, row["delta_Q"], "global delta Q", atol=1e-14)
                target = 100 * delta / meta["reference_Q"]
                close(target, row["target"], "signed learning label", atol=1e-14)
                fraction = (step - 2) / max(config["solve_budget"] - 2, 1)
                temperature = (
                    config["temperature_start"]
                    * (config["temperature_end"] / config["temperature_start"]) ** fraction
                )
                close(temperature, row["temperature"], "temperature", atol=1e-14)
                probability = (
                    float(delta > config["min_improvement"])
                    if method == "flux_greedy"
                    else annealing_probability(delta, meta["reference_Q"], temperature)
                )
                close(probability, row["acceptance_probability"], "acceptance probability")
                if (probability == 0 and row["accepted"]) or (
                    probability == 1 and not row["accepted"]
                ):
                    raise ValueError("Impossible acceptance decision")
                features = state["selected_features"]
                prediction = float(ranker.scores(features[None])[0])
                close(prediction, row["prediction"], "response prediction", atol=1e-13)
                if method == "learned_anneal":
                    ranker.update(features, row["target"])
                if row["accepted"]:
                    current = (mask.copy(), field.copy(), row["candidate_Q"])
                if row["candidate_Q"] > best[2]:
                    best = (mask.copy(), field.copy(), row["candidate_Q"])
            for label, expected in (("current", current), ("best", best)):
                if not np.array_equal(state[f"{label}_mask"], expected[0]):
                    raise ValueError(f"{label} mask does not follow the recorded decisions")
                if not np.array_equal(state[f"{label}_field"], expected[1], equal_nan=True):
                    raise ValueError(f"{label} field does not follow the recorded decisions")
                close(row[f"{label}_Q"], expected[2], f"{label} Q", atol=1e-14)
            if meta["updates"] != ranker.updates:
                raise ValueError("Response update count mismatch")
            np.testing.assert_array_equal(ranker.weights, state["weights"])
            np.testing.assert_array_equal(ranker.inverse, state["inverse"])
    np.testing.assert_array_equal(best[0], np.load(directory / "final_geometry.npy"))
    np.testing.assert_array_equal(best[1], np.load(directory / "concentration.npy"))
    with np.load(directory / "response_model.npz", allow_pickle=False) as model:
        np.testing.assert_array_equal(ranker.weights, model["weights"])
        np.testing.assert_array_equal(ranker.inverse, model["inverse"])
    close(best[2], summary["best_Q"], "reported best Q", atol=1e-14)
    return {
        "PDE_solves_reconstructed": len(recorded),
        "new_parameter_updates_replayed": ranker.updates - initial_updates,
        "maximum_conservation_error": maximum_conservation,
        "maximum_cell_residual": maximum_residual,
        "status": summary["status"],
        "input_sha256": hashes,
    }


def run(root, partial=False):
    if not partial and json.loads((root / "status.json").read_text())["phase"] != "finished":
        raise RuntimeError("Finish the study or explicitly request a partial audit")
    repository = Path(__file__).resolve().parents[1]
    for name, expected in json.loads((root / "source_manifest.json").read_text()).items():
        if hashlib.sha256((repository / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source mismatch: {name}")
    config = yaml.safe_load((root / "config.yaml").read_text())
    tasks, warm = [], None
    for index, family in enumerate(config["training_families"]):
        directory = root / "training" / f"{index:02d}_{family}"
        initial = make_research_geometry(family, seed=config["training_bank_seed"]).geometry
        tasks.append((directory, initial, warm))
        warm = directory / "response_model.npz"
    for job in json.loads((root / "jobs.json").read_text()):
        directory = root / job["path"]
        if not (directory / "summary.json").exists():
            if partial:
                continue
            raise FileNotFoundError(directory / "summary.json")
        initial = make_research_geometry(job["family"], seed=job["geometry_seed"]).geometry
        warm = root / "trained_response_model.npz" if job["method"].startswith("learned") else None
        tasks.append((directory, initial, warm))
    output = root / "postprocess/journal_audit"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (directory, initial, warm) in enumerate(tasks):
        result = audit_job(directory, initial, warm)
        label = directory.relative_to(root).as_posix()
        (output / (label.replace("/", "_") + ".json")).write_text(
            json.dumps(result, indent=2) + "\n"
        )
        rows.append({"path": label, **{k: v for k, v in result.items() if k != "input_sha256"}})
        if (index + 1) % 10 == 0:
            print(f"Audited {index + 1}/{len(tasks)} committed trajectories", flush=True)
    summary = {
        "scope": "partial" if partial else "all training and search trajectories",
        "jobs": len(rows),
        "recorded_PDE_solves_reconstructed": sum(r["PDE_solves_reconstructed"] for r in rows),
        "new_parameter_updates_replayed": sum(r["new_parameter_updates_replayed"] for r in rows),
        "maximum_conservation_error": max(r["maximum_conservation_error"] for r in rows),
        "maximum_cell_residual": max(r["maximum_cell_residual"] for r in rows),
        "additional_PDE_solves": 0,
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trajectories": rows,
    }
    (output / ("partial_summary.json" if partial else "summary.json")).write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "trajectories"}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--partial", action="store_true")
    arguments = parser.parse_args()
    run(arguments.root, arguments.partial)
