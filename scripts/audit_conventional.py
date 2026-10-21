"""Independent saved-field/state audit for paid classical controls; no PDE solve."""

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_morphology_journals import close, field_diagnostics


def audit_job(directory: Path, initial):
    identity = json.loads((directory / "identity.json").read_text())
    payload = {key: value for key, value in identity.items() if key != "identity"}
    if (
        hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        != identity["identity"]
    ):
        raise ValueError("Classical identity digest mismatch")
    config = identity["config"]
    summary = json.loads((directory / "summary.json").read_text())
    files = sorted((directory / "states").glob("state_*.npz"))
    if len(files) != summary["pde_solves"]:
        raise ValueError("Classical journal and charged solve count differ")
    current = best = None
    previous_pending, previous_rows = [], []
    maximum_conservation = maximum_residual = 0.0
    protected = (np.arange(initial.inside.shape[1]) + 0.5) * initial.h < config["protected_length"]
    for step, file in enumerate(files, 1):
        with np.load(file, allow_pickle=False) as state:
            meta = json.loads(str(state["metadata"]))
            row = meta["row"]
            if meta["identity"] != identity["identity"] or row["pde_solves"] != step:
                raise ValueError("Classical journal identity/sequence mismatch")
            mask, field = state["candidate_mask"], state["candidate_field"]
            if mask.sum() != initial.inside.sum() or not np.array_equal(
                mask[:, protected], initial.inside[:, protected]
            ):
                raise ValueError("Area or source collar changed")
            q, perimeter, conservation, residual = field_diagnostics(
                initial.with_mask(mask), field, config["model"]["Bi"]
            )
            maximum_conservation = max(maximum_conservation, conservation)
            maximum_residual = max(maximum_residual, residual)
            close(q, row["candidate_Q"], "classical candidate Q")
            if perimeter > config["perimeter_limit"] + 1e-12:
                raise ValueError("Classical perimeter cap violated")
            if step == 1:
                current = best = (mask.copy(), field.copy(), row["candidate_Q"])
            else:
                if previous_pending:
                    np.testing.assert_array_equal(mask, previous_pending[0])
                    if row["proposal_seed"] != previous_rows[0]["proposal_seed"]:
                        raise ValueError("Pending batch proposal identity changed")
                    expected_pending = np.array(previous_pending[1:], dtype=bool).reshape(
                        (-1, *initial.inside.shape)
                    )
                    np.testing.assert_array_equal(state["pending_masks"], expected_pending)
                delta = row["candidate_Q"] - current[2]
                close(delta, row["delta_Q"], "classical global delta Q", atol=1e-14)
                expected_acceptance = row["candidate_Q"] > current[2] + config["min_improvement"]
                if expected_acceptance != row["accepted"]:
                    raise ValueError("Classical greedy acceptance mismatch")
                if row["accepted"]:
                    current = (mask.copy(), field.copy(), row["candidate_Q"])
                if row["candidate_Q"] > best[2]:
                    best = (mask.copy(), field.copy(), row["candidate_Q"])
            for label, expected in (("current", current), ("best", best)):
                np.testing.assert_array_equal(state[f"{label}_mask"], expected[0])
                np.testing.assert_array_equal(state[f"{label}_field"], expected[1])
                close(row[f"{label}_Q"], expected[2], f"classical {label} Q", atol=1e-14)
            previous_pending = [mask.copy() for mask in state["pending_masks"]]
            previous_rows = meta["pending_rows"]
            if len(previous_pending) != len(previous_rows):
                raise ValueError("Pending batch mask/metadata lengths disagree")
    np.testing.assert_array_equal(best[0], np.load(directory / "final_geometry.npy"))
    np.testing.assert_array_equal(best[1], np.load(directory / "concentration.npy"))
    close(best[2], summary["best_Q"], "classical reported best", atol=1e-14)
    return dict(
        candidate_fields=len(files),
        maximum_conservation_error=maximum_conservation,
        maximum_cell_residual=maximum_residual,
        additional_PDE_solves=0,
    )
