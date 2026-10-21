"""Check fixed-domain permeability inequalities using the saved catalog only."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def run(root):
    data_path = root / "catalog_metrics.csv"
    data = pd.read_csv(data_path, float_precision="round_trip")
    beta = np.logspace(-3, 3, 17)
    step = beta * 1e-4
    finite_derivative = ((beta + step) / (1 + beta + step) - (beta - step) / (1 + beta - step)) / (
        2 * step
    )
    np.testing.assert_allclose(finite_derivative, 1 / (1 + beta) ** 2, rtol=2e-8, atol=1e-12)
    rows = []
    tolerance = 1e-10
    for (family, n), group in data.groupby(["family", "n"]):
        group = group.sort_values("Bi")
        secants = np.diff(group.physical_flux_Q) / np.diff(group.Bi)
        violations = {
            "flux_decrease": max(0.0, float(-np.min(secants))),
            "concavity_violation": max(0.0, float(np.max(np.diff(secants)))),
            "eta_increase": max(0.0, float(np.max(np.diff(group.efficiency_eta)))),
            "ideal_upper_violation": max(0.0, float(np.max(group.efficiency_eta - 1))),
        }
        if max(violations.values()) > tolerance:
            raise ArithmeticError(f"Permeability inequality failed: {family}, n={n}: {violations}")
        rows.append(dict(family=family, n=int(n), points=len(group), **violations))
    output = root / "postprocess/permeability_review"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "fixed_geometry_checks.csv", index=False)
    summary = dict(
        fixed_geometry_grid_curves=len(rows),
        points=len(data),
        analytical_slab_derivative_check="passed",
        tolerance=tolerance,
        maximum_observed_violation=max(max(r[k] for k in violations) for r in rows),
        additional_PDE_solves=0,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    note = Path("docs/permeability_response.md").resolve()
    script = Path(__file__).resolve()
    for file in (note, script):
        shutil.copyfile(file, output / file.name)
    (output / "input_manifest.json").write_text(
        json.dumps(
            {
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (data_path, note, script)
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root.resolve())
