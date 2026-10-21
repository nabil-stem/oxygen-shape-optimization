"""Analytical 2D strip flux with touching Dirichlet and Robin boundaries."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from oxygen_shape.experiments.morphology_validation import cached_fvm
from oxygen_shape.experiments.reporting import write_csv
from oxygen_shape.geometry.raster import FaceKey, RasterGeometry, Side
from oxygen_shape.physics.model import SolverSettings


def strip_flux(Bi, length=0.5, half_width=0.05, terms=8192):
    if not np.isfinite([Bi, length, half_width]).all() or Bi < 0 or min(length, half_width) <= 0:
        raise ValueError("Finite Bi>=0 and positive strip dimensions required")
    if not isinstance(terms, int) or terms < 2:
        raise ValueError("At least two spectral terms required")
    if Bi == 0:
        return {"partial_Q": 0.0, "tail_bound": 0.0, "lower": 0.0, "upper": 0.0}
    b = Bi * half_width
    contributions = []
    for n in range(terms):
        delta = brentq(
            lambda d, index=n: (index * np.pi + d) * np.sin(d) - b * np.cos(d),
            0,
            np.pi / 2,
            xtol=5e-15,
        )
        t = n * np.pi + delta
        contributions.append(
            4
            * np.sin(delta) ** 2
            * np.tanh(t * length / half_width)
            / (t + np.sin(delta) * np.cos(delta))
        )
    value = float(np.sum(contributions))
    tail = 2 * b * b / (np.pi**3 * (terms - 1) ** 2)
    guard = 64 * np.finfo(float).eps * terms * (1 + value)
    return dict(partial_Q=value, tail_bound=tail, lower=value - guard, upper=value + tail + guard)


def strip_geometry(n, length=0.5, width=0.1):
    ny, nx = round(width * n), round(length * n)
    if min(ny, nx) < 1 or not np.allclose([ny / n, nx / n], [width, length]):
        raise ValueError("Strip dimensions must match the physical grid")
    return RasterGeometry(
        np.ones((ny, nx), dtype=bool),
        1 / n,
        tuple(FaceKey(r, 0, Side.WEST) for r in range(ny)),
        tuple(FaceKey(r, nx - 1, Side.EAST) for r in range(ny)),
    )


def run(root):
    output = root / "postprocess/source_corner"
    output.mkdir(parents=True, exist_ok=True)
    rows, series = [], []
    for Bi in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        exact = strip_flux(Bi)
        series.append({"Bi": Bi, **exact})
        if Bi not in (0.1, 1.0, 10.0, 100.0):
            continue
        for n in (100, 200, 400):
            result = cached_fvm(
                output / f"Bi{Bi}_n{n}.fvm.npz", strip_geometry(n), Bi, SolverSettings()
            )
            rows.append(
                dict(
                    Bi=Bi,
                    n=n,
                    Q=result.metrics.physical_flux_Q,
                    analytical_lower=exact["lower"],
                    analytical_upper=exact["upper"],
                    relative_difference_percent=100
                    * (result.metrics.physical_flux_Q / exact["partial_Q"] - 1),
                    conservation_error=result.metrics.conservation_error,
                )
            )
        print(f"Analytical strip Bi={Bi:g}: Q={exact['partial_Q']:.9f}", flush=True)
    write_csv(output / "refinement.csv", rows)
    write_csv(output / "analytical_flux.csv", series)
    table = "\n".join(
        f"| {row['Bi']:g} | {row['partial_Q']:.9f} | {row['tail_bound']:.3g} |" for row in series
    )
    refinement_table = "\n".join(
        f"| {row['Bi']:g} | {row['n']} | {row['Q']:.9f} "
        f"| {row['relative_difference_percent']:+.3f}% |"
        for row in rows
    )
    (output / "report.md").write_text(
        "# Independent analytical 2D corner validation\n\n"
        "Strip length 0.5, width 0.1, source on the full left edge, Robin top/bottom, "
        "Neumann right edge. This validates the solver and inlet-corner interpretation; "
        "it is not a new optimization experiment.\n\n"
        "| Bi | Analytical partial Q | Proven series tail bound |\n|---:|---:|---:|\n"
        + table
        + "\n\n| Bi | Grid n | FVM Q | Difference from analytical Q |\n"
        "|---:|---:|---:|---:|\n"
        + refinement_table
        + "\n\nThe series grows with permeability and has no finite Bi-to-infinity "
        "limit when source and sink walls touch. Finite-Bi grid convergence is a "
        "separate question. Twelve additional PDE solves are charged; spectral "
        "terms are not PDE solves. The finite series tail bound is analytical; "
        "floating arithmetic is guarded but not interval-certified.\n",
        encoding="utf-8",
    )
    sources = [Path(__file__).resolve(), Path("docs/source_corner_limit.md").resolve()]
    for file in sources:
        shutil.copyfile(file, output / file.name)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "additional_FVM_solves": len(list(output.glob("*.fvm.npz"))),
                "source_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
                },
                "length": 0.5,
                "width": 0.1,
                "terms": 8192,
                "seed": None,
                "solver_settings": SolverSettings().model_dump(),
                "scope": "analytical 2D validation, not a morphology search",
            },
            indent=2,
        )
        + "\n"
    )
    print(output / "report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
