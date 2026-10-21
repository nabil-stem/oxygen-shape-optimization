"""Check source scaling, then improve shared plot contrast without changing physics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pydantic import Field

from oxygen_shape.experiments.config import ExperimentConfig
from oxygen_shape.experiments.reporting import create_run, provenance, save_summary, write_csv
from oxygen_shape.experiments.source_amplitude import solve_source_amplitude
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.generators import rectangle
from oxygen_shape.geometry.raster import refine_geometry
from oxygen_shape.physics.model import DiffusionModel
from oxygen_shape.solvers.analytical_1d import slab_flux, slab_solution
from oxygen_shape.viz.concentration import (
    ConcentrationPanel,
    plot_concentration_panels,
    plot_contrast_comparison,
)


class SourceDisplayConfig(ExperimentConfig):
    reference_root: Path
    seed: int = 191000
    gamma: float = Field(default=0.35, gt=0)
    amplitudes: tuple[float, ...] = (1, 2, 5)
    validation_Bi: tuple[float, ...] = (0.1, 1, 10)
    display_seed: int = 181002
    display_source_value: float = 1


def run(reference: Path, gamma: float):
    config = SourceDisplayConfig(
        reference_root=reference, output_root=reference.parent, gamma=gamma
    )
    root = create_run(config, "source_amplitude_display")
    panels, inputs, cases, display_rows = [], {}, {}, []
    for start, family in (
        ("deep_cavity_canonical", "deep_cavity"),
        ("rectangle_canonical", "rectangle"),
        ("improved_deep_0", "deep_cavity"),
        ("improved_rectangle_0", "rectangle"),
    ):
        for method in ("fixed8_flux", "refill2_flux", "refill4_flux"):
            path = reference / "runs" / f"{start}__{config.display_seed}__{method}"
            for file in (
                path / "final_geometry.npy",
                path / "summary.json",
                path / "concentration.npy",
                path / "refinement_4/final/concentration.npy",
            ):
                inputs[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
            summary = json.loads((path / "summary.json").read_text())
            assert summary["Bi"] == 1 and summary["grid_shape"] == [100, 100]
            coarse = matched_area(family, 100, 0.24).geometry.with_mask(
                np.load(path / "final_geometry.npy")
            )
            geometry = refine_geometry(coarse, 4)
            field = np.load(path / "refinement_4/final/concentration.npy")
            Q = summary["fine_final_Q"]
            eta = Q / summary["robin_perimeter"]  # display Bi=1; same raster boundary measure
            title = f"{start}\n{method}: Q={Q:.5f}, eta={eta:.3f}"
            panels.append(ConcentrationPanel(geometry, field, title))
            display_rows.append(
                {
                    "start": start,
                    "method": method,
                    "Q": Q,
                    "eta": eta,
                    "Bi": 1,
                    "n": 400,
                    "source_value": 1,
                    "minimum_u": float(np.nanmin(field)),
                    "median_u": float(np.nanmedian(field)),
                    "mean_u": float(np.nanmean(field)),
                    "maximum_u": float(np.nanmax(field)),
                }
            )
            if method == "fixed8_flux" and start.endswith("_canonical"):
                cases[family] = (coarse, np.load(path / "concentration.npy"))
    assert len(panels) == 12
    (root / "input_manifest.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    np.save(root / "initial_geometry.npy", panels[0].geometry.inside)
    np.save(root / "final_geometry.npy", panels[0].geometry.inside)
    write_csv(root / "display_metrics.csv", display_rows)
    cases["slab"] = (rectangle(64, 3, 1 / 64, slab=True), None)
    rows, ranking = [], []
    maximum_error = maximum_normalized_error = maximum_flux_error = 0.0
    for Bi in config.validation_Bi:
        scaled_fluxes = {a: [] for a in config.amplitudes}
        for family, (geometry, saved_field) in cases.items():
            base = None
            for amplitude in config.amplitudes:
                result = solve_source_amplitude(
                    geometry, DiffusionModel(Bi=Bi), amplitude, config.solver
                )
                if base is None:
                    assert amplitude == 1
                    base = result
                error = float(np.nanmax(np.abs(result.normalized_u - base.normalized_u)))
                flux_error = abs(result.normalized_flux - base.normalized_flux)
                assert error < 1e-11 and flux_error < 1e-11
                assert abs(result.eta - base.eta) < 1e-11
                if family == "slab":
                    exact = np.broadcast_to(
                        amplitude * slab_solution((np.arange(64) + 0.5) / 64, Bi), (3, 64)
                    )
                    np.testing.assert_allclose(result.v, exact, atol=2e-11)
                    assert abs(result.flux / (3 / 64) - amplitude * slab_flux(Bi)) < 2e-11
                else:
                    scaled_fluxes[amplitude].append(result.flux)
                    if Bi == 1:
                        np.testing.assert_allclose(
                            result.normalized_u, saved_field, atol=1e-11, equal_nan=True
                        )
                name = f"{family}_Bi{Bi:g}_a{amplitude:g}"
                np.savez_compressed(
                    root / f"{name}.npz",
                    mask=geometry.inside,
                    v=result.v,
                    normalized_u=result.normalized_u,
                )
                rows.append(
                    {
                        "case": family,
                        "Bi": Bi,
                        "amplitude": amplitude,
                        "grid_shape": str(list(geometry.inside.shape)),
                        "h": geometry.h,
                        "Q_fixed_reference": result.flux,
                        "Q_normalized": result.normalized_flux,
                        "eta": result.eta,
                        "normalized_field_error": error,
                        "normalized_flux_error": flux_error,
                        "conservation_error": result.conservation_error,
                        "normalized_residual": result.normalized_residual,
                    }
                )
                maximum_error = max(maximum_error, result.conservation_error)
                maximum_normalized_error = max(maximum_normalized_error, error)
                maximum_flux_error = max(maximum_flux_error, flux_error)
        orders = [np.argsort(scaled_fluxes[a]).tolist() for a in config.amplitudes]
        assert all(order == orders[0] for order in orders)
        ranking.append(
            {
                "Bi": Bi,
                "order": orders[0],
                "amplitudes": list(config.amplitudes),
                "unchanged": True,
                "comparison": (
                    "Two chamber endpoints, matched area/source, bounded Robin perimeter; "
                    "slab excluded"
                ),
            }
        )
    write_csv(root / "metrics.csv", rows)
    for mode in ("linear", "power"):
        plot_concentration_panels(
            panels, root / "figures" / f"refined_examples_{mode}.png", mode=mode, gamma=gamma
        )
    plot_contrast_comparison(
        [panels[0], panels[3]], root / "figures/linear_vs_power.png", gamma=gamma
    )
    for file, expected in inputs.items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == expected
    summary = {
        **provenance(),
        "status": "complete",
        "seed": config.seed,
        "Bi_display": 1,
        "grid_shape_display": [400, 400],
        "solver": "finite_volume_scaled_rhs_diagnostic",
        "solver_tolerance": None,
        "diagnostic_tolerances": config.solver.model_dump(),
        "number_of_pde_solves": len(rows),
        "max_conservation_error": maximum_error,
        "max_normalized_field_error": maximum_normalized_error,
        "max_normalized_flux_error": maximum_flux_error,
        "ranking_checks": ranking,
        "display_source_value": 1,
        "display_power_gamma": gamma,
        "reference_input_files_unchanged": True,
        "optimizer_changed": False,
        "source_physics_changed": False,
    }
    save_summary(root, summary)
    print(
        json.dumps(
            {
                "output": str(root),
                **{
                    key: summary[key]
                    for key in (
                        "number_of_pde_solves",
                        "max_normalized_field_error",
                        "max_normalized_flux_error",
                        "max_conservation_error",
                        "display_source_value",
                        "reference_input_files_unchanged",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_directory", type=Path)
    parser.add_argument("--gamma", type=float, default=0.35)
    args = parser.parse_args()
    run(args.reference_directory.resolve(), args.gamma)
