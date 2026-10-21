"""Validated development starts and independently generated held-out pioneers."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oxygen_shape.experiments.candidate_storage import run_stored_rollout
from oxygen_shape.experiments.frozen_geometry import load_reference
from oxygen_shape.experiments.reporting import save_summary, write_csv
from oxygen_shape.geometry.baselines import matched_area
from oxygen_shape.geometry.constraints import raster_perimeter
from oxygen_shape.geometry.raster import RasterGeometry, refine_geometry
from oxygen_shape.optimization.guided_search import validate_candidate
from oxygen_shape.optimization.pioneer_config import PioneerRLConfig
from oxygen_shape.optimization.refinement import validate_refinement
from oxygen_shape.solvers.finite_volume import solve


@dataclass(frozen=True)
class PioneerStart:
    name: str
    family: str
    split: str
    geometry: RasterGeometry


def geometry_hash(geometry: RasterGeometry) -> str:
    return hashlib.sha256(geometry.inside.tobytes()).hexdigest()


def prepare_bank(
    root: Path, config: PioneerRLConfig, progress=None
) -> tuple[list[PioneerStart], dict]:
    bank = root / "bank"
    bank.mkdir()
    starts, rows, generations = [], [], []
    solves, maximum_error = 0, 0.0
    historical_solves = 0
    occupied = set()

    def add(name, family, split, geometry, fluxes, source):
        signature = geometry_hash(geometry)
        if signature in occupied:
            raise ValueError(f"Duplicate starting geometry across bank: {name}")
        occupied.add(signature)
        path = bank / f"{name}.npy"
        np.save(path, geometry.inside)
        starts.append(PioneerStart(name, family, split, geometry))
        rows.append(
            {
                "name": name,
                "family": family,
                "split": split,
                "mask_file": path.name,
                "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "geometry_hash": signature,
                "Q": fluxes[0],
                "Q_200": fluxes[1],
                "Q_400": fluxes[-1],
                "area": geometry.area,
                "robin_perimeter": raster_perimeter(geometry),
                "eta": fluxes[0] / (config.model.Bi * raster_perimeter(geometry)),
                "source": source,
            }
        )

    def verify(geometry, family, expected_Q=None):
        nonlocal solves, maximum_error
        cfg = config.model_copy(update={"initial_shape": family})
        reason = validate_candidate(
            geometry, matched_area(family, config.n, config.target_area).geometry, cfg
        )
        if reason:
            raise ValueError(f"Pioneer constraints: {reason}")
        fluxes = []
        for factor in [1, *config.refinement_factors]:
            result = solve(
                geometry if factor == 1 else refine_geometry(geometry, factor),
                config.model,
                config.solver,
            )
            solves += 1
            maximum_error = max(maximum_error, result.metrics.conservation_error)
            fluxes.append(result.metrics.physical_flux_Q)
        if expected_Q is not None and not np.isclose(fluxes[0], expected_Q, rtol=1e-10, atol=1e-12):
            raise ValueError("Source pioneer Q mismatch")
        return fluxes

    for family in config.families:
        geometry = matched_area(family, config.n, config.target_area).geometry
        add(
            f"train_{family}_canonical",
            family,
            "train",
            geometry,
            verify(geometry, family),
            "canonical",
        )
    for index, item in enumerate(config.training_references):
        geometry, metadata = load_reference(
            config.reference_config(),
            item.family,
            item.reference,
            bank / "source_inputs" / f"reference_{index:02d}",
        )
        old = json.loads(
            (bank / "source_inputs" / f"reference_{index:02d}" / "summary.json").read_text()
        )
        historical_solves += old["number_of_pde_solves"] + old["refinement_pde_solves"]
        add(
            f"train_{item.family}_{item.reference.expected_seed}",
            item.family,
            "train",
            geometry,
            verify(geometry, item.family, metadata["recorded_Q"]),
            metadata,
        )
    # Held-out shapes are generated using fixed controls before any learner exists.
    for split, seeds in (
        ("validation", config.validation_generation_seeds),
        ("test", config.test_generation_seeds),
    ):
        for family in config.families:
            initial = matched_area(family, config.n, config.target_area).geometry
            for seed in seeds:
                choices = []
                for method in ("screened_uniform", "perimeter_pool", "flux_pool"):
                    name = f"{split}_{family}_{seed}_{method}"
                    path = root / "pioneer_generation" / name
                    cfg = config.model_copy(update={"initial_shape": family, "seed": seed})
                    result, _ = run_stored_rollout(path, cfg, initial, method=method)
                    refinements = validate_refinement(cfg, result.search, path)
                    cost = result.search.number_of_pde_solves + 2 * len(config.refinement_factors)
                    solves += cost
                    maximum_error = max(
                        maximum_error,
                        result.maximum_conservation_error,
                        max(r["conservation_error"] for r in refinements),
                    )
                    fine = refinements[-1]
                    sensitivity = abs(fine["gain_Q"] - refinements[-2]["gain_Q"])
                    eligible = (
                        result.status == "complete"
                        and fine["gain_Q"] > max(config.min_improvement, sensitivity)
                        and geometry_hash(result.search.final_geometry) not in occupied
                        and geometry_hash(result.search.final_geometry)
                        not in getattr(config, "excluded_geometry_hashes", ())
                    )
                    row = {
                        "split": split,
                        "family": family,
                        "seed": seed,
                        "method": method,
                        "status": result.status,
                        "initial_Q": result.search.initial_solution.metrics.physical_flux_Q,
                        "final_Q": result.search.final_solution.metrics.physical_flux_Q,
                        "fine_Q": fine["final_Q"],
                        "fine_gain_Q": fine["gain_Q"],
                        "grid_sensitivity": sensitivity,
                        "eligible": eligible,
                        "number_of_pde_solves": cost,
                        "raw_draws": result.search.attempts,
                    }
                    save_summary(path, row)
                    generations.append(row)
                    write_csv(root / "pioneer_generation.csv", generations)
                    if eligible:
                        choices.append(
                            (
                                fine["final_Q"],
                                method,
                                result.search.final_geometry,
                                [r["final_Q"] for r in refinements],
                                name,
                            )
                        )
                    if progress:
                        progress(f"pioneer {name}: Q={row['final_Q']:.7g}, eligible={eligible}")
                if not choices:
                    raise ValueError(
                        f"No distinct, refined improving {split} pioneer for {family}/{seed}"
                    )
                _, method, geometry, fluxes, source = max(choices, key=lambda v: (v[0], v[1]))
                add(
                    f"{split}_{family}_{seed}",
                    family,
                    split,
                    geometry,
                    fluxes,
                    {"generation_job": source, "method": method},
                )
    manifest = {
        "schema": 1,
        "n": config.n,
        "Bi": config.model.Bi,
        "starts": rows,
        "preparation_pde_solves": solves,
        "historical_pioneer_pde_solves": historical_solves,
        "max_conservation_error": maximum_error,
        "selection": "Best eligible refined Q among three fixed controls per seed; before training",
    }
    (bank / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_csv(
        root / "starting_geometries.csv",
        [{k: v for k, v in row.items() if k != "source"} for row in rows],
    )
    return starts, manifest


def load_bank(root: Path, config: PioneerRLConfig) -> tuple[list[PioneerStart], dict]:
    manifest = json.loads((root / "bank/manifest.json").read_text())
    if manifest["n"] != config.n or manifest["Bi"] != config.model.Bi:
        raise ValueError("Pioneer bank physics mismatch")
    starts, seen = [], set()
    for row in manifest["starts"]:
        if row["family"] not in config.families or row["split"] not in (
            "train",
            "validation",
            "test",
        ):
            raise ValueError("Invalid pioneer family or split")
        path = root / "bank" / row["mask_file"]
        if path.resolve().parent != (root / "bank").resolve():
            raise ValueError("Pioneer path escapes the bank")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["mask_sha256"]:
            raise ValueError("Pioneer mask checksum mismatch")
        mask = np.load(path, allow_pickle=False)
        if mask.dtype != bool or mask.shape != (config.n, config.n):
            raise ValueError("Invalid pioneer mask")
        canonical = matched_area(row["family"], config.n, config.target_area).geometry
        geometry = canonical.with_mask(mask)
        reason = validate_candidate(geometry, canonical, config)
        if reason or geometry_hash(geometry) in seen:
            raise ValueError(f"Invalid or overlapping pioneer: {reason}")
        seen.add(geometry_hash(geometry))
        starts.append(PioneerStart(row["name"], row["family"], row["split"], geometry))
    expected = {
        "train": len(config.families) + len(config.training_references),
        "validation": len(config.families) * len(config.validation_generation_seeds),
        "test": len(config.families) * len(config.test_generation_seeds),
    }
    if any(sum(s.split == key for s in starts) != count for key, count in expected.items()):
        raise ValueError("Pioneer split counts disagree with protocol")
    return starts, manifest
