"""Cost-aware scientific comparisons and joint proposal-map visualization."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oxygen_shape.optimization.guided_config import SensitivityBenchmarkConfig


def plot_pair_map(source: Path, output: Path) -> None:
    with np.load(source, allow_pickle=False) as data:
        points, p, mask = data["points"], data["probabilities"], data["geometry"]
        figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        for ax, values, label in zip(
            axes[:2],
            (p.sum(axis=1), p.sum(axis=0)),
            ("Addition marginal probability", "Donor marginal probability"),
            strict=True,
        ):
            ax.imshow(mask, origin="lower", extent=(0, 1, 0, 1), cmap="Greys", alpha=0.25)
            dots = ax.scatter(points[:, 0], points[:, 1], c=values, cmap="viridis", s=12)
            figure.colorbar(dots, ax=ax, label="probability per face")
            ax.set(title=label, xlabel="x", ylabel="y", xlim=(0, 1), ylim=(0, 1), aspect="equal")
        image = axes[2].imshow(p, origin="lower", cmap="magma", aspect="auto")
        axes[2].set(
            title="Joint addition/donor probability",
            xlabel="donor face index",
            ylabel="addition face index",
        )
        figure.colorbar(image, ax=axes[2], label="pair probability")
        figure.savefig(output, dpi=160)
        plt.close(figure)


def plot_comparison(
    histories: list[dict], evaluations: list[dict], config: SensitivityBenchmarkConfig, output: Path
) -> None:
    complete = {
        (r["shape"], r["method"], r["seed"]) for r in evaluations if r["status"] == "complete"
    }
    figure, axes = plt.subplots(
        len(config.shapes),
        1,
        figsize=(8, 4 * len(config.shapes)),
        layout="constrained",
        squeeze=False,
    )
    for ax, shape in zip(axes[:, 0], config.shapes, strict=True):
        for method in ("uniform", "flux", "sensitivity"):
            curves = []
            for seed in config.evaluation_seeds:
                if (shape, method, seed) not in complete:
                    continue
                rows = [
                    r
                    for r in histories
                    if (r["shape"], r["method"], r["seed"]) == (shape, method, seed)
                ]
                curves.append(
                    [100 * (r["physical_flux_Q"] / rows[0]["physical_flux_Q"] - 1) for r in rows]
                )
            if not curves:
                continue
            values = np.asarray(curves)
            x = np.arange(1, config.solve_budget + 1)
            (line,) = ax.plot(x, np.median(values, axis=0), label=f"{method} (n={len(curves)})")
            ax.fill_between(
                x,
                np.quantile(values, 0.25, axis=0),
                np.quantile(values, 0.75, axis=0),
                color=line.get_color(),
                alpha=0.2,
            )
        ax.set(
            title=f"{shape}, Bi={config.model.Bi:g}: median and interquartile range",
            xlabel="All search PDE solves, including probes and initial solve",
            ylabel="Best evaluated Q gain (%)",
        )
        ax.grid(alpha=0.2)
        if complete:
            ax.legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def write_report(
    directory: Path, config: SensitivityBenchmarkConfig, rows: list[dict], status: str
) -> None:
    lines = [
        "# Empirical sensitivity-guided search benchmark",
        "",
        f"Status: **{status}**.",
        "",
        "All methods freeze the incumbent within a round, evaluate up to "
        f"{config.batch_size} valid area-balanced paired disk edits, and accept the best global "
        f"Q gain above {config.min_improvement:g}. Each run pays for "
        f"{config.solve_budget} PDE solves including the initial state. Sensitivity's first "
        f"{config.probe_count} valid candidates per round are uniform probes; all probe "
        "costs are included and all probes can win the round. Remaining proposals use a "
        f"joint spatial map with bandwidth {config.kernel_bandwidth:g} and uniform mixture "
        f"{config.epsilon_uniform:g}. Maps are discarded between rounds. Flux guidance "
        "uses high-flux additions and low-flux donors, with the same uniform mixture. "
        "After the configured invalid-draw stall limit both guided methods temporarily "
        "use uniform pairs. The uniform control uses the same batched acceptance schedule; "
        "it differs from the earlier single-candidate greedy benchmark.",
        "",
        f"Constraints: area={config.target_area:g} within 1e-12, Robin perimeter <= "
        f"{config.perimeter_limit:g}, source length 0.1/location fixed, collar x<"
        f"{config.protected_length:g} fixed, unit bounding box, no holes, source connectivity "
        f"and operational morphology width {config.minimum_feature_width:g}. Feature checks "
        f"use n={config.n} and refinement factors {config.refinement_factors}. All final "
        "states undergo full PDE refinement on those same staircase masks.",
        "",
        f"Fresh seeds: {config.evaluation_seeds}. Method execution order rotates by seed. "
        f"Bi={config.model.Bi:g}. No policy is trained or selected. This is a signed finite "
        "paired-edit response per reallocated area, not a formal shape derivative. Radius "
        "and donor/addition interactions remain part of the measured response.",
        "",
        "## Completion, flux and cost",
        "",
        "Medians and IQR use completed runs only. Incomplete runs remain in evaluations.csv "
        "and their folders. A missing solve budget is not treated as a comparable success. "
        "Search time includes geometry checks, maps and map-file writes. Elapsed run time "
        "also includes saved state and refinement; representative plots are excluded.",
        "",
        "| Shape | Method | Complete / recorded | Median Q [IQR] | Median search seconds | "
        "Median seconds incl. refinement | Target reached | Median target solve (reached only) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for shape in config.shapes:
        for method in ("uniform", "flux", "sensitivity"):
            group = [r for r in rows if r["shape"] == shape and r["method"] == method]
            done = [r for r in group if r["status"] == "complete"]
            if not done:
                if group:
                    lines.append(
                        f"| {shape} | {method} | 0/{len(group)} | unavailable | — | — | — | — |"
                    )
                continue
            values = [r["final_Q"] for r in done]
            target = [r["first_target_solve"] for r in done if r["first_target_solve"] is not None]
            target_text = f"{np.median(target):g}" if target else "unreached"
            lines.append(
                f"| {shape} | {method} | {len(done)}/{len(group)} | "
                f"{np.median(values):.8g} "
                f"[{np.quantile(values, 0.25):.8g}, {np.quantile(values, 0.75):.8g}] | "
                f"{np.median([r['search_wall_seconds'] for r in done]):.4g} | "
                f"{np.median([r['wall_clock_seconds'] for r in done]):.4g} | "
                f"{len(target)}/{len(done)} | {target_text} |"
            )
    lines += [
        "",
        f"The target is a {100 * config.target_relative_gain:g}% gain over the shared initial Q. "
        "Target-solve medians are conditional on reaching the target, so failures to reach "
        "it must be considered alongside them.",
        "",
        "## Paired comparisons against uniform",
        "",
        "| Shape | Method | Complete pairs | Median paired Q difference | Coarse wins | "
        "Finest-grid wins | Finest positive differences exceeding grid sensitivity |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for shape in config.shapes:
        baseline = {
            r["seed"]: r
            for r in rows
            if r["shape"] == shape and r["method"] == "uniform" and r["status"] == "complete"
        }
        for method in ("flux", "sensitivity"):
            group = [
                r
                for r in rows
                if r["shape"] == shape
                and r["method"] == method
                and r["status"] == "complete"
                and r["seed"] in baseline
            ]
            if not group:
                continue
            delta, fine_wins, resolved = [], 0, 0
            for r in group:
                base = baseline[r["seed"]]
                delta.append(r["final_Q"] - base["final_Q"])
                fine_delta = r["fine_final_Q"] - base["fine_final_Q"]
                previous_delta = r["previous_grid_final_Q"] - base["previous_grid_final_Q"]
                fine_wins += fine_delta > config.min_improvement
                resolved += fine_delta > abs(fine_delta - previous_delta) + config.min_improvement
            lines.append(
                f"| {shape} | {method} | {len(group)} | {np.median(delta):.7g} | "
                f"{sum(d > config.min_improvement for d in delta)}/{len(group)} | "
                f"{fine_wins}/{len(group)} | {resolved}/{len(group)} |"
            )
    lines += [
        "",
        "![Equal-cost search histories](figures/comparison.png)",
        "",
        "## Costs and limits",
        "",
        f"Recorded search PDE solves: {sum(r['number_of_pde_solves'] for r in rows)}. "
        f"Sensitivity probe solves already included: {sum(r['probe_pde_solves'] for r in rows)}. "
        f"Additional refinement solves: {sum(r['refinement_pde_solves'] for r in rows)}.",
        "",
        "The kernel map preserves joint addition/donor observations and smooths signed "
        "delta-Q/reallocated-area before retaining positive weights. Radius variation and "
        "geometric feasibility can limit its predictive value. It is a proposal heuristic; "
        "every candidate is globally re-solved. No convergence or speedup is assumed. "
        f"{len(config.evaluation_seeds)} seeds at one Bi and "
        f"{len(config.shapes)} initial shapes do not establish generality. "
        "The grid sensitivity check is empirical, not a certified continuum error bound. "
        "Do not interpret a weak/mixed comparison as a completed scientific advantage gate.",
        "",
        "runs/ retains configs, initial/final masks, fields, all proposal attempts, probe "
        "responses, joint map arrays, round summaries and refinement states. The first seed "
        "of every method/family has history, field and sensitivity-map figures. Root masks "
        "represent the first uniform run. Source snapshots, hashes, solver tolerances, "
        "Git state and environment versions accompany the experiment.",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
