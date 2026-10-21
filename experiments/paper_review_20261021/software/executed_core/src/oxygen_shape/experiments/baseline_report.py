"""Generated baseline results and grid-ranking tables."""

from oxygen_shape.experiments.config import BaselineConfig


def build_report(config: BaselineConfig, rows: list[dict], refinements: list[dict]) -> str:
    n = config.resolutions[-1]
    last = [r for r in refinements if r["fine_n"] == n]
    worst = max(r["relative_Q_change"] for r in last)
    flags = sum(r["relative_Q_change"] > config.refinement_warning_rtol for r in last)
    lines = [
        "# Baseline geometry and screening results",
        "",
        "This is a deterministic 2D baseline study. "
        "No optimization or global-optimality claim is made.",
        "",
        "## Declared comparisons",
        "",
        f"All cases use the unit bounding box, fixed source x=0, 0.45<=y<=0.55 (length 0.1), "
        f"u=1, and Robin conditions on every other exposed face. Grids: {config.resolutions}. "
        f"Bi: {config.permeabilities}. Seed metadata: {config.seed}; no randomness is consumed.",
        "",
        f"**fixed_area:** seven shapes with area {config.target_area:g} (tolerance 1e-12) "
        f"and Robin perimeter <= {config.perimeter_budget:g}. Actual membrane length and occupied "
        "diameter vary. This is a common-budget comparison, not an equal-membrane comparison.",
        "",
        "**fixed_perimeter:** rectangle and square with target Robin length "
        f"{config.target_perimeter:g}, within 4h on each grid. Area varies. "
        "The matching tolerance and measured errors are stored; "
        "small flux differences can reflect that mismatch.",
        "",
        "All masks include the same source corridor to x=0.55; the deep cavity extends its neck "
        "to x=0.70. Shape labels describe the attached body, including its connecting corridor.",
        "",
        "## Finest-grid metrics",
        "",
    ]
    for group in ("fixed_area", "fixed_perimeter"):
        lines += [
            f"### {group}",
            "",
            "| Shape | Area | Robin length | Diameter | "
            + " | ".join(f"Q (Bi={b:g})" for b in config.permeabilities)
            + " |",
            "|---|---:|---:|---:|" + "---:|" * len(config.permeabilities),
        ]
        for name in dict.fromkeys(r["shape"] for r in rows if r["comparison"] == group):
            series = [
                r for r in rows if r["n"] == n and r["comparison"] == group and r["shape"] == name
            ]
            first = series[0]
            lines.append(
                f"| {name} | {first['area']:.6g} | {first['robin_perimeter']:.6g} | "
                f"{first['bounding_diameter']:.6g} | "
                + " | ".join(f"{r['physical_flux_Q']:.7g}" for r in series)
                + " |"
            )
        lines += ["", f"![Flux and utilization](figures/{group}_sweep.png)", ""]
    lines += [
        "## Screening and utilization",
        "",
        "The screened fraction below is the fraction of Robin length with wall u<0.1; "
        "it is a diagnostic threshold, not an optimization rule.",
        "",
        "| Comparison | Shape | eta at lowest Bi | eta at highest Bi | "
        "wall fraction u<0.1 at highest Bi |",
        "|---|---|---:|---:|---:|",
    ]
    for group in ("fixed_area", "fixed_perimeter"):
        for name in dict.fromkeys(r["shape"] for r in rows if r["comparison"] == group):
            series = [
                r for r in rows if r["n"] == n and r["comparison"] == group and r["shape"] == name
            ]
            lines.append(
                f"| {group} | {name} | {series[0]['efficiency_eta']:.6g} | "
                f"{series[-1]['efficiency_eta']:.6g} | {series[-1]['screened_wall_fraction']:.6g} |"
            )
    lines += [
        "",
        "## Refinement and uncertainty",
        "",
        f"Largest relative Q change on the final refinement: **{100 * worst:.4g}%**. "
        f"Cases exceeding the {100 * config.refinement_warning_rtol:g}% warning threshold: "
        f"**{flags}**. This threshold is a reporting flag, not proof of convergence.",
        "",
        "| Comparison | Shape | max final Q change over Bi | max final Robin length change |",
        "|---|---|---:|---:|",
    ]
    for group in ("fixed_area", "fixed_perimeter"):
        for name in dict.fromkeys(r["shape"] for r in rows if r["comparison"] == group):
            series = [r for r in last if r["comparison"] == group and r["shape"] == name]
            lines.append(
                f"| {group} | {name} | "
                f"{100 * max(r['relative_Q_change'] for r in series):.4g}% | "
                f"{100 * max(r['relative_robin_perimeter_change'] for r in series):.4g}% |"
            )
        lines += ["", f"![Grid refinement](figures/{group}_refinement.png)", ""]
    lines += [
        "`refinement.csv` reports area drift, eta changes, and field differences after averaging "
        "2x2 fine cells. Fields are compared only where the coarse cell and all four fine cells "
        "are inside; coverage is reported. Regenerated masks can shift by a raster layer. "
        "Successive-grid changes are sensitivity estimates, not rigorous continuum error bounds.",
        "",
        "Full rank orderings on the last two grids and their stability are in `rankings.csv`. "
        "A stable ordering alone does not establish a resolved difference: compare flux gaps with "
        "grid changes and resource-matching error before interpreting a ranking.",
        "",
        "## Geometry and numerical limitations",
        "",
        "Perimeter is the exposed Cartesian face measure. Diagonal and curved raster boundaries "
        "retain orientation bias even when their perimeter stabilizes. These computations solve "
        "the staircase-domain problem; they do not establish flux accuracy for a smooth disk. "
        "No Euclidean-perimeter correction or Robin-coefficient adjustment is applied.",
        "",
        "Exact area matching selects the final equal-score layer in deterministic row-major order. "
        "This can break ideal continuous symmetry by one cell. The fixed-perimeter group selects "
        "whole layers. Generation rejects disconnected masks or modified source faces.",
        "",
        "The opening diagnostic reports the fraction of cells removed by a radius-one-cell "
        "morphological opening, including endpoint and inlet-corner effects. It is not a proven "
        "minimum-neck estimate. `minimum_neck` is null: no hard neck constraint or comparable "
        "minimum-thickness claim is made for this baseline study. Such validation is required "
        "before constrained optimization.",
        "",
        "High Bi strongly weights access near the shared source. Source/Robin corners and inlet "
        "resolution can dominate small differences; no infinite-Bi continuum plateau is asserted.",
        "",
        "Every solve checks conservation, residual, and bounds. Full case masks, concentrations, "
        "and face fluxes are retained under `cases/`; resolved configuration, seed, tolerances, "
        "environment, Git state, and hashed source snapshot are retained at the experiment root.",
        "",
        "## Field comparisons",
        "",
        "Every overview uses a shared concentration scale and a shared logarithmic Robin-density "
        "scale. The display floor is max(j)*1e-6; full unclipped numerical values remain in CSV.",
        "",
    ]
    for group in ("fixed_area", "fixed_perimeter"):
        for Bi in config.permeabilities:
            lines += [f"![{group}, Bi={Bi:g}](figures/{group}_Bi_{Bi:g}.png)", ""]
    return "\n".join(lines)


def ranking_rows(config: BaselineConfig, rows: list[dict]) -> list[dict]:
    rankings = []
    for group in ("fixed_area", "fixed_perimeter"):
        for Bi in config.permeabilities:
            previous = None
            for n in config.resolutions:
                series = sorted(
                    [r for r in rows if r["comparison"] == group and r["Bi"] == Bi and r["n"] == n],
                    key=lambda r: -r["physical_flux_Q"],
                )
                ordering = ",".join(r["shape"] for r in series)
                rankings.append(
                    {
                        "comparison": group,
                        "Bi": Bi,
                        "n": n,
                        "ordering": ordering,
                        "same_as_previous_grid": None if previous is None else ordering == previous,
                    }
                )
                previous = ordering
    return rankings
