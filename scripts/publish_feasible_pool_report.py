"""Publish audited development results and small portable data/figure copies."""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def table(header, rows):
    return "\n".join(
        ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def publish(root):
    repository = Path(__file__).resolve().parents[1]
    docs = repository / "docs"
    summary = json.loads((root / "summary.json").read_text())
    analysis = json.loads((root / "descriptive_analysis.json").read_text())
    audit = json.loads((root / "artifact_audit.json").read_text())
    interruption = ""
    if (root / "recovery.json").exists():
        recovery = json.loads((root / "recovery.json").read_text())
        interruption = (
            f"A turn interruption stopped execution after {recovery['completed_before_recovery']} "
            "completed jobs. Their summary and final-mask files were hash-verified unchanged; "
            "only unfinished work was restarted with the same frozen algorithms. "
            f"Interrupted overhead is {summary['interrupted_pde_solves_lower_bound']}--"
            f"{summary['interrupted_pde_solves_upper_bound']} additional PDE solves: "
            "one in-flight completion cannot be determined exactly. The partial artifacts "
            "and accounting are retained in recovery.json and interrupted_attempts/. "
            "This overhead is separate from the completed matrix totals below."
        )
    assert audit["passed"] and summary["runs"] == 180
    data = pd.read_csv(root / "evaluations.csv")
    data["fine_eta"] = data.fine_final_Q / (data.Bi * data.robin_perimeter)
    methods = pd.read_csv(root / "method_summary.csv")
    strata = pd.read_csv(root / "stratum_summary.csv")
    pairs = pd.read_csv(root / "paired_comparisons.csv")
    four = pairs[pairs.method == "refill4_flux"]
    improved_four = strata[(strata.stratum == "improved") & (strata.method == "refill4_flux")].iloc[
        0
    ]
    improved_fixed = strata[(strata.stratum == "improved") & (strata.method == "fixed8_flux")].iloc[
        0
    ]
    copied = [
        "summary.json",
        "protocol.json",
        "artifact_audit.json",
        "descriptive_analysis.json",
        "evaluations.csv",
        "method_summary.csv",
        "paired_comparisons.csv",
        "stratum_summary.csv",
        "stratified_pairs.csv",
        "ranking_comparisons.csv",
    ]
    if (root / "recovery.json").exists():
        copied.append("recovery.json")
    (docs / "data").mkdir(exist_ok=True)
    (docs / "figures").mkdir(exist_ok=True)
    for file in copied:
        shutil.copy2(root / file, docs / "data" / f"feasible_pool_{file}")
    shutil.copy2(root / "bank/manifest.json", docs / "data/feasible_pool_bank_manifest.json")
    for name in ("pool_comparison", "refined_examples"):
        shutil.copy2(
            root / "figures" / f"{name}.png", docs / "figures" / f"feasible_pool_{name}.png"
        )
    method_table = table(
        [
            "Method",
            "Completed / 30",
            "Mean fine gain Q",
            "Multiple-choice pools",
            "Eligible raw fraction",
            "Median seconds",
        ],
        [
            [
                r.method,
                f"{r.completed}/30",
                f"{r.mean_fine_gain_Q:.6f}",
                f"{100 * r.mean_multi_choice_fraction:.1f}%",
                f"{100 * r.mean_raw_valid_fraction:.2f}%",
                f"{r.median_workflow_seconds:.2f}",
            ]
            for r in methods.itertuples()
        ],
    )
    physical_table = table(
        [
            "Method",
            "Mean final Q at n=400",
            "Mean ideal Q",
            "Mean fine eta",
            "Mean Robin perimeter",
        ],
        [
            [
                name,
                f"{group.fine_final_Q.mean():.6f}",
                f"{(group.Bi * group.robin_perimeter).mean():.6f}",
                f"{group.fine_eta.mean():.6f}",
                f"{group.robin_perimeter.mean():.4f}",
            ]
            for name, group in data.groupby("method", sort=False)
        ],
    )
    stratum_table = table(
        [
            "Starting group",
            "Method",
            "Runs",
            "Mean fine gain Q",
            "Multiple-choice pools",
            "Median seconds",
        ],
        [
            [
                r.stratum,
                r.method,
                r.runs,
                f"{r.mean_fine_gain_Q:.6f}",
                f"{100 * r.mean_multi_choice_fraction:.1f}%",
                f"{r.median_workflow_seconds:.2f}",
            ]
            for r in strata.itertuples()
        ],
    )
    pair_rows = []
    for method, group in pairs.groupby("method"):
        pair_rows.append(
            [
                method,
                int((group.outcome == "win").sum()),
                int((group.outcome == "tie").sum()),
                int((group.outcome == "loss").sum()),
                f"{group.fine_delta_Q.mean():.6f}",
                f"{group.seconds_ratio.median():.2f}",
            ]
        )
    pair_table = table(
        [
            "Refill vs fixed8, same ranker",
            "Wins",
            "Ties",
            "Losses",
            "Mean fine delta Q",
            "Median time ratio",
        ],
        pair_rows,
    )
    ranking_table = table(
        ["Flux vs uniform", "Wins", "Ties", "Losses", "Identical final masks"],
        [
            [name, row["wins"], row["ties"], row["losses"], row["identical_masks"]]
            for name, row in analysis["ranking"].items()
        ],
    )
    run_link = f"../experiments/{root.name}"
    body = f"""# Distinct valid proposal pools: completed development results

The 180-run comparison is complete, with {summary["completed_budgets"]} completed
PDE budgets and {analysis["budget_exhaustions"]} raw-budget exhaustions. It tests
candidate availability before further RL training. No policy is trained or
promoted. [Frozen protocol](feasible_pool_method.md);
[original run and exact source snapshot]({run_link}/summary.json).

Under the fixed PDE budget, four-choice flux ranking beats fixed8 flux on
{int((four.outcome == "win").sum())}/30 matched pairs, with
{int((four.outcome == "tie").sum())} ties and {int((four.outcome == "loss").sum())}
losses. Its median matched workflow-time ratio is {four.seconds_ratio.median():.2f}.
On previously improved starts, multiple-choice pool availability rises from
{100 * improved_fixed.mean_multi_choice_fraction:.1f}% to
{100 * improved_four.mean_multi_choice_fraction:.1f}%. This supports the pool
mechanism, but does not demonstrate an equal-time or learned-policy advantage.

The user-shared PINN topology post motivated explicit constraint attack tests
and independent physical verification. Its Re=30/Stokes and mathematical
optimality claims need evidence; collapsing or displaced material can expose a
missing admissibility condition. See the
[critique and primary references](pinn_topology_critique.md).

{interruption}

Finalization identified an all-zero derived refinement CSV and completion log
in the last pre-interruption job. Their original bytes are archived. The table
was rebuilt from independently checked saved concentration fields and boundary
fluxes; summaries, masks and numerical arrays were unchanged, with no new PDE
solve. [Repair record]({run_link}/artifact_repairs/repair.json).
The audit separately records {len(audit["discarded_interrupted_damaged_files"])}
unreadable candidate archives in the discarded interrupted attempt; they were
never used in the completed comparison. All completed-matrix fields pass.

## Design and interpretation

Six known development starts: canonical deep cavity and rectangle, plus two
previously improved masks of each family. Their former test role ended before
this study; they are now development knowledge. Five fresh seeds per start,
six methods, Bi=1, n=100 and independently solved endpoints at n=200/400.
Area 0.24 +/- 1e-12; Robin perimeter <=4; source length 0.1 and protected collar
0.12; connected air region; no holes; operational feature width 0.04 checked at
all three resolutions. The feature criterion is a raster morphological test,
not an exact continuum minimum-neck certificate.

Fixed pools contain eight raw draws. Refill seeks two/four distinct eligible
edits after at least eight draws, capped at 256 draws per pool. All methods
retain probability 0.25 on the first unconditional raw proposal; otherwise
select uniformly or by incumbent-wall-flux score. No candidate PDE informs the
ranking. Only a fully valid candidate can be solved, and only actual global
coarse Q improvement >1e-7 accepts an edit. Every raw draw and full geometry
check is charged. All runs have a 17-PDE upper budget and 6000-raw-draw cap;
endpoint refinement adds four solves. Refill is rejection sampling, not a
learned improvement of raw proposal quality. [Capacity calculation](valid_pool_capacity.md).

## Outcomes under the fixed PDE/raw budgets

These are equally weighted per-run means and medians. Workflow seconds include
candidate artifact storage and endpoint refinement; they exclude any training
because there is none. Timing is descriptive and is not an equal-wall-time test.

{method_table}

Physical transfer and membrane utilization are reported separately. Refinement
keeps the same raster cell union and Robin boundary measure, so fine eta is
Q_400 / (Bi * Robin perimeter). Mean eta is the mean of these per-run ratios.
Maximizing Q and maximizing eta remain different objectives.

{physical_table}

![Flux, choice availability, raw eligibility and runtime](figures/feasible_pool_pool_comparison.png)

Greater multiple-choice availability is a mechanism result, not proof of better
raw yield, learned decision quality, computational speed or global optimality.
Raw eligibility additionally excludes incumbent, duplicate and already-solved
masks. State trajectories differ, so its changes do not isolate a causal change
in the ordinary disk primitive's feasibility.

## Starting-shape strata

Canonical and previously improved shapes have different remaining headroom.
The descriptive split prevents large canonical gains from hiding harder cases.

{stratum_table}

## Matched comparisons

Each refill method is compared with the fixed8 control using the same starting
mask, seed and ranker. Fine Q at n=400 defines a win/loss with tolerance 1e-7.
All outcomes, including budget exhaustion, are retained. Controls are reused
across the two quota comparisons; these counts are not independent replicates.

{pair_table}

Within a quota, comparing the flux heuristic with a uniform ranker measures
whether a particular simple ranking can use the alternatives. It does not
establish that the current learned policy could do so.

{ranking_table}

![Selected endpoint fields](figures/feasible_pool_refined_examples.png)

Fields use the middle rollout seed 181002 and the first improved start in each
family, selected by a descriptive display rule rather than the best outcome.
Only the flux-ranked methods are shown; complete uniform outcomes are retained
in the data. White is outside the domain; red is the protected source.

## Numerical and reproducibility audit

- {int((data.fine_gain_Q > 0).sum())}/180 fine gains are positive; minimum
  {analysis["minimum_fine_gain_Q"]:.9g}. In
  {analysis["gains_exceed_observed_grid_sensitivity"]}/180 runs the fine gain
  exceeds the observed change in gain between n=200 and n=400. This observation
  is a refinement diagnostic, not a rigorous continuum error bound.
- {summary["number_of_pde_solves"]} new experimental PDE solves, zero training
  solves; historical preparation of reused masks belongs to the earlier study.
  Total raw draws {audit["raw_draws"]}; full candidate geometry checks
  {audit["full_geometry_checks"]}; accepted edits {audit["accepted_edits"]}.
- Maximum recorded conservation error {summary["max_conservation_error"]:.4g}.
  The separate audit recomputes every saved field's face fluxes and cell balance
  using vectorized differences without a linear solve: {audit["stored_fields_rechecked"]}
  fields, maximum recomputed conservation error
  {audit["maximum_recomputed_conservation_error"]:.4g}. It replays all
  {audit["candidate_edits_replayed"]} selected valid disk edits from their seeds
  and verifies exact candidate masks, hard constraints and global acceptance.
- Exact source files, input checksums, configuration, software versions, raw
  proposal seeds and selection probabilities are retained. The source audit
  passes. Git commit is null because this repository has no first commit.
  Recorded completed workflow durations sum to {summary["wall_clock_seconds"]:.1f}
  seconds, excluding interrupted work, downtime, setup, regression tests and
  subsequent audit/report generation.
- Regression gate: 217 tests pass. Ruff checks and formatting pass. New attack
  tests reject collapse, isolated islands, spatial evasion, source changes and
  Robin-to-Neumann relabeling. Existing edits already preserve boundary labels;
  there is no evidence earlier optimizers exploited relabeling.

## Consequences and next work

Retain the finite-volume reference and current hard constraints. This mechanism
study does not establish a reliable RL advantage or select a production default.
The next comparison should address useful choices per unit cost: feasibility-aware
location/radius proposals, followed by learned-versus-untrained/flux selection
with all generation and training costs counted. Fresh final-test geometries
must be reserved before that comparison; these six starts are development data.
An equal-wall-time comparison, transfer across Bi and richer observations remain
open. Monte Carlo and 3D gates are unchanged.

[GPU availability and measured CPU phases](gpu_readiness.md) records the verified
RTX 4060 Laptop GPU and why this small linear selector remains on the CPU.

## Portable data

[Run totals](data/feasible_pool_summary.json),
[all 180 outcomes](data/feasible_pool_evaluations.csv),
[method summaries](data/feasible_pool_method_summary.csv),
[starting-group summaries](data/feasible_pool_stratum_summary.csv),
[quota pairs](data/feasible_pool_paired_comparisons.csv),
[ranking pairs](data/feasible_pool_ranking_comparisons.csv),
[artifact audit](data/feasible_pool_artifact_audit.json),
[descriptive analysis](data/feasible_pool_descriptive_analysis.json).
"""
    destination = docs / "feasible_pool_results.md"
    destination.write_text(body, encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    publish(parser.parse_args().run_directory.resolve())
