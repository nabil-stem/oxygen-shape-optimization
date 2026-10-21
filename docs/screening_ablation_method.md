# Cheap screening versus minimum-perimeter selection

Protocol fixed 2026-10-19 before collecting new results. The preceding study
found shorter runtime but inconsistent flux with minimum-perimeter pools.
This experiment separates that ranking choice from cheap feasibility screening.

## Frozen rules and costs

Retain eight independent ordinary paired-disk draws, the existing cheap area,
protected-collar and perimeter-budget checks, and the 25% slot-zero uniform
branch. The new screened-uniform rule chooses uniformly among all candidates
passing the cheap screens, rather than minimizing their perimeter. If none
passes it falls back to slot zero. Neither rule sees full topology/feature
results or candidate Q when choosing. Full geometric validation and global Q
acceptance remain unchanged. All eight raw draws are charged.

Compare ordinary one-draw uniform, minimum-perimeter pool and screened-uniform
pool. No existing default, predictor or RL checkpoint is changed. Pool size,
mixture probability, tolerances and rules are frozen for the entire study;
the audit is not used to tune the subsequent search.

## Shared-pool audit with globally solved responses

Use the same two pinned uniform seed-8000 outputs and initial controls, Bi=1,
n=100 and all existing constraints. Five fresh audit seeds 26000-26004 each
generate 64 pools per case. Both pool rules use the same selection seed on
each identical raw pool, coupling their uniform branch. Selection seeds are
separate from raw-pool generation; all choices precede full validation and PDE
response evaluation. Equal seed labels across cases do not identify identical
physical edits.

As before, fully validate every raw edit for auditing only. Additionally, solve
each unique valid selected mask once per pool and share the response if multiple
selectors choose it. Record the shared solve ID and actual cost explicitly;
invalid candidates are never solved. Keep actual delta-Q for each feasible
selection. Define retrospective one-step greedy gain as delta-Q if it exceeds
the existing improvement tolerance, otherwise zero, including zero for invalid
choices. This statistic describes one fixed state, not a sequential optimizer.

Report replicate-level paired gain differences, both-valid conditioning,
feasibility, selected perimeter, face coverage and all costs. Selection happens
before responses are known. Refine the largest positive and negative
screened-minus-minimum contrasts per case for which both selections are valid
and distinct, at n=200/400. These are descriptive extremes; report missing
directions explicitly. A contrast is Q_screened-Q_minimum, so its common
baseline cancels and only two PDE solves per refinement factor are needed.
Count these separately; do not use them to revise the sampler.

## Fresh-seed search

Run all three methods on both optimized inputs, five search seeds 28000-28004,
rotating method order. Each run has 17 PDE solves including its own initial
solve and a strict 6000-raw-draw cap. A truncated final pool uses the remaining
raw allowance. Retain incomplete runs. Both pool samplers spend the same
nominal pool effort per selected check. Actual trajectories diverge after
different choices and acceptances, so identical raw pools apply to the fixed
audit, not to the sequential search trajectories.

Keep greedy global-Q acceptance, duplicate handling and all physical constraints.
Refine each initial/final search geometry at 200/400. Report median/IQR Q and
eta, paired wins/losses/ties using absolute tolerance 1e-7, raw draws, selected
checks, all candidate solves and elapsed time. One frozen input per family,
one Bi and five seeds limit generality. This is an equal-PDE-budget comparison;
an equal-wall-time study would require a separate protocol.
