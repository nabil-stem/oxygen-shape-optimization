# Reward alignment, flux prior and full geometry screening

This note describes an earlier exploratory representation. Original run-bound
configs and commands are retained in the [v0.1.0 source snapshot](https://github.com/nabil-stem/oxygen-shape-optimization/tree/v0.1.0).
They are separate from the current cap-8 paper benchmark.

Protocol fixed before running the revised study on 2026-10-19. The original
study is retained. Its test bank and seeds are now development knowledge.
This study tests candidate causes, not an assurance that RL will beat physics.

## Objective derivation

Let Q_t be the accepted incumbent flux after decision t and Q_0>0 its initial
flux. Accepted edits require actual global delta-Q>1e-7. Invalid and valid
rejected edits leave Q_t unchanged. Define physical reward

    r_t^Q = 100*(Q_t-Q_(t-1))/Q_0.

For an episode with T decisions, telescoping gives

    sum(t=1..T) r_t^Q = 100*(Q_T-Q_0)/Q_0.

Thus gamma=1 and zero invalid penalty align episodic return with final normalized
flux gain for a fixed start. Every episode terminates within its finite raw-draw
cap, so the sum is finite. This is an objective identity, not a convergence
theorem for linear Expected Sarsa. Other arms change the objective by discounting
every selected decision or adding an invalid-edit penalty. Source Q, efficiency
eta, constraints and computational costs remain separate physical diagnostics.

The original study has 13706 invalid selections among 15626 training decisions.
Its median episode has 122 decisions and 106 invalid selections. A reward 100
decisions later is weighted by 0.95^100=0.0059205292. These measured facts motivate
the experiment; they do not establish that discounting caused the observed losses.
The reward identity is checked against analytical toy trajectories and actual
undiscounted zero-penalty rollouts before production.

See [Sutton and Barto, returns and episodes](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf)
for the return definition. Bounded episodes allow the undiscounted objective;
function approximation still has no convergence guarantee here.

## Seven fixed settings

| Setting | Gamma | Invalid reward | Fixed flux prior | Candidate screening |
|---|---:|---:|---:|---|
| g095_penalty | 0.95 | -0.01 | 0 | Cheap |
| g095_zero | 0.95 | 0 | 0 | Cheap |
| g1_penalty | 1 | -0.01 | 0 | Cheap |
| g1_zero | 1 | 0 | 0 | Cheap |
| g1_zero_prior | 1 | 0 | 1 | Cheap |
| g1_zero_full | 1 | 0 | 0 | Full |
| g1_zero_prior_full | 1 | 0 | 1 | Full |

The first four isolate gamma and invalid penalty. The last four rows, including
g1_zero, form a 2x2 comparison of flux prior and full screening at the same aligned
reward. No setting changes the physical solver, acceptance rule or constraints.

## Flux prior and update

The fixed prior has coefficient one on the existing flux_product feature and zero
elsewhere. Total action value is q(s,a)=(w_prior+w_residual)^T phi(s,a); residual
weights start at zero. Its initial frozen policy exactly matches the flux-pool
control, including tie handling and the 25% slot-zero branch. Expected Sarsa's
target and current value both use total q, while only residual weights update:

    delta = r + gamma*E_pi[q(s',a')] - q(s,a)
    w_residual += alpha*delta*phi/(1+||phi||^2).

Terminal bootstrap is zero. This is a fixed feature offset/initialization,
not imitation learning, not the exact robot-control algorithm in
[Johannink et al.](https://arxiv.org/abs/1812.03201), and not a guarantee against
degrading the initial heuristic. Both prior and residual are checkpointed and
hashed. Old zero-prior policies retain their original digest and remain loadable.

## Full geometry screening and costs

All candidates are ordinary area-balanced disk pairs. The cheap-screen proposal
law is unchanged. Full mode additionally checks every cheap-pass pool member's
source, connectivity, holes and operational feature constraints at 100/200/400
before selection, without any candidate PDE solve. The exploitation mask then
contains only full-pass, not-yet-solved candidates at that incumbent. The 25%
unconditional slot-zero branch is retained even when that candidate is invalid.
If no member is eligible, use slot zero. Do not repair or silently redraw edits.

Geometry validity is cached per incumbent and cleared after acceptance. Full
checks, cache hits, screening rejections, raw draws, selected decisions and
validation seconds are recorded. A checked selected candidate reuses its result;
there is no second hidden check. Cheap mode retains the original selected-only
full validation. This can improve selected feasibility, not validity per raw
draw. Extra screening costs can outweigh savings, so both flux and elapsed time
are reported against controls using the same full-screen mechanism.

## Bank, training and independent streams

Reuse the same 12 validated development starts (two canonical, ten pinned old
uniform improvements), and generate 2 fresh validation and 4 fresh test starts
using original cheap-screen fixed controls before learning. Exclude every mask
hash from the previous bank when selecting held-out starts. All masks must be
distinct across splits, valid under unchanged constraints, and numerically checked
at 100/200/400. No quality rule or constraint is relaxed if generation fails.

Generation seeds are 161000 and 162000/162001; validation rollout seeds are
171000/171001; final test seeds are 172000/172001. Training replica seeds
150000--150004 produce episode seeds replica*1000+episode. The same episode
streams and curriculum permutations are used across settings, but choices and
trajectories can diverge. These are paired starts/streams, not shared physical
candidate pools after divergence.

Each setting has five replicas and 12 episodes, one full curriculum cycle.
All settings get 17 PDE solves including the initial state and 6000 raw draws
per rollout, alpha=0.05, epsilon 1 to 0.15, pool size 8, uniform branch 0.25.
The original setting is retrained under this same pilot budget; comparisons
against its earlier 24-episode results are descriptive only. The full study has
420 planned training episodes, not an automatic claim of adequate convergence.

## Validation selection and fresh final test

Validation evaluates every final model from every setting and nine controls:
ordinary uniform; cheap/full screened-uniform, minimum-perimeter, current-flux
and untrained. There are 2 validation starts and 2 rollout seeds. No individual
replica is selected. Choose one setting by highest mean n=400 normalized flux
gain across all five replicas and validation pairs, among settings with all
validation runs complete. Ties within 1e-7 choose lexical setting name. If no
setting has a complete quota, no winner is declared and that fact stays explicit.

Before final test, persist the winner, validation scores, all frozen checkpoint
digests and the test method list. Test all five replicas from the selected setting
plus all five from g095_penalty and g1_zero_prior_full (deduplicating settings),
and all nine controls, on four fresh starts and two seeds. Core settings run even
if they lose validation. No retuning follows the final test. Show all validation
settings, every selected/core final replica, incomplete runs and paired refined
wins/ties/losses. Controls reused across replicas are not independent trials.

All validation/test initial and final states are re-solved at 200/400. Report
preparation, preflight, each setting's training, validation, test and refinement
PDE counts separately. Include measured full rollout wall time (setup, final
serialization and refinement), environment time, geometry-validation time and
preparation/training costs. Equal PDE/raw budgets are not an equal-wall-time
experiment or an amortized-speedup guarantee. No policy is promoted automatically.

## Reproducibility and recovery

The run saves protocol, environment versions, exact source snapshot, bank hashes,
per-episode atomic policy/history checkpoints, raw and selected traces, masks,
fields, diagnostics, durable evaluation index and validation selection. Resume
requires unchanged source, bank, configuration and numerical environment.
Completed jobs are skipped after digest checks; an interrupted training episode
replays its deterministic seed. Partial preparation retains evidence and requires
a fresh run. Tests and solver/preflight gates must pass before production.

These remain two related 2D families at Bi=1. MC, moving-wall derivatives and 3D
remain gated. Every claimed improvement concerns better observed constrained
flux, never global optimality or clinical performance.
