# Scientific progress and next steps

## Completed morphology study and paper review

The original 195 jobs, 90 additional classical controls and independent review
are complete. The [paper evidence](../experiments/paper_review_20261021/report.md)
contains 285 returned endpoints, verified costs (50,721 solver calls), all five
method comparisons, corrected vector figures and two unresolved leading masks.
The [manuscript](manuscript_draft.md) and [technical supplement](manuscript_supplement.md)
report negative learned-search findings and the inlet-wall/geometry limitations.

The highest observed fine-FEM endpoint is an edited sac-proxy descendant from
long flux-annealed search: Q400=0.579831643, FEM Q=0.581930801, area 0.24,
perimeter 7.82. It gains 27.55% FEM flux over its own start while adding 30.33%
membrane; another leading endpoint has overlapping practical estimates. This
cap-8 study is separate from the historical cap-4 work below.

The next publication steps are human mathematical/subject review, sharpening
novelty against close prior work, authorship/licensing and persistent code/data
deposition. No submission or public release has occurred. A stronger physiology
claim needs anatomical/transport validation and territory/packing constraints;
a stronger learning claim needs fresh held-out cases and training replication.
The current finite benchmark does not establish natural-acinar optimality.

## Completed duration study and best observed geometry

The duration run has finished: five policies reach 96 recorded episodes and all
252 development evaluations complete their PDE budgets. One training episode
hits its proposal limit; this is why the generated study status is incomplete.
At episode 96 the learner wins 21/60 shared comparisons against flux and loses
39, with mean delta Q=-0.00265121. Longer training has not established a reliable
advantage under the unchanged representation. The generated report is in
`20261020T162617162903Z_learning_duration_789b687b/report.md`.

The user asks for the best geometry. A [saved-endpoint review](best_geometry.md)
compares 1078 complete endpoints from eleven studies with matching recorded
Bi/area/perimeter/source-collar/feature constraints and n=400 refinement. The
highest Q400=0.567037493 comes from the earlier 24-episode policy rl_230002,
rollout 252001, job test_rectangle_242001_252001_rl_230002. It is an asymmetric
lobed chamber, area 0.24, Robin perimeter 3.96 and eta400=14.32%, gaining 5.67%
over its own improved starting geometry. It passes the geometry checks and an
independent saved-field conservation check (5.53e-13). This is a post-hoc best
observed endpoint, not a globally optimal geometry or policy-average claim.
Masks, field, ranking, input hashes and figure are retained in
`20261020T223106280735Z_best_geometry_review_e827c658`; no new PDE is solved.

## Earlier launch and recovery: controlled continuation to 96 episodes

The user approved the diagnosis and longer-training plan on 2026-10-20.
The [duration protocol](learning_duration_protocol.md) keeps the original
features, physics, constraints and exploration schedule and continues all five
audited policies to 96 episodes. Checkpoints at 12/24/48/96 are evaluated on the
six known development geometries with two fresh seeds and the frozen flux control.
No new final test is used or claimed. The first 24 episodes are reused exactly.

The zero-PDE diagnostic reproduces all 2573 original updates and all 1252 frozen
evaluation decisions. Learned ranking differs from the flux prior in 58.55% of
decisions; unselected candidate gains remain unknown. Regression passes 234 tests
in 308.33 seconds and Ruff checks pass. All 268 required historical files are pinned.

Active run: `20261020T162617162903Z_learning_duration_789b687b`.
After the PC crash it was verified and resumed: replicas 230000/230001 had
48 committed episodes, 230002 had 26, and 230003/230004 had 24. All 50 newly
committed episodes and 849 saved fields pass the recovery audit, including exact
weight-update replay. The all-zero status file and copies of the five intact
checkpoints are preserved in `crash_recovery_20261020T183042746209Z/`.
Only the unfinished episode 27 is replayed; its original partial files are
archived, with 3--5 additional interrupted PDE solves reported separately.
The resumed worker has already completed episodes 27/28 and continues training.
The independent background process writes `status.json`, `run.log`, atomic
`policies/` checkpoints, immutable `checkpoints/` versions, and, after training,
the development matrix and report. Launch metadata and stdout/stderr are in
`experiments/launches/20261020T183053Z_d2b27c19/` (resumed launch; the original
launch logs remain in `20261020T162615Z_a050b102/`). Until the actual final summary
exists, this study is running; do not claim that 96-episode training completed.

Maximum planned new cost is 11452 PDE solves if all budgets complete. The 2040
reused prefix solves are historical and separate. Continue or inspect this run
before changing its frozen sources or starting another duration experiment.

## Completed: RL training on meaningful admissible choices

The [refill RL results](refill_rl_results.md) complete the user's request: five
replicas, 120 episodes, 2573 parameter updates and 120 frozen evaluations.
The 38-minute pipeline costs 5014 new PDE solves, with no exhausted budgets or
interruptions. New held-out selection excludes 771 known historical masks.
Training pools offer multiple admissible choices in 96.7% of decisions.

Refined tests favor learning over raw uniform in 29/40 shared comparisons,
but over the frozen flux prior in only 17/40. Mean delta Q versus flux is
positive while median delta Q is negative; validation also favors flux in
13/20 comparisons. No reliable advantage over the strongest heuristic is
established. Retain all five models without selecting a winner or changing
the default. The full regression gate passes 231 tests. A completed-resume
check skips all 240 production training/evaluation rollouts with zero new PDE
solves and unchanged checkpoints. The artifact audit passes 271 rollouts,
4977 saved fields, 4154 exact selected edit replays and all 2573 parameter
updates, with no new PDE solves.

Next: investigate learning duration or richer observations using development
data, retaining the explicit frozen flux control and full computational costs.
These test outcomes are now development knowledge; reserve fresh masks for any
new final comparison. Equal-time performance and transfer across Bi remain open.

## Completed: source-amplitude assessment and comparable concentration views

[Source/display results](source_display_results.md) validate the user's contrast
observation. In the current linear model raising C0-Cb scales dimensional fields
and flux but leaves normalized u, Q, eta and rankings unchanged. Twenty-seven
independent scaled-RHS solves confirm this (maximum normalized field difference
8.88e-16). A shared power color scale gamma=.35 now reveals low concentrations
without changing source u=1, any physical field or optimizer input. A shared
linear reference is retained. The full regression gate passes 226 tests; all
48 old input files and 78 executed source hashes verify unchanged.

## Implemented: distinct admissible proposal pools and constraint attack tests

The user-shared PINN topology example is assessed in
[pinn_topology_critique.md](pinn_topology_critique.md). The useful immediate lesson
is explicit admissibility and independent physical verification; it does not
justify replacing the Robin diffusion solver or claiming mathematical optimality.
An explicit reference check now also protects impermeable boundary labels.

[feasible_pool_method.md](feasible_pool_method.md) freezes a development comparison
of fixed eight-draw pools with bounded refill to two/four distinct admissible
alternatives, using both uniform and flux ranking. All geometry work and raw
draws are charged; the physical constraints and PDE acceptance rule are unchanged.
The full regression gate passes 217 tests. The
[completed comparison](feasible_pool_results.md) contains 180 complete runs,
3780 matrix PDE solves and no budget exhaustion. Four-choice flux ranking beats
fixed8 flux on 24/30 matched pairs, with a median 2.40 workflow-time ratio.
On improved starts, multi-choice pools rise from 4.1% to 97.8%; the ordinary
raw primitive is unchanged. Flux beats uniform ranking on 26/30 four-choice pairs. This is
evidence of usable choices, not an equal-time or learned-policy advantage.

The audit passes all 180 jobs, 2880 exact edit replays and 3780 fields. Recovery
preserves 97 completed job summaries and final masks. Interrupted overhead is
5--6 solves; one damaged derived table was reconstructed without a new solve,
and two incomplete discarded candidate archives remain excluded. All details
and original bytes are retained. No policy or default is promoted.

The user's RTX 4060 Laptop GPU has 8 GB class VRAM. A separate three-solve
[workload profile](gpu_readiness.md) finds geometry validation costs 78.3% of
environment time; the 24-feature selector needs about 5.68 microseconds per
256-candidate call. Sparse matrix assembly dominates the profiled FVM solve.
No GPU dependency was added to this CPU implementation.

The subsequent [RL training study](refill_rl_results.md) uses these meaningful
candidate sets. Feasibility-aware primitive generation and independently
equivalent geometry/assembly acceleration remain possible cost improvements.
A GPU model should be justified by its workload rather than hardware availability.

## Completed: reward, flux-prior and feasibility revision

The user-requested revision is complete. See [rl_revision_results.md](rl_revision_results.md):
seven settings, 35 policies, 420 completed episodes, 176 validation and 152 final
test rollouts, 14509 new experimental PDE solves, 211 passing tests and passed
artifact/resume audits. No budgets were exhausted.

Validation selects the original g095_penalty setting. The combined revision
reduces training invalid selections from about 87% to 81%, but refined test
comparisons against cheap flux total 10 wins and 30 losses; against full flux,
7 wins, 25 ties and 8 losses. Controls are reused, not independent trials.
Thirty-five of forty combined final masks equal the full untrained control.
Only about 4% of fully screened pools offer multiple eligible candidates.
No policy is promoted. Reward misalignment is not established as the main
performance bottleneck under this budget.

Next, investigate proposal generation and distinct valid alternatives before
training a richer or longer controller. Learnable locations/radii/paired edits
and richer geometry/field observations are candidate directions, not proven
remedies. Charge all raw draws, geometry work and PDE solves. Freeze a new
protocol and final bank before comparison; this final bank is now development
knowledge. Equal-wall-time, transfer, MC and 3D questions remain pending or gated.

## Earlier multi-start RL study

The current user priority is RL training with strong controls and starting
geometries. The frozen protocol is in [pioneer_rl_method.md](pioneer_rl_method.md).
Production run `20261019T031120855645Z_pioneer_rl_4ad9c2ee` is complete: 12
training, 2 validation and 4 test masks, all distinct and checked at 100/200/400;
five Expected Sarsa replicas with 24 completed episodes each; and 100 completed
held-out rollouts across five controls and all five frozen models. It uses 4594
new experimental PDE solves. See [pioneer_rl_results.md](pioneer_rl_results.md).
No model selection uses those outcomes. Every replica beats uniform on 6/8
test pairs, but comparisons with current flux total 13 wins, 11 ties and 16 losses
across reused controls. No checkpoint is promoted and the scientific advantage
gate remains open.

Next, diagnose decision opportunities and candidate-choice quality using the
training/development inputs before revising rewards, features or training budget.
Freeze a revised protocol before any new final test; the present held-out bank
and seeds are now development knowledge. Preserve the flux and untrained controls
and include training cost in any equal-wall-time comparison. The pending sampler
comparison below remains a separate open study. Advanced Monte Carlo and 3D
remain gated.

## Earlier bounded tabular RL study

The completed study contains 46 successful evaluations and four failed jobs:
two feasibility-budget failures and two Windows progress-file replacement errors.
The reference policy beats uniform on 8/9 complete matched seeds, but
four new training replicas are inconsistent. The untrained action-space control
is competitive. Training and proposal overhead prevent a wall-time speedup.
The controller finished at 22:28 UTC because its estimate for the next training
job did not fit before 22:44 UTC. Only four of five training replicas ran; transfer
and the remaining planned seed batches did not run. Completed-pair comparisons
are conditional on completion and do not establish a universal RL advantage.

## Implemented: Milestone 4, deletion hypothesis

1. Sample donor faces uniformly by boundary measure or uniformly from the bottom
   20% of eligible initial wall flux. Keep the addition sampler identical.
2. Generate actual area-balanced disk pairs. For each admissible pair, also
   evaluate its removal alone. Always solve the full PDE for both candidates.
3. Pure deletion permits a stated, bounded area loss; paired edits retain exact
   area. Keep these protocols separate in reports. Both protect the source,
   topology, membrane budget, footprint and minimum-feature checks.
4. Use fixed seeds, multiple Bi values and initial shapes. Log every rejection;
   stop at a fixed attempt budget, retaining incomplete quotas explicitly.
5. Refine representative positive and negative effects on the same staircase
   geometry at 200/400 resolution. Treat grid sensitivity as an empirical check.
6. Save masks, source snapshots, configurations, fluxes, efficiency, costs,
   diagnostics and plots. Test sampling support, area balance, global re-solving,
   source protection, numerical diagnostics and reproducibility.

These comparisons are conditional on geometric feasibility. The low-flux and
uniform groups need not remove identical areas or use identical membrane lengths;
record both rather than asserting a matched-resource pure-deletion comparison.
Low flux is a proposal feature, never an acceptance criterion.

## After this implementation

The audit completed 192 candidates and 268 PDE solves. Among 48 low-flux cuts,
37 increase Q, three decrease Q, and eight are below tolerance. Selected gains
and losses survive 200/400 refinement. The default shapes are deep cavity and
rectangle; the sinusoidal baseline was rejected by the existing feature test
before inclusion. See deletion_results.md for data and the numerical counterexample.

Progress and checkpoint replacement now retry short PermissionError locks for
at most 0.8 seconds before preserving a hard failure. Completed-study evidence
and frozen source snapshots are retained unchanged.

Milestone 5's empirical sampler and comparison are now implemented; see
sensitivity_method.md and sensitivity_results.md. Thirty completed runs use
1230 search PDE solves (including 200 probes) and 120 refinement solves.
Sensitivity beats uniform on 3/5 seeds in each family, but beats flux on only
2/5 deep-cavity and 1/5 rectangle seeds. This is not a reliable learned or
sensitivity-guided advantage. The scientific advantage gate remains open.

The fixed-initial-geometry diagnostic is now implemented and completed; see
prediction_method.md and prediction_results.md. It uses 797 PDE solves with
disjoint calibration/held-out masks, nested calibration prefixes and independent
radius families. At 24 probes the current score's median rank correlations are
0.882/0.522, versus flux's 0.908/0.727. Four probes are particularly weak on the
rectangle. Neither radius conditioning nor absolute-gain scoring consistently
improves both families. No score or optimizer was promoted using these results.

The frozen-incumbent diagnostic is now complete; see
optimized_prediction_method.md and optimized_prediction_results.md. It uses
1531 PDE solves on four fixed cases, five fresh replicates each, with all quotas
met. Valid yield drops from 20.80% to 1.12% for the deep cavity and from 37.82%
to 4.30% for the rectangle. Sensitivity ranks better than flux in all five
optimized deep-cavity replicates but none of the optimized rectangle replicates.
The latter's top sensitivity-ranked quartile still loses Q on average. These
results describe one frozen incumbent per family, not a general trajectory law.

The geometry-only perimeter-pool revision is implemented and its study is
complete; see feasibility_method.md and feasibility_results.md. The audit uses
10240 raw edits, followed by 20 completed searches and 424 PDE solves including
input checks/refinement. Selected yield improves but valid yield per charged
raw draw decreases. Measured search time is lower in every matched pair;
final Q wins are only 2/5 deep-cavity and 3/5 rectangle seeds, with one
rectangle tie. Lower perimeter alone is not a flux objective. No default or
checkpoint is promoted.

The screening ablation is complete; see screening_ablation_method.md and
screening_ablation_results.md. All 30 searches finish, with 1297 PDE solves
including shared audit responses and refinement. Screened-uniform is faster
than ordinary sampling in all ten pairs, with median paired time ratios
2.04/1.88, but wins Q against minimum-perimeter in only 2/5 seeds per family.
No default or checkpoint is promoted. Distinct both-valid audit comparisons
are absent for the optimized deep cavity and sparse for the optimized rectangle.

The derivation in screening_distribution.md shows why necessary cheap screens
plus uniform screened choice preserve ordinary rejection's conditional valid
proposal law under iid assumptions. Exact rational enumeration checks the
formula, and observed raw edits satisfy the necessary-screen relation. This is
not a finite-budget flux advantage or all-geometries certificate.

Next, prespecify an equal-wall-time comparison of ordinary and screened-uniform
sampling across multiple frozen incumbents per family and fresh seeds. Keep
constraints and the global Q objective unchanged, record every raw draw/check/
solve and incomplete run, and distinguish finite-cap effects from proposal bias.
Use current data for development only. The supplied morphology papers are
recorded in pulmonary_papers.md; their structural proxies and coarse human
geometry do not replace the PDE or calibrate a new wall-permeability law.

Freeze any later policy revision before evaluating new search seeds. Reserve
fresh streams outside benchmark seeds 8000â€“8004, diagnostic streams
8200â€“8204/9200â€“9204/10200â€“10204 and 8400â€“8404/9400â€“9404/10400â€“10404,
and RL seeds 100â€“104 and 200â€“209. No predictor or RL checkpoint was promoted.
Also exclude the new audit seeds 8600-8604 and search seeds 8800-8804.
The completed screening ablation also uses audit 26000-26004 and search 28000-28004.
Keep Monte Carlo, moving-boundary derivatives and 3D behind these validations;
the scientific advantage gate remains open.
