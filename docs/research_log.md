# Research log

## 2026-10-18 â€” deterministic reference

Read AGENTS.md and started Milestones 0 and 1 from the existing pyproject and
empty package scaffold. Retained the specified physical model and conductances.
Separated geometry, physics, solver, plotting, and experiment output.

The analytical slab is exactly linear: its prescribed finite-volume scheme has
roundoff error rather than a measurable asymptotic convergence order. Added a
separate non-linear 2D rectangular refinement test to complement this check.
Zero-permeability efficiency is undefined and represented as null.

Git has no initial commit. Experiments record a null commit plus dirty status and
environment versions; no commit is created automatically. See numerical_validation.md
for executed checks and results.

## 2026-10-18 â€” baseline shape and screening study

Completed Milestone 2 after rechecking all 60 reference tests. Added seven
deterministic baseline bodies with an identical source, exact raster area matching,
a separate membrane-matched rectangle/square cohort, geometry metrics, a
compare-shapes CLI, sweep plots, screening diagnostics, and grid-refinement tables.
The test suite now has 97 passing tests; Ruff lint and formatting checks pass.

Executed all 135 configured solves on 100/200/400 grids with Bi 0.01/0.1/1/10/100.
Maximum relative conservation error was 5.97e-11. The largest final flux change
was 2.102%; no case exceeded the declared 5% reporting threshold. This threshold
is not a continuum accuracy guarantee. Perimeter orientation bias, resource
matching errors, field-comparison coverage, and missing certified neck width are
reported explicitly. No numerical model or face conductance was changed.

High-Bi fluxes almost coincide because the common inlet dominates, so those tiny
shape rankings are unresolved despite their consistent ordering across grids.
The intermediate regime gives a more informative baseline for subsequent disk
search. All results and plots are in baseline_results.md, with machine-readable
tables in docs/data and full state/provenance in the saved experiment directory.

## 2026-10-18 â€” constrained disk search and requested RL extension

Completed Milestone 3: paired Boolean disk proposals, protected source collar,
exact area matching, Robin-length budget, no holes, connected air, and explicit
opening-based feature checks. A coarse-only feature check initially admitted a
thin appendage that failed refinement. Candidate filtering now repeats the
feature test on the 200 and 400 grids before a candidate PDE solve. The failing
test was retained. Seed 42 improves Q by 17.00% on the search grid and 17.31% on
the finest grid, under the same declared constraints.

Added NumPy tabular Q-learning at the user's explicit request. Eight coarse
observation states and twelve region/radius actions control geometry proposals;
the deterministic PDE and greedy global-flux acceptance remain the evaluator.
Forty episodes used 809 PDE solves and 26,821 transitions. Three observation
states and 36 state-action pairs were visited. Checkpoints and seeded learning
are reproducible; the final checkpoint is frozen for evaluation.

The pure-policy feasibility pilot failed to complete 41 solves within 4100
attempts on seed 0. Added an evaluation-only uniform fallback after fifty invalid
attempts, shared with the untrained action-space control, and used fresh held-out
seeds 100â€“104. No reward-based model selection or tuning was performed.

All fifteen comparison runs completed with 41 solves each and two refinement
levels. Median final Q: uniform 0.502121, untrained 0.507865, trained hybrid
0.523730. The trained hybrid has higher coarse and fine Q than uniform in four
of five matched-seed runs. Median elapsed run time, including refinement, is
about 18.3/24.1/30.9 seconds respectively, so the learned hybrid is slower in
wall time. Training cost is additional; no end-to-end speedup or generalization
to unseen initial geometries is established. Every final gain exceeds its
observed last-grid sensitivity, with all declared constraints passing.

111 tests pass, including delayed-reward Q-learning, reproducible updates,
checkpoint reuse, frozen weights, terminal attempt caps, feasibility fallback,
equal evaluation budgets, and geometric/PDE checks. Ruff lint and formatting pass.
See optimization_v0.md, rl_method.md, and rl_results.md. The saved model is
models/disk_q_policy.npz with its metadata. The deletion-hypothesis and
sensitivity-guided milestones remain open; no Monte Carlo solver or 3D was added.

## 2026-10-18 â€” bounded-study analysis and Milestone 4

Reviewed the finished bounded RL study: 46 successful evaluations, four completed
training replicas, and four failed evaluation jobs. Two failures exhausted the
proposal budget; two failed with Windows access denied while replacing progress
JSON. Complete-pair reference-policy wins are 8/9 versus uniform, but the
untrained control is competitive and independent training replicas are variable.
Transfer jobs and the fifth training replica did not run before admission closed.
The sealed artifacts and original checkpoint remain unchanged. See two_hour_study.md.

Implemented test-deletions with prespecified low-flux versus uniform donor sampling.
Each cut is evaluated alone under bounded area loss and as an exactly balanced
cut/addition pair. Both variants must satisfy the stated source, topology,
perimeter and feature constraints before either enters the sample. Every valid
candidate uses the global PDE; invalid proposals and quota exhaustion are logged.
The first attempted sinusoidal baseline failed the existing feature test. The
failed run remains available, the default matrix uses valid rectangle/deep-cavity
shapes, and all initial shapes are now validated before any PDE solves.

The completed matrix uses Bi=0.1/1/10, eight pairs per group, and seed schedule
7300â€“7311. It produces 192 candidate evaluations and 268 total PDE solves, including
selected-example validation at n=200/400. Low-flux deletion alone gives 37 gains,
three losses and eight sub-tolerance changes. Selected positive and negative
effects survive refinement. At Bi=0.1 a cut decreases Q by about 2.40e-4 while
increasing eta, demonstrating both the failure of a local deletion rule and the
distinction between transfer and utilization. Geometry resource changes and
joint-feasibility conditioning are explicitly reported. See deletion_results.md.

Two complete sweeps have identical metrics.csv hashes. Maximum conservation error
is below 4.38e-12. Regression tests preserve both signs on n=100/200 and check
weighted sampling, physical re-solving, source protection, quotas and provenance.
Added bounded retries for transient Windows replacement locks on progress JSON
and policy checkpoints; permanent failures still raise and retain old JSON.
Controller exit clears stale progress/sleep flags. Simulated transient/permanent
lock tests pass. Full suite: 135 passed; Ruff lint and format checks pass.

Milestone 5 is next: empirical sensitivity-guided search with fresh seeds, equal
total PDE budgets including probing, and explicit uniform/flux controls. Do not
claim an RL speedup or promote a checkpoint based on these held-out results.

## 2026-10-18 â€” empirical sensitivity sampler and equal-budget comparison

Implemented a joint addition/donor response map from signed global delta-Q per
reallocated area. The incumbent is fixed within each eight-candidate round.
Sensitivity spends four uniform candidate solves on probing and four on guided
proposals; every probe is eligible for greedy selection and every solve counts.
The map is discarded before the next round. Uniform and flux controls use the
same batch schedule, constraints and 41-solve budget. The method is an empirical
finite-pair heuristic, not a formal shape derivative. Cached boundary samples
have an owner/collar guard and preserve the original seeded proposals.

The first timing pilot overlapped regression and was stopped/excluded. The
unchanged primary protocol ran afterward: seeds 8000â€“8004, Bi=1, rectangle and
deep cavity, 30 successful runs. Search used 1230 solves including 200 probes;
refinement used 120 more. Maximum conservation error is below 4.37e-13; all
30 gains survive the empirical last-grid sensitivity check and all final
constraints pass. No checkpoint or learned model was selected or changed.

Deep-cavity median Q: uniform 0.501055, flux 0.511863, sensitivity 0.505707.
Rectangle medians: 0.508157, 0.510665 and 0.510392 respectively. Sensitivity
wins 3/5 against uniform in each family, but only 2/5 and 1/5 against flux.
Deep-cavity median target-solving counts for a 10% gain are 26/14/23, with all
probes included. No rectangle run reaches that target. Sensitivity is slower
on the deep cavity; the small rectangle timing difference is preliminary.
The implementation milestone is delivered, while a convincing sensitivity
advantage remains unestablished. Do not advance through that scientific gate.

Full suite: 152 tests pass. Signed-kernel formula, joint credit, exploration,
fixed-base probes, exact solve accounting, attempt exhaustion, partial rounds,
constraints, maps and deterministic replay are tested. Ruff checks pass.
See sensitivity_method.md and sensitivity_results.md. Next: fixed-geometry
predictive diagnostics, radius/area-response ablations and fresh-seed tests.

## 2026-10-18 â€” held-out prediction and radius diagnostic

Implemented diagnose-prediction. Geometry stays fixed; each of five replicates
per shape uses 24 unique valid calibration edits and 48 disjoint held-out edits.
Prespecified ablations compare spatial/radius kernels with absolute/per-area
global responses. Ranked scores use the sampler's clipping, top-quartile ties
receive fractional weights, and sign accuracy is compared with always-positive
prediction. No optimizer or checkpoint is changed. Independent uniform face-pair
families replay the same angular/jitter stream across three radii, including all
infeasible variants in their denominator. Numerical failures retain attempt logs.

The completed study uses 797 solves: 2 baselines, 240 calibration, 480 held-out,
63 radius variants and 12 refinement solves. All 20 quotas are met. Maximum
conservation error is below 2.845e-13; the selected positive and negative effects
retain their signs at 200/400. Calibration/test/radius seed streams are
8200â€“8204, 9200â€“9204 and 10200â€“10204, evaluated separately within each shape.

With four probes, the current map's median rank correlations are 0.692 on the
deep cavity and -0.092 on the rectangle. With 24 they are 0.882/0.522, while the
flux ranking control gives 0.908/0.727 without additional probe solves. Radius
conditioning and absolute-response ablations do not consistently improve both
families. The current score's 91.7% sign accuracy must be compared with positive
baseline rates of 87.5%/85.4%. More accurate classification alone is not evidence
of solve-budget efficiency. These metrics describe the n=100 discrete problem.

Independent matched-radius feasibility drops from 12/40 to 4/40 across the radius
range on the deep cavity and from 19/40 to 9/40 on the rectangle. Only 7/80
families are valid at every radius; 31 mix valid and invalid variants. This
supports investigating feasibility while keeping response prediction separate.

See prediction_method.md and prediction_results.md. Next: fixed optimized
incumbents near the perimeter budget, fresh seeds, and the same strong flux
control before any proposal-policy revision or advanced Monte Carlo work.

Final verification: 168 tests passed in the complete suite; Ruff lint and
format checks pass. New tests include score equivalence with the existing
probability map, tie-aware held-out metrics, disjoint-mask pools, fixed-base
replay, exact PDE counts, incomplete quotas and preserved numerical-failure logs.

## 2026-10-19 â€” frozen optimized-incumbent prediction

Implemented verified frozen inputs for diagnose-prediction, with fresh initial
controls, pinned mask checksums, source metadata/constraint checks, portable
input copies and recomputed baseline Q before candidate sampling. The inputs
are the first uniform seed-8000 outputs, not best-flux selections. Their Robin
perimeters are 4.00 and 3.96 against an unchanged budget of 4.00.

All 20 replicates met their 24-calibration/48-held-out quotas within the fixed
6000-attempt pool cap. Streams are 8400â€“8404, 9400â€“9404 and 10400â€“10404.
The run used 1531 PDE solves: 4 baseline, 480 calibration, 960 held-out,
63 radius and 24 refinement. Maximum conservation error is 6.243e-13;
selected positive/negative effects retain signs at grids 200/400. Source hashes
remain unchanged and all calibration/test mask sets are disjoint.

Deep-cavity valid pool yield falls from 20.80% to 1.12%; rectangle yield from
37.82% to 4.30%. Median sensitivity/flux rank correlations are 0.553/0.348
on the optimized deep cavity and 0.145/0.610 on the optimized rectangle.
The rectangle's sensitivity top-quartile mean delta-Q is -0.001637, versus
flux's +0.003015. Its sensitivity sign accuracy 62.5% must be compared with
the 85.4% always-negative baseline, not just 14.6% always-positive.

No score or checkpoint is promoted. One frozen incumbent per family and
conditioning on valid candidate pools limit generality. Next: specify and
evaluate a geometry-only feasibility revision under unchanged constraints,
then freeze it before fresh-seed equal-budget search. The scientific advantage
gate remains open. Full suite: 181 tests pass; Ruff lint/format checks pass.
See optimized_prediction_method.md and optimized_prediction_results.md.

## 2026-10-19 â€” supplied morphology references and feasibility protocol

Read Ishikawa/Koshiyama (2022) and Fujita/Koshiyama (2025), including rendered
method, equation, table and figure pages. Recorded source hashes and page
references in pulmonary_papers.md. Their objectives are structural proxies;
the human model's compartments are ducts/sacs without explicit alveolar walls.
Their logistic weighting is neither permeability nor fractional wall coverage.
No physical model, dimensional parameter or source boundary is changed.

Specified the perimeter-pool proposal study before collecting new results:
pool size eight, 25% uniform branch, fixed audit streams 8600-8604, search
streams 8800-8804, 17 search solves and a strict 6000 raw-draw cap. Full geometry
validation and global Q acceptance remain mandatory. Audit outcomes do not
tune this revision; every unselected draw is charged. See feasibility_method.md.

The completed run contains all 10240 audit draws and all 20 searches. It uses
424 solves (4 input, 340 search, 80 refinement), max conservation 5.1643e-13.
Optimized selected yield increases from 3/320 to 15/320 on deep cavity and
16/320 to 51/320 on rectangle, but yield per raw-draw cost declines. Search is
faster in all pairs; median paired time ratios are 2.59 and 1.44. Equal-budget
flux wins are 2/5 and 3/5 with one rectangle tie, and persist at n=400.
Median deep-cavity Q is lower for the pool; no default or checkpoint is promoted.
One small uniform rectangle gain is below its observed refinement change.

Full regression: 188 tests pass. Seven focused tests pass again after adding
SD/IQR figure summaries and explicit Q ties; Ruff passes. Original run artifacts
remain unchanged; a reviewed presentation records its renderer and input hashes.
See feasibility_results.md. Next: isolate cheap screening from minimum-perimeter
bias on identical raw pools, then use a frozen revision and fresh search seeds.

Final visual source check: the 2025 Table 2 heading is micrometers, not the mm
produced by text extraction. Corrected the literature notes to identify an
extraction error rather than a published unit error. The separately verified
seed-count arithmetic discrepancy remains flagged; no calibration was imported.

## 2026-10-19 â€” screening versus perimeter bias: prespecified ablation

Added screened_uniform pool selection: uniform among candidates passing the
existing cheap screens, with the same 25% exploration branch and eight raw draws.
Default methods and physics remain unchanged. The shared-pool audit couples
selection seeds, chooses before validation/responses, solves each unique valid
selection once per pool, and records actual delta-Q and retrospective greedy gain.
Extreme distinct both-valid contrasts are refined descriptively in both directions;
missing directions stay explicit and baseline cancellation avoids extra solves.

Protocol screening_ablation_method.md freezes audit streams 26000-26004 and
search streams 28000-28004, absent from the old bounded-study plan. Four fixed
audit cases, five replicates, 64 pools each, and three methods on two optimized
inputs use unchanged constraints, 17 search solves and a 6000 raw-draw cap.
No tuning follows the audit. Focused tests pass before the full regression gate.

The study completed all 30 searches and 1297 solves: 4 input, 639 shared audit,
24 contrast refinement, 510 search and 120 search refinement. The audit has
906 valid selector uses but charges shared selections only once per pool.
No distinct both-valid pair exists for the optimized deep cavity; only three
exist for the optimized rectangle. Six available positive/negative extreme
contrasts retain signs at 200/400; absent directions stay explicit.

Screened-uniform wins Q against minimum-perimeter on 2/5 seeds per family;
against ordinary sampling, wins are 3/5 deep and 2/5 rectangle with one tie.
Every pairwise classification persists at 100/200/400. It is faster than
ordinary sampling in all ten pairs, median paired time ratios 2.04 and 1.88,
but does not consistently beat minimum-perimeter on runtime or flux.
Maximum conservation error is 5.0939e-13. No default or checkpoint is promoted.

During the run, derived screening_distribution.md without changing the frozen
algorithm. For iid draws and necessary cheap screens, screened-uniform
preserves the conditional fully valid proposal law; minimum-perimeter generally
does not. Exact rational enumeration validates a finite example. All 10240 raw
audit edits satisfy the observed necessary-screen relation. Finite caps and
conditioning on completion remain limitations, as do one incumbent per family
and five seeds. Source hashes, masks, selection traces and solve counts pass
artifact checks. Full regression: 192 tests; Ruff and the exact check pass.
See screening_ablation_results.md. Next: a prespecified equal-wall-time study
with more frozen incumbents and fresh seeds, rather than claiming a PDE-budget
or RL advantage from this sampler.

## 2026-10-19 â€” stronger RL training foundations

User requested all preparation steps for RL training with strong baselines and
starting geometries. Implemented a separate 24-feature linear Expected Sarsa
candidate selector, keeping the existing tabular model intact. The new environment
charges current and bootstrap pools, enforces all existing constraints, and accepts
only by global Q. It includes budget fractions, geometric changes and current
wall-field features without candidate-response leakage.

Protocol pioneer_rl_method.md fixes 12 development starts (two canonical plus
ten pinned improvements), 2 fresh validation and 4 fresh test starts, five learner
replicas of 24 episodes, and five controls. New pioneers are generated by three
controls and selected by refined Q before any learner exists. Every mask hash is
unique across splits. Checkpoints atomically bundle weights and completed-episode
records and pin protocol, bank, sources and numerical dependency versions.
No held-out selection of final replicas or hyperparameters occurs.

Nine focused tests pass, including exact delayed-reward targets, terminal and
bootstrap budget accounting, seeded learning replay, frozen policy evaluation,
identity checks, and an end-to-end completed resume with zero additional solves.
Full regression and production readiness gates follow before training.

The production study completed all 120 training episodes and all 100 held-out
rollouts. Five final checkpoints contain 15626 total Expected Sarsa updates.
The 12/2/4 bank split is hash-disjoint and verified at 100/200/400. Fresh
experimental cost is 4594 PDE solves: 414 preparation, 40 preflight, 2040
training, 1700 evaluation and 400 refinement; reused pioneer generation has
another 450 recorded historical solves. No episode or rollout exhausted its
budget. Maximum conservation error is 5.8252e-13.

Every replica beats uniform on 6/8 refined test pairs. Across the five replicas,
screened-uniform/untrained comparisons total 26 wins and 14 losses, while
current-flux comparisons total 13 wins, 11 ties and 16 losses. Controls are
reused, so totals are not independent trials. One coarse/fine classification
changes against flux. No model or default is promoted. The strongest heuristic
remains competitive; no general or complete-workflow advantage is established.

The post-run audit caught an aggregate metadata bug: generic rollout method
`learned` overwrote replica names. Recovered 50 labels from immutable job names
and matching frozen model digests, preserving original exports and verifying
unchanged physical-record and checkpoint hashes. No numerical run was repeated.
Fixed row construction for future runs and expanded the end-to-end regression
to require distinct replica rows, paired reporting and frozen-digest rejection
on resume. Full regression after correction: 201 tests pass; Ruff passes.
The executed source snapshot/run identity are preserved. A documented source
patch fixes reporting; the live source guard intentionally rejects resuming this
completed old-source run. No source identity or guard was bypassed.

The artifact audit verifies 3520 saved candidate masks/fields, 276528 raw draws,
40376 decisions, rewards, source collars, fixed cell counts, bounds and frozen
checkpoints. Figures and full data are in pioneer_rl_results.md. Next: diagnose
choice quality and sparse feasibility on development inputs; any revised learner
requires a frozen protocol and fresh final test bank. The current test bank is
now development knowledge. Scientific advantage remains unresolved; MC and 3D
remain gated.

## 2026-10-19 â€” reward, flux prior and feasibility revision

User authorized the complete next experiment: reward alignment, learning from
the flux baseline, feasible selection and fresh evaluation. Fixed seven settings:
the 2x2 gamma/invalid-penalty ablation, and a 2x2 prior/full-screening comparison
at gamma=1 with zero invalid penalty. Derived the telescoping physical reward
identity and verified it on a toy trajectory and a real bounded rollout.

Added a fixed flux-product prior with learned additive residual, versioned
checkpoints retaining old zero-prior compatibility, and full geometry screening
with per-incumbent validity caching. All raw draws, full checks, cache hits and
validation times are exposed; the original exploration branch and constraints
remain fixed. Zero residual exactly matches the flux control before training.

The revision protocol reserves five replicas 150000--150004, twelve episodes
per setting, validation generation 161000, test generation 162000/162001,
validation rollout 171000/171001 and final test 172000/172001. All previous bank
mask hashes are excluded from new held-out selection. No best individual replica
is selected. All validation settings are visible; final tests cover the validation
winner and fixed original/combined settings. See rl_revision_method.md.

The production revision run 20261019T111653094459Z_rl_revision_694225e7 completed
all 420 training episodes and 328 held-out rollouts (176 validation, 152 final
test). All seven settings have five final policies. Fresh experimental cost is
14509 PDE solves: 414 preparation, 67 preflight, 7140 training, 2992 validation,
2584 test and 1312 refinement. Reused pioneers retain another 450 historical
generating solves. No quota failed. Maximum conservation error is 7.2773e-13.

Validation selects original g095_penalty at 1.6280% mean normalized fine gain;
the combined revision has 0.8645%. Gamma=1 and penalty removal each reduce mean
validation gain under these conditions. This pilot does not support reward
misalignment as the main performance bottleneck. All prespecified original and
combined replicas enter fresh test; no best individual replica is selected.

Original learned comparisons versus cheap/full flux total 19/0/21 and 16/0/24
wins/ties/losses. Combined comparisons total 10/0/30 and 7/25/8. Controls are
reused. Combined mean test gain exceeds the original learned mean, but both
are below cheap flux. All refined test gains are positive; eleven coarse/fine
model/control classifications change. No policy is promoted.

Combined final masks equal the full untrained control in 35/40 outcomes.
Only 4.1--4.2% of its test pools offer multiple eligible candidates. These
observations motivate studying proposal generation, not a claim that a new
generator or a larger model is already a proven remedy.

Full regression: 211 tests pass; Ruff passes. Source/Neumann labels are included
in validity cache keys, with dedicated tests before production. The audit
verifies 11968 fields, 712420 raw draws, 93655 decisions, 97315 checks, rewards,
quotas, hashes and constraints. All 74 executed sources and eight previous model
files remain intact. Completed production resume blocks every new rollout and
reloads all 35 models and 328 evaluations without a new PDE solve or changed
checkpoint. See rl_revision_results.md.

## 2026-10-19 â€” PINN-post assessment and distinct valid pool implementation

Assessed the supplied LinkedIn text against primary Stokes topology and PINN
references; no author implementation or animation was supplied for verification.
Re=30 does not itself justify Stokes flow, a converged bulbous design does not
prove global optimality, and collapse/fragmentation can expose an underspecified
admissible set. The direct project lesson is hard constraints plus independent
physical verification. See pinn_topology_critique.md.

Implemented bounded full-geometry rejection sampling to provide two/four distinct
candidate choices before ranking, alongside fixed-eight controls and uniform/flux
rankers. All draws and checks remain charged, with unchanged PDE and constraints.
Explicit reference Neumann-label protection supplements existing source protection
and label-preserving disk primitives. Added actual geometry attack cases and tests
for pool stopping, duplicates, exploration, budgets, reproducibility and storage.
All 217 tests and Ruff pass before production. The frozen development protocol
uses six known starts, five fresh seeds and six methods (180 runs), at most 3780
new PDE solves. No learning or model promotion is part of this mechanism study.
Production root: 20261019T122613613156Z_feasible_pool_b8b49d2b.

## 2026-10-19 â€” Valid pools complete; GPU hardware verified and workload profiled

All 180 runs complete their 17-solve budgets plus four endpoint solves:
3780 matrix PDE solves, 243686 raw draws, 45924 full geometry checks and 1397
accepted edits. Fine gains are positive in 179 runs and zero in one; none
exhausts the raw budget. Refill4 flux versus fixed8 flux gives 24 wins, zero ties,
six losses and a median matched workflow-time ratio 2.403. On improved inputs
alone: 14/20 wins, mean fine Q difference 0.006146 and median time ratio 2.598.
Multi-choice availability on those inputs rises from 4.13% to 97.85%. Within
four-choice pools flux beats uniform ranking 26/30, with no identical masks.
Refill2 flux has 19/30 wins but essentially no mean advantage on improved inputs.
No equal-time, transfer or learned-policy advantage is established.

A turn interruption stopped the process after 97 completed jobs. Their summaries
and final masks remain hash-identical. The unfinished attempt is retained with
5--6 extra PDE solves (one in-flight completion is uncertain), separate from the
matrix. Finalization found an all-zero derived refinement CSV and completion log
in the last pre-interruption job. Original bytes are archived; the table was
recovered from independently verified fields and boundary fluxes without another
solve or altered summary/array. Two unreadable NPZ files belong only to the
discarded interrupted attempt; they are preserved and excluded from outcomes.

The independent audit passes all 180 completed jobs, exactly replays all 2880
selected valid edits and checks all 3780 saved fields with no new PDE solve.
Maximum recomputed conservation error is 6.0276e-13. Two additional readable
interrupted fields are checked separately. Frozen source identity and completed
file hashes pass. See feasible_pool_results.md for tables, figures and raw data.

The user reports an RTX 4060; local nvidia-smi verifies RTX 4060 Laptop GPU,
8188 MiB VRAM, driver 616.87. The project environment has no Torch/CuPy/JAX.
The separate workload profile uses three new PDE solves, seed 190000, and
matches the saved input's Q at 100/200/400. Profiling finds matrix assembly
dominates the sparse-solve phase. A 256-candidate linear policy call takes
5.68 microseconds; geometry validation accounts for 78.3% of matrix environment
time. No GPU framework is installed without a suitable validated workload.
All new experimental cost including profiling and interruption is 3788--3789
PDE solves; regression tests are separate. See gpu_readiness.md.

## 2026-10-19 â€” Source amplitude suggestion checked and display improved

The user notices most of the shared linear concentration plot appears dark
and proposes raising the source concentration. Derived the distinction between
increasing the physical drive C0-Cb and normalized source u=1. Under the current
linear constant-coefficient model, dimensional concentration/flux scale with
that drive; u, dimensionless Q, eta and geometry rankings do not. The source
amplitude cannot remove relative screening in this model.

Implemented an independent scaled-RHS diagnostic, with analytical slab tests,
irregular-domain checks, nonzero-Cb reasoning and fixed-bound display guards.
Production validates factors 1/2/5 at Bi=.1/1/10 on the slab and two saved
irregular chamber endpoints: 27 solves, normalized field error <=8.88e-16,
normalized flux error <=4.44e-16, eta unchanged and all three chamber rankings
unchanged. Conservation error <=1.03e-12. Reference source-one fields match the
original stored Bi=1 fields.

Applied the valid observation by adding explicitly labeled common power color
scales (gamma=.35, bounds 0--1), plus the shared linear reference. Source u=1
and all physical quantities and optimizer inputs remain unchanged. Forty-eight
input hashes verify the original masks, fields and summaries were not modified.
The complete 12-panel view and a side-by-side two-geometry illustration were
visually inspected. See source_amplitude_derivation.md and source_display_results.md.
The full regression suite passes 226 tests in 127.69 seconds; Ruff checks and
formatting pass. All 78 executed source-file hashes and 48 historical input
hashes match after validation and plotting.

## 2026-10-19 â€” Feasible-pool RL training completed

The user explicitly requests starting RL now. Connected the previously validated
four-choice bounded-refill proposal mechanism to the existing linear Expected
Sarsa environment, preserving all physical constraints and the fixed source.
Added disjoint proposal/selection streams for the opt-in refill protocol,
charged bootstrap pools, per-decision choice counts, atomic candidate artifacts
and interrupted-rollout preservation. Replica checkpoints validate fixed priors
and cumulative update counts. Existing configurations keep fixed-pool defaults.
Five new integration tests bring the regression gate to 231 passing tests in
127.95 seconds; Ruff checks pass. No GPU dependencies were added.

Frozen configs/train_rl_feasible.yaml before outcomes: five independent replicas,
24 episodes each, twelve canonical/improved training starts, fixed flux prior one,
discount one, zero invalid reward and retained raw-proposal exploration floor.
New held-out generation excludes 771 known masks. Production
20261019T133653820304Z_pioneer_rl_2382a80e completes in 2280.65 seconds with
120 training episodes, 2573 real parameter updates and 120 frozen evaluations.
All budgets complete. Cost is 5014 new PDE solves: 414 preparation, 40 preflight,
2040 training, 2040 evaluation and 480 refinement. There are no interruptions;
the reused training pioneers' 450 historical generating solves are separate.

Training choices are meaningful in 96.70% of decisions. Refined test comparisons
favor learning over raw uniform in 29/40 shared pairs and over pooled uniform
in 27/40. Against its frozen flux initialization, learning wins only 17/40;
mean delta Q is +0.00177676 but median is -0.00164557. Validation favors flux
in 13/20 pairs. No dependable superiority over the strongest heuristic is
established and no replica is selected/promoted. Learned median refined gain
over its own initial geometry is 1.224%, which alone is not a learning benefit.
The five replicas share starts/controls; their comparisons are not independent.

The artifact audit passes all 271 stored rollouts, 4977 saved fields, 4154 exact
selected edit replays and 2573 update replays. It checks 572671 raw draws,
110760 full geometry validations and 1849 accepted edits. Independent maximum
conservation error is 5.73266e-13, with no additional PDE solves. Thirty-six
training-bank checks and one slab preflight solve were not saved as field arrays
and are explicitly excluded from that field count. All 78 executed source
hashes match. A completed-resume check on an isolated copy skips all 120 training
episodes and 120 evaluations, changes none of the five checkpoints and solves
no new PDE. Figures show all four test starts at the first configured replica
and first rollout seed, with common contrast and separate Q400 annotations.
See refill_rl_results.md and models/refill_rl_2382a80e/ for retained models.

## 2026-10-20 â€” Diagnosed learning and launched duration continuation

The user approves the proposed next steps. A read-only diagnostic reconstructs
all five final policies exactly, replaying 2573 updates with round-trip CSV
precision. It reconstructs the 1252 frozen learned evaluation decisions from
saved proposal metadata and incumbent concentration fields, checks selected
features/probabilities, and finds a 58.55% ranking-change fraction versus the
fixed flux prior. This is not a counterfactual gain estimate: only selected
candidate outcomes were solved. No new PDE is used by the diagnostic.
Diagnostic run: 20261020T161304514936Z_learning_diagnostic_ef1e29c9.

Implemented a new duration configuration, continuation runner and checkpoint
report without changing the historical solver/rollout/policy modules. Freeze
the first 24 episodes' exact exploration arithmetic and keep its final level
thereafter, instead of stretching decay over 96 episodes. Preserve the original
five replicas, twelve-start permutation curriculum, 24 features and all physics
and geometry constraints. Pin 268 historical input files and reuse the exact
24-episode prefixes; reconstruct episode-12 weights from their recorded updates.
Each new episode is atomic; snapshots at 12/24/48/96 are immutable and evaluated
after the fixed training schedule finishes. All six known validation/test masks
are now explicitly development data with fresh seeds 263000/263001. No final
test or model promotion is claimed. Twelve shared flux controls plus 240 frozen
model evaluations are planned, each with endpoint refinement at 200/400.

Tests verify exact prefix exploration, saved-field flux reconstruction,
interruption/resume through both training and evaluation, and completed-resume
skipping all new numerical work. The full regression gate passes 234 tests in
308.33 seconds; Ruff checks and formatting pass 119 Python files. The new run
retains 82 executed source hashes and the previous audited run remains intact.

Launched independently in a hidden background process with captured logs:
20261020T162617162903Z_learning_duration_789b687b, launch metadata directory
20261020T162615Z_a050b102. The source/config/bank checks and 40-solve preflight
pass and the worker reaches new episode 25. Status is running, not completed.
Maximum planned new cost is 6120 continuation + 5292 development evaluation +
40 preflight = 11452 PDE solves; the reused 2040 training solves remain separate.

## 2026-10-20 â€” Recovered duration training after the user's PC crash

The process was stopped and status.json contained 97 zero bytes. All five
committed model checkpoints remained readable: 48, 48, 26, 24 and 24 episodes.
Verified the 82 frozen source hashes, 268 pinned historical inputs, geometry
bank, checkpoint curriculum/exploration/update counts, and all fifty newly
committed episodes. Their 849 saved fields satisfy independent finite-volume
balances and constraints; all recorded learning updates reproduce the latest
checkpoint digests exactly. Maximum independent conservation error is
5.50338e-14. This recovery audit performs no new PDE solve.

Preserved the damaged status bytes, copies/hashes of every committed checkpoint,
the audit helpers, launch metadata and verification in
crash_recovery_20261020T183042746209Z/ inside the active run. Resume uses the
unchanged config/source and replays only model 230002 episode 27. The original
partial episode is retained in training/replica_230002/interrupted_attempts/
episode_0026_568af0a3/; three persisted candidates give conservative bounds of
3--5 interrupted PDE solves, separate from the planned study cost.

New independent hidden launch: 20261020T183053Z_d2b27c19. Episodes 27 and 28
complete successfully, stderr is empty, and training continues toward all five
96-episode policies. Earlier completed episodes and model snapshots are retained.

## 2026-10-21 â€” Literature-inspired bank and measured-response trajectories

Implemented nine finite research geometry families with fixed area/source and a
common perimeter cap 8. Their generators are proxies rather than anatomical
reconstructions. The outcome-blind bank contains 45 distinct evaluation masks
and seven disjoint training masks; maze and Hilbert families are unseen in
training. The common cap differs from previous cap-4 work. Raw square-Koch
level-2 contacts are explicitly regularized, with the cutoff recorded.

Added independent P1 FEM, physical-polygon/raster comparisons, a signed-response
ridge model, heuristic annealing, exact-area/perimeter cell exchanges, atomic
solve journals and exact interruption/resume tests. This regressor is distinct
from the historical RL value learner. The full regression gate passed 278 tests
before launch. Frozen study 20261021T122626095890Z_morphology_98e4aafa launched
at 12:26 UTC. It plans 195 searches and 36,170 training/search finite-volume
solves, plus catalog, diagnostic and independent validation costs.

All seven training episodes finish with 455 solves and 448 signed-response
updates. At 13:04 UTC all 135 main comparison searches are complete; the deep
and exact-perimeter phases remain active. The first 90 endpoints pass prescribed
refinement and independent FEM validation, and further validation overlaps the
two search workers in a separate single-thread process. Frozen search sources,
settings and budgets are unchanged. Do not report the entire study as finished.

Independent postprocessing adds complementary energy estimates, a full saved-
journal audit and a conducting-collar model sensitivity. A partial audit exactly
replays 8,195 fields and 3,008 parameter updates with zero new PDE solves.
A separately derived analytical 2D strip reveals why source-touching absorbing
corners do not generally have a finite large-Bi flux plateau. Five strip tests,
twelve energy tests, two collar tests and three audit tests pass, 22 targeted
checks total. No combined 300-test rerun is claimed. The main package remains
frozen during execution.

Completed initial-bank results do not assume nature-inspired superiority:
at Bi=1 the representative rectangle has Q400=0.492772, versus 0.455786 for
the sac proxy with more membrane. At Bi=100 family spread is much smaller than
observed grid changes. These are unoptimized representatives with unequal
perimeters, not final family or method conclusions. The completed sweep is
plotted in postprocess/catalog_review with zero additional PDE solves.

Prepared a manuscript draft, explicit interpretation limits and a final endpoint
mechanism figure that reuses checked fields. The final figures, comparisons,
numerical ranking and boundary-sensitivity results still await completion.


## 21 October 2026 — completed paper-evidence study

Completed 195 original searches, 90 paid classical controls and all independent
reviews. Total committed production cost 50,721 solver calls; 285 checked
endpoints, 284 separated initial/final improvements, two unresolved leading
masks. The 47,780-field audit and 12,288 exact learning-update replays pass.
Learning does not improve on greedy flux guidance on descriptive average.
Source-wall sensitivity, source-corner analysis and proximal centroid shifts
limit biological interpretation. A post-run curve-indexing presentation bug
was corrected with original plots and source diffs retained; numerical results
are unchanged. Manuscript, supplement, tables and vector figures are collected
in experiments/paper_review_20261021. Public release and submission are unperformed.
