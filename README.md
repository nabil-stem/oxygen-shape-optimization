# Oxygen diffusion shape optimization

Dimensionless, steady 2D diffusion with a fixed concentration source and absorbing
Robin membrane. The implementation includes a validated finite-volume reference,
constrained geometry search, screening/deletion studies, sensitivity guidance,
tabular and linear RL experiments, and a measured-response search benchmark.
Independent triangular FEM checks distinguish solver errors from raster-boundary
errors. This remains an exploratory 2D research prototype.

## Completed paper review, 21 October 2026

The [manuscript](docs/manuscript_draft.md), [technical supplement](docs/manuscript_supplement.md)
and [review packet](experiments/paper_review_20261021/README.md) collect the completed
five-method benchmark, 285 returned endpoints, independent FEM/energy checks and
50,721 committed solver calls. The tested response ranker does not improve on
greedy flux guidance on descriptive average. Two leading geometries have
overlapping practical flux ranges; no global or natural-acinar optimum is claimed.
The packet preserves original plots and documents a post-run curve-indexing fix.
The manuscript is unreviewed and has not been submitted to a journal.

Author: **Nabil ES-SAADI, independent researcher**. Code: [MIT](LICENSE).
Original data/figures/research text: [CC BY 4.0](LICENSE-DATA.md).
See the [research release instructions](RELEASE.md),
[data availability](DATA_AVAILABILITY.md),
[targeted novelty review](docs/novelty_review_20261021.md) and
[scientific reviewer brief](docs/scientific_review_brief.md).
The [v0.1.0 assets](https://github.com/nabil-stem/oxygen-shape-optimization/releases/tag/v0.1.0)
provide complete primary data with per-file hashes. Optional Triangle has
[separate terms](THIRD_PARTY.md). No archival DOI or human peer review is claimed.

## Install and run (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m oxygen_shape.cli validate-1d --config configs/slab_1d.yaml
.\.venv\Scripts\python.exe -m oxygen_shape.cli solve --config configs/rectangle_2d.yaml
.\.venv\Scripts\python.exe -m oxygen_shape.cli compare-shapes --config configs/baseline_shapes.yaml
.\.venv\Scripts\python.exe -m oxygen_shape.cli optimize --config configs/optimize_disks.yaml
.\.venv\Scripts\python.exe -m oxygen_shape.cli train-rl --config configs/train_rl.yaml
.\.venv\Scripts\python.exe -m oxygen_shape.cli test-deletions --config configs/deletion_hypothesis.yaml
.\.venv\Scripts\python.exe -m oxygen_shape.cli benchmark-sensitivity --config configs/sensitivity_benchmark.yaml
.\.venv\Scripts\python.exe -m oxygen_shape.cli diagnose-prediction --config configs/prediction_diagnostic.yaml
```

Each command creates a unique directory under `experiments/`, containing its
resolved configuration, metrics, initial/final geometry, field and boundary data,
figures, log, and summary with environment and Git provenance. Output paths in
YAML are resolved relative to the configuration file. No commits are made by the
commands; an uncommitted repository is recorded explicitly.
Each run also keeps a source snapshot with SHA-256 hashes, so the code behind an
experiment is retained even when the repository has no commit yet.

The solver uses square cells and explicit exposed faces, `u=1` at source faces,
zero flux at Neumann faces, and `du/dn + Bi*u=0` at membrane faces. Fluxes use the
same conductances as matrix assembly. Disconnected domains, missing sources,
invalid face labels, and failed numerical checks raise errors.

The 1D solution is linear, so this discretization reproduces it to roundoff on
every grid. Its validation plot shows numerical error, not an invented convergence
order. A separate 2D refinement check tests a non-linear field. At `Bi=0`,
`Q_ideal=0` and efficiency is undefined (`null` in JSON).

## Baseline comparisons

`compare-shapes` runs 135 solves: three grids (100, 200, 400), five Bi values
(0.01, 0.1, 1, 10, 100), and nine cases per grid/Bi. Its fixed-area group contains
rectangle, disk-like chamber, square, sinusoidal walls, branching tubes, a
Hilbert-like path, and a deep cavity, all with area 0.24, source length 0.1,
the unit bounding box, and membrane budget 18. Actual perimeter varies.
A separate rectangle/square group matches Robin length 3 within 4h and allows
area to vary. Body names include their connecting inlet corridor.

The output report includes shared-scale fields and boundary fluxes, Q, ideal Q,
eta, area, perimeter, diameter, screening diagnostics, grid changes, and rank
stability. `cases/` holds full numerical data for every solve. The root geometry
files are the finest-grid fixed-area rectangle at the lowest Bi, as labelled in
the summary. Coarse-grid and raster matching effects are reported explicitly.

## Optimization and RL

`optimize` proposes area-balanced attached disk edits, validates geometry, solves
the PDE, and accepts positive global flux gains. The default example fixes area
0.24, source length 0.1, collar x<0.12, bounding box, membrane budget 4, and
an operational morphology feature constraint checked on 100/200/400 grids.

`train-rl` learns a NumPy tabular Q-policy with eight coarse observation states
and twelve region/radius actions. It saves checkpoints, freezes the final table,
and compares it with uniform search and an untrained policy on five held-out
seeds at equal evaluation solve budgets. Training and refinement costs are
reported separately. Frozen policies fall back to uniform proposals after fifty
consecutive invalid attempts; this safeguard is shared by the untrained control.
Reward uses actual globally re-solved flux improvements.

To reuse a checkpoint without training, keep its metadata file alongside it:

```powershell
.\.venv\Scripts\python.exe -m oxygen_shape.cli evaluate-rl --config configs/train_rl.yaml --policy models/disk_q_policy.npz
```

The metadata guard checks the geometry, physics, constraints, action space, and
validation grids; choose a seed outside training. These experiments establish
neither policy generalization to unseen geometries nor a globally optimal shape.

See [the derivation](docs/derivations.md),
[reference validation](docs/numerical_validation.md), and
[baseline results](docs/baseline_results.md). Raster perimeter uses exposed face
length; curved/diagonal boundaries retain orientation bias. See also the
[random optimizer](docs/optimization_v0.md), [RL method](docs/rl_method.md), and
[RL results](docs/rl_results.md). The morphology constraint is a raster proxy,
not a certified continuum neck bound. Monte Carlo and 3D are subsequent stages.

The user-requested bounded follow-up study is documented in
[two-hour study](docs/two_hour_study.md), with links to its automatically updated
report, job plan, checkpoints, and live status.

## Controlled deletion experiments

`test-deletions` compares random donor faces with the bottom 20% of initial
Robin flux. Every sampled cut is globally re-solved both as a deletion alone
(area loss bounded by 5%) and as an exactly area-balanced cut/addition pair.
It preserves the source collar, connectivity, hole exclusion, membrane budget
and minimum-feature checks on the search and refinement grids. Every rejected
proposal is recorded; incomplete quotas and numerical failures remain visible.

The completed audit evaluates 192 candidates over two shapes and three Bi
values, using 268 PDE solves. Low-flux deletion alone produces both gains and
losses that survive refinement. See [results and examples](docs/deletion_results.md)
and [the implementation plan and next milestone](docs/next_steps.md).

## Empirical sensitivity-guided search

`benchmark-sensitivity` compares uniform, flux-guided and empirical-response
samplers using the same batched greedy acceptance schedule and total PDE budget.
Sensitivity probes measure global flux change per reallocated area. A joint
addition/donor map guides subsequent candidates on that fixed geometry; all
probe solves count toward the budget and every probe can be accepted. The map
is discarded between rounds, and uniform exploration remains active.

The default comparison uses five fresh seeds, deep-cavity/rectangle families,
Bi=1, 41 search solves per run, and 200/400-grid validation. It saves signed
probe responses, probability maps, every invalid attempt, equal-cost histories,
physical constraints, wall time and paired comparisons. This is an empirical
proposal heuristic, not a formal shape derivative or a proven speedup. See
[the protocol and derivation](docs/sensitivity_method.md).
The [completed comparison](docs/sensitivity_results.md) records 30 runs and
1350 PDE solves including refinement. Sensitivity wins 3/5 seeds against
uniform in each family; a reliable advantage over the flux control remains
unproven, so the scientific advantage gate stays open.

`diagnose-prediction` tests those scores on disjoint calibration and held-out
edits of a fixed geometry. It compares per-area/absolute flux responses, radius
conditioning, calibration-pool sizes and the flux control, while recording every
geometric rejection and independent matched-radius family. It measures prediction
quality without accepting edits or revising the search policy. The
[prespecified protocol](docs/prediction_method.md) describes sampling, costs,
tie handling, limitations and the always-positive classification baseline.
The [completed diagnostic](docs/prediction_results.md) uses 797 PDE solves and
finds that sparse calibration and geometric feasibility matter; the flux
control remains stronger on these two fixed initial geometries.

To test later search states with fresh initial controls, run:

```powershell
.\.venv\Scripts\python.exe -m oxygen_shape.cli diagnose-prediction --config configs/optimized_prediction.yaml
```

This variant pins the first uniform seed-8000 outputs by SHA-256, checks their
source metadata and constraints, recomputes their fluxes, and copies the verified
inputs into the new run. All input verification precedes candidate sampling.
The five fresh replicate streams, 6000-attempt pool cap, unchanged geometry
constraints and treatment of incomplete quotas are specified in
[the frozen-incumbent protocol](docs/optimized_prediction_method.md).
The [completed study](docs/optimized_prediction_results.md) uses 1531 PDE solves
across four fixed inputs and five replicates each. Feasible proposal yield falls
sharply near the perimeter budget; predictor performance depends on the shape.
No checkpoint or search policy is changed using these held-out results.

## Geometry feasibility and pulmonary references

The supplied 2022 and 2025 acinar morphology papers are summarized in
[the research notes](docs/pulmonary_papers.md), with source identities, PDF page
references, checked equations, limitations and implications for this project.
Their structural count/path objectives do not replace total oxygen flux Q.
The human polyhedron model is coarse and does not resolve individual alveolar
walls; no dimensional calibration or biological coverage law is imported.

The geometry-only perimeter-pool variant is evaluated separately:

```powershell
.\.venv\Scripts\python.exe -m oxygen_shape.cli benchmark-feasibility --config configs/feasibility_benchmark.yaml
```

It retains uniform exploration and the existing full geometry/PDE checks.
All eight raw draws count toward its cost. The
[prespecified protocol](docs/feasibility_method.md) separates shared-pool
feasibility from fresh-seed equal-PDE-budget searches on frozen optimized inputs.
The existing default samplers and RL checkpoint remain unchanged.
The [completed feasibility study](docs/feasibility_results.md) records 10240
audit proposals, 20 completed searches and 424 PDE solves including refinement.
Pool selection reduces measured search time on these inputs but does not
consistently improve flux at equal PDE budgets.

To separate cheap screening from minimum-perimeter selection, run:

```powershell
.\.venv\Scripts\python.exe -m oxygen_shape.cli benchmark-feasibility --config configs/screening_ablation.yaml
```

The [screening ablation protocol](docs/screening_ablation_method.md) adds
uniform choice among cheaply screened candidates. Both pool rules share audit
proposals and exploration coins, followed by globally solved selected responses
and a three-method fresh-seed search. Shared audit solves are counted once,
invalid choices remain explicit, and no response tunes the frozen search rule.
The [conditional-distribution derivation](docs/screening_distribution.md) explains
why screened-uniform preserves ordinary rejection sampling's valid-edit law
under stated assumptions. An exact rational finite-example check accompanies it;
finite budget caps and measured runtime remain separate questions.
The [completed ablation](docs/screening_ablation_results.md) uses 1297 PDE solves
and completes all 30 searches. Screened-uniform is faster than ordinary sampling
in all ten matched pairs here, but does not consistently beat minimum-perimeter
on flux. Defaults and checkpoints remain unchanged.

## RL with validated pioneers and strong controls

The multi-start candidate selector uses a separate linear Expected Sarsa model,
budget-aware incumbent/candidate features and the validated PDE environment.
The old tabular policy remains intact. Its starting bank includes ten pinned
improved development masks and two canonical curriculum starts, with new
validation/test pioneers selected and frozen before training.

```powershell
.\.venv\Scripts\python.exe -m oxygen_shape.cli train-rl-candidates --config configs/train_rl_candidates.yaml
```

Use `--prepare-only` to build the bank and run readiness checks, or
`--resume "RUN_DIRECTORY"` to continue a compatible prepared/interrupted run.
The [fixed training protocol](docs/pioneer_rl_method.md) specifies five replicas,
five controls, disjoint data, all compute costs, atomic episode checkpoints and
frozen held-out evaluation. No model is promoted automatically.
The [first completed study](docs/pioneer_rl_results.md) trains five policies over
120 completed episodes and completes all 100 held-out rollouts, using 4594 new
experimental PDE solves. All replicas beat uniform on 6/8 test pairs, but do
not reliably beat the flux heuristic. Checkpoints, starting masks, full costs,
refinement checks and the documented aggregate-label correction are retained.

## Reward, flux-prior and feasibility revision

The [revision protocol](docs/rl_revision_method.md) compares four reward settings,
then isolates a fixed flux prior and full geometry screening. All settings use
the same training budgets and constraints. Validation selects a setting across
all five replicas; fresh tests also include the original and combined settings,
plus cheap/full-screen controls. Preparation and training costs remain explicit.

```powershell
.\.venv\Scripts\python.exe -m oxygen_shape.cli revise-rl --config configs/rl_revision.yaml
```

Use `--prepare-only` and `--resume "RUN_DIRECTORY"` for compatible checkpoints.
The previous study's masks are excluded from the new held-out bank. Full
screening uses geometry checks only, and records every check and its time;
it does not guarantee more flux or lower total computational cost.
The [completed revision study](docs/rl_revision_results.md) trains 35 policies
over 420 episodes and completes 176 validation/152 final-test rollouts, with
14509 new experimental PDE solves. Validation prefers the original reward;
full screening improves selected feasibility, but the combined learner matches
the untrained full-screened final geometry on 35/40 outcomes. No reliable
advantage over the strongest flux control is established.

## Distinct valid proposal pools

The [valid-pool protocol](docs/feasible_pool_method.md) tests whether bounded
refilling supplies meaningful choices before further RL training. Fixed pools
of eight raw edits are compared with targets of two/four distinct valid edits,
under uniform and incumbent-flux ranking, identical physical constraints and
explicit raw/PDE budgets. It records all rejected work and endpoint refinement.

```powershell
.\.venv\Scripts\python.exe -m oxygen_shape.cli benchmark-valid-pools --config configs/feasible_pools.yaml
```

This is a development comparison on previously observed geometries with fresh
rollout seeds. A [critique of the shared PINN topology post](docs/pinn_topology_critique.md)
explains the relevant constraint lessons and the limits of transferring its
fluid-flow formulation to a permeable oxygen membrane. The deterministic
finite-volume reference remains the state solver.

The [completed valid-pool study](docs/feasible_pool_results.md) finishes all 180
runs and 3780 matrix PDE solves. Four-choice flux ranking beats fixed8 flux on
24/30 matched pairs, at a median 2.40 times the workflow duration. On improved
starts, pools offering multiple eligible choices rise from 4.1% to 97.8%.
This does not establish equal-time or RL superiority. The independent audit
checks all 3780 fields and replays 2880 candidate edits; recovery and discarded
interrupted artifacts are documented. See [GPU profiling](docs/gpu_readiness.md)
for the verified RTX 4060 Laptop GPU and measured CPU bottlenecks.

## Concentration source and comparable color scales

[Source/display validation](docs/source_display_results.md) checks the suggestion
to increase source concentration. In the present linear model that scales
absolute transfer while leaving normalized fields, eta and rankings unchanged.
The 27 independent checks confirm this. New common power-color views reveal
small concentrations while retaining source u=1, identical fields, physical
metrics and a shared linear reference. No per-geometry autoscaling is used.

## Train RL with admissible candidate pools

The [refill RL protocol](docs/refill_rl_protocol.md) trains five independent
Expected Sarsa replicas on strong starting geometries. Each pooled decision
targets four distinct admissible edits with bounded, fully counted refill.
The fixed flux prior provides a strong initialization; evaluation compares
learned residuals against that frozen flux control and four other controls.

```powershell
.\.venv\Scripts\python.exe -m oxygen_shape.cli train-rl-candidates --config configs/train_rl_feasible.yaml
```

Use `--resume "RUN_DIRECTORY"` after the initial bank preparation finishes.
Every completed episode has an atomic model checkpoint. Resume verifies source,
config, bank and numerical environment; interrupted uncommitted rollouts are
preserved before replay and their costs reported separately.

The [completed training](docs/refill_rl_results.md) finishes five policies,
120 episodes, 2573 parameter updates and 120 held-out evaluations in 38 minutes,
with 5014 new PDE solves and 231 passing tests. Refined tests favor learning
over raw uniform in 29/40 shared comparisons but over the frozen flux control
in only 17/40; no reliable advantage over that strongest heuristic is established.
All final replicas are retained for research in `models/refill_rl_2382a80e/`.

## Controlled continuation to 96 episodes

The [duration protocol](docs/learning_duration_protocol.md) continues the five
audited policies, preserving their exact 24-episode prefix and exploration
schedule. It evaluates immutable versions at 12/24/48/96 episodes against the
frozen flux heuristic on development geometries. It uses no new final test.

```powershell
.\.venv\Scripts\python.exe -u -m oxygen_shape.experiments.learning_duration --config configs/learning_duration.yaml
```

For a persistent background run with captured stdout/stderr:

```powershell
.\scripts\start_duration_training.ps1
```

The launcher rejects a duplicate active duration process. Resume the same study
with `--resume "RUN_DIRECTORY"` in the Python command after stopping any existing
process. Atomic policy checkpoints and the frozen source/config/bank identity
protect completed episodes. See docs/next_steps.md for the current run and logs.

## Literature-inspired geometries and long trajectories

The [literature review](docs/morphology_literature.md) motivates nine finite control
families: rectangle, hexagon, regularized square-Koch chambers, a maze, binary
branching, terminal hexagonal sacs, asymmetric sacs and a Hilbert-like path.
They are geometric proxies, not anatomical reconstructions. The [protocol](docs/morphology_protocol.md)
defines an outcome-blind bank with 45 distinct evaluation masks and seven training
masks, three methods, five search seeds, exact-perimeter controls and trajectories
with up to 1,024 evaluated edits each.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-validation.txt
.\.venv\Scripts\python.exe -m oxygen_shape.experiments.morphology_study --config configs/morphology_trajectories.yaml
```

For a hidden background process with two workers and one numerical thread each:

```powershell
.\scripts\start_morphology_study.ps1
.\scripts\start_morphology_study.ps1 -ResumeRoot "RUN_DIRECTORY"
```

The objective is total Q. Actual perimeter and utilization are reported separately.
Search mixes area-balanced disks with explicitly identified perimeter-preserving
cell exchanges; every proposal still passes geometry checks. The response learner
predicts measured signed global flux changes and can use annealing to leave a
local optimum while retaining its best geometry. It is a regression/bandit
baseline, not a replacement name for a new deep-RL model.

Run artifacts include per-solve atomic recovery states, independent FEM checks of
returned endpoints, fixed-physical-polygon comparisons, model parameters, costs,
figures and report.md. Source/config changes invalidate resume. No biological
optimality, globally optimal shape or reliable learning advantage is assumed.
