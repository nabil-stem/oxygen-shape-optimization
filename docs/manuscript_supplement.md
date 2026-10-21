# Technical supplement and evidence guide

Completed local computational supplement for human review, 21 October 2026. Read with manuscript_draft.md and the
completed experiment reports. A local evidence package is not a public data
deposit, a software license or an accepted manuscript.

## S1. Governing equations and conserved discretization

The air-domain equation is Laplacian(u)=0 with source u=1 and reactive wall
du/dn+Bi*u=0. The optional Neumann boundary belongs only to a declared
nonabsorbing conducting-wall segment; absorbing distal membrane remains Robin.
Total transfer is Q=integral_R Bi*u ds and eta=Q/(Bi*P_R). Geometry acceptance
and search labels use global Q, not eta or a concentration threshold.

On cell width h, interior conductance is 1, the Dirichlet source conductance
is 2, and Robin face conductance is Bi*h/(1+Bi*h/2). Source and absorbed flux
use the same assembled face conductances. See [derivations](derivations.md),
[numerical validation](numerical_validation.md) and the archived solver sources.
The independent triangular P1 solver uses consistent Robin edge integration.
Refining the frozen mask subdivides the same staircase; it does not produce
a smooth physical boundary. Separate physical-polygon checks address that issue.

## S2. Practical flux estimates

The exact-arithmetic primal/complementary argument, explicit RT0 bulk norm,
boundary terms, cell-balance residual correction and floating guards are given
in [complementary_flux_bounds.md](complementary_flux_bounds.md). A conforming
FEM energy is the upper estimate, and an H(div) reconstruction from FVM fluxes
provides the lower estimate. The reported ranges are practical numerical
estimates on each fixed staircase, not interval-arithmetic certification or
uncertainty in biological parameters. FVM point estimates need not lie inside
these variational ranges. Observed grid changes are separate diagnostics.

[permeability_response.md](permeability_response.md) derives monotonicity,
concavity and the envelope derivative for a fixed domain. The analytical slab
identity and all 27 saved geometry/grid curves pass their checks, without
additional PDE evaluations. These facts do not impose a finite high-Bi plateau
or a shared small-Bi threshold across different geometries.

## S3. Source-wall boundary model

[source_corner_limit.md](source_corner_limit.md) derives the strip eigenfunction
series, spectral root equation and analytical truncation-tail bound. It explains
why touching ideal source/sink corners invalidate a universal finite high-Bi
plateau. The production strip length is 0.5 and width 0.1, with a Neumann end;
it is an independent solver check, not a searched geometry.

[collar_ablation.md](collar_ablation.md) specifies nonabsorbing proximal lengths
0, 0.06 and 0.12. For positive length ell, the common width-0.1 unreactive neck
gives Q<=0.1/ell. The 14 selected fixed cases comprise nine initial representatives
and five distinct original-study endpoints. Their selection is post hoc and
does not reoptimize shapes under the changed boundary conditions. Collar length
is a model hypothesis, not a measured anatomical parameter.

## S4. Geometry bank, resources and operational regularity

The [frozen primary protocol](morphology_protocol.md) and
[interpretation limits](morphology_limits.md) specify exact area 0.24, source
length 0.1, allowed unit box, protected x<0.12 collar, connected no-hole air
domain and Robin-length cap 8. Disk opening and residual-distance checks are
operational regularity rules; they do not prohibit every short pixel-scale bump.
An allowed bounding box does not fix occupied diameter, centroid or territory
coverage. The evidence tables also report actual footprint and centroid changes.

Nine finite generator families and all validity/identity-only retries are
retained in bank_manifest.json. There are 45 distinct evaluation masks, seven
training masks and zero mask overlap. Family names label the initial generator;
later edits can remove its distinctive morphology. Maze/Hilbert exclusion from
warm training does not turn the known development bank into a fresh final cohort.

## S5. Paid search, learning and interruption accounting

The main matrix has 135 searches at 129 solves, 15 long sac-family searches at
1,025 solves, and 45 exact-perimeter searches at 65 solves. All original budgets
complete. Paired disk redistribution has probability 0.75; a single-cell
area/perimeter-preserving exchange has probability 0.25. Every selected
candidate is evaluated by the conserved PDE solve. A minimum-four-proposal pool
refills for at least two valid choices, capped at 96 draws; the total raw-draw
limit is 500,000 per trajectory.

The learned response target is 100*deltaQ/Q_initial. The recursive ridge ranker
has 28 features, ridge 1, a fixed flux prior, a leverage bonus weighted 0.03 and
uniform eligible exploration probability 0.20. There is one 455-solve warm
training sequence; evaluations restore that same model and update from their
own observed selected responses. Unselected actions receive no invented labels.
Annealing uses temperatures 0.003 to 0.00002 and saved-best restarts at 256-solve
intervals. Asymmetric proposals are not asserted to sample an equilibrium law.

The [separate conventional-control protocol](classical_control_protocol.md)
adds 90 uniform-greedy and paid-pool-greedy searches on the same starts after the
original comparison was observed. Every pool-member solve is charged, and the
last pool can be budget truncated explicitly. Tests verify exact continuation
inside a pool and distinguish committed work from a possible 0-1 uncommitted
solve per interrupted attempt. Both studies retain rejected evaluated moves,
current and best states, source/input hashes and actual costs.

## S6. Statistical and interpretive scope

Five combinations per family pair one geometry variant with one search seed.
They do not separately estimate geometry and search variance. The new model has
one warm training sequence, so there is no training-replica variance estimate.
Learning curves show medians and interquartile variation, not confidence bands.
Pooled win counts and response errors are descriptive. Numerical range separation
does not establish statistical significance or population-level superiority.

The four-choice initial-state diagnostic spends 180 candidate solves and does
not train on its oracle labels. Its warm-ranker predictions exclude the leverage
bonus and random exploration used in trajectories. The separate selected-response
diagnosis uses pre-update observed labels; it cannot infer regret on unselected
actions or causally explain endpoint differences. Conventional search and modern
deep-RL/adjoint methods outside the tested edit space are not ruled out.

## S7. Execution, files and human review

[reproducibility.md](reproducibility.md) gives local installation and replay
commands; [morphology_execution.md](morphology_execution.md) records the passing
306-case regression, execution scheduling, archived sources and launcher repair.
The repository has no HEAD commit; exact hashes preserve provenance without
inventing retrospective history. The prepared Windows/Ubuntu CI has not been
executed remotely. Concurrent timings are descriptive, with training and
independent validation costs reported separately.

The main run is `20261021T122626095890Z_morphology_98e4aafa`; the conventional
extension is `20261021T141950978652Z_classical_72101312`. Each retains configuration,
seeds, numerical settings, geometry masks, fields, journals and reports.
The documented post-run curve-index correction preserves original plots and
numerical results, with four added regression cases. Exact historical replay
uses archived executed sources; current reporting includes that correction.
`scripts/paper_tables.py` checks reviewed input hashes and collects completed
tables without solving PDEs. `python -m scripts.paper_figures` creates vector
figures from those tables, and `scripts/winner_mechanisms.py` reuses checked fields.

Use the [claim review](claim_evidence_review.md) and
[related-work assessment](related_work_assessment.md) when judging the scope.
Human mathematical/subject review, authorship, licensing, persistent code/data
deposition and a venue-specific submission remain author activities. None is
implied by a successful test suite or by the existence of these local files.

Final corrected full regression: **310 passed in 511.75 seconds** on 21 October 2026.
The XML is retained in tmp/paper_gate_corrected_20261021/tests.xml and the review packet.
