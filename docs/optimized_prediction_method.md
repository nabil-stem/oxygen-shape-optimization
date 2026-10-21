# Fixed optimized-incumbent prediction study

The prior diagnostic used initial shapes. This study repeats the same predictor
tests on frozen search outputs near the membrane budget, with fresh initial
controls. It does not optimize or promote a new policy.

## Prespecified inputs

Select the first uniform run, seed 8000, for each family in the completed
equal-budget sensitivity benchmark. This rule is fixed before the diagnostic;
the run is not selected for maximum Q. Deep cavity has Robin perimeter 4.00
and rectangle 3.96, against the unchanged 4.00 budget.

Pin each mask's file SHA-256. Verify source seed/method/status, physics, grid,
solver settings, constraints, original mask, source collar, topology and feature
checks. Copy both masks and their source config/summary into the new experiment.
Recompute Q and compare it with the source summary before sampling any edits.
Reject mismatches; do not repair a frozen input silently.

## Protocol

Use initial and optimized states for both families, Bi=1 and n=100. Five fresh
replicates use calibration seeds 8400–8404, held-out 9400–9404 and independent
radius-family seeds 10400–10404. Equal seed labels across states do not imply
identical physical edits: their eligible face populations differ.

Keep prefixes 4/8/16/24, 48 held-out valid edits per replicate, the four scoring
ablations, flux control, three matched radii, constraints and bandwidths from
prediction_method.md. Every pool has a hard cap of 6000 proposal attempts.
Exhausted quotas remain incomplete and all failures remain in the report.
The geometry is fixed throughout; calibration and test masks remain disjoint.

Report per-state rank metrics, positive-effect frequency, valid/attempt counts,
radius feasibility, all costs, and exact reference identities. Score statistics
requiring a calibration prefix are conditional on attaining it; show available
and planned replicate counts. Do not increase budgets or relax constraints to
force complete groups after observing results.

Refine selected positive/negative effects at 200/400 as before. The diagnostic
targets the discrete n=100 problem; only selected effects receive full PDE
refinement. One frozen incumbent per family cannot establish a universal
optimization-trajectory effect. No model or score is selected from this study.
