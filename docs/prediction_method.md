# Fixed-geometry predictive diagnostic: prespecified protocol

The previous optimizer comparison showed mixed sensitivity-guided results.
This experiment tests prediction without changing the optimizer or geometry.
All probes and held-out edits use the same unedited reference shape per family.
Constraints and finite-volume physics remain those of the preceding study.

Use five replicates, deep-cavity/rectangle shapes, Bi=1 and n=100. Calibration
seeds are 8200–8204; held-out seeds 9200–9204; controlled-radius seeds
10200–10204. Within each replicate, collect 24 unique valid calibration edits
and 48 unique valid held-out edits. Training/held-out masks must be disjoint.
Report incomplete quotas and every invalid attempt. Solve every valid edit;
there is no acceptance or geometry update. All calibration/validation costs count.

Freeze spatial bandwidth 0.15 and radius bandwidth 0.01. Compare these four
prespecified ablations at calibration prefix sizes 4, 8, 16 and 24:

1. Spatial sum of signed delta-Q/reallocated-area (the current map).
2. Spatial sum of signed delta-Q (absolute-gain ablation).
3. The first response with a Gaussian factor in addition radius.
4. The second response with that radius factor.

At a query edit, K=exp(-sum of squared addition/donor point distances/(2*l^2)).
The radius factor is exp(-(r_query-r_train)^2/(2*l_r^2)). These are definitions
of diagnostic scores, not calibrated delta-Q predictions or shape derivatives.
For ranking, clip the summed signed scores below at zero, matching the current
sampler's ranking. Uniform mixing/normalization are monotone transforms on one
fixed geometry and do not change this order. Negative-score ties therefore
remain ties instead of receiving artificial predictive credit.

Report Spearman rank correlation with actual global delta-Q and delta-Q/area,
top-quartile mean global delta-Q, and sign accuracy. Compare sign accuracy with
the always-positive baseline so a mostly-positive pool cannot look predictive
merely by predicting improvement everywhere. The flux heuristic is an additional
fixed ranking control. Undefined constant-vector correlations are null. Use
fractional weights at top-quartile ties; never break ties using observed Q.

Stratify scores by small/large addition radius around 0.065. Independently draw
eight uniform face-pair families per replicate. Replay each family's identical
angular/jitter RNG stream at radii 0.05, 0.065 and 0.08. Record every feasibility
outcome, including families with no valid edits. This selection is independent
of whether a family happens to produce a useful candidate. Compare valid-family
radius responses descriptively; do not treat them as derivative estimates.

Refine the largest positive and negative held-out effects in each family at
200/400 as descriptive numerical checks, retaining the same staircase masks.
Predictive rank metrics otherwise describe the n=100 discrete reference problem.
Do not tune or promote an ablation based on these diagnostic test scores.
Any revised optimizer needs a new frozen protocol and fresh search seeds.

The same harness also supports verified frozen input states. See
optimized_prediction_method.md for the separately prespecified late-state
comparison and its fresh initial controls. Sources are identified by seed,
method and pinned mask hash, verified against the original canonical geometry,
and copied together with their config/summary. Source fluxes are re-solved and
checked for every case before any perturbation is sampled. Loader and metadata
failures are distinct from an exhausted candidate-feasibility quota.
