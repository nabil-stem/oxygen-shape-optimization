# Claim-to-evidence review

Completed computational claim review, 21 October 2026. The main study,
supplemental review and added classical controls have finished. Evidence is available in the public v0.1.0 release and the curated paper view.
Human scientific review and journal publication remain separate activities.

| Proposed claim | Evidence and required qualification | Current disposition |
|---|---|---|
| The PDE conserves total oxygen transfer | Analytical slab, independent face-balance replay, source/Robin conservation, symmetry and permeability tests | Complete: 306-case numerical gate, 47,780 saved-field replays and 12,288 exact parameter updates; corrected plotting gate recorded separately |
| Search improves finite designs | Endpoint Q400, independent FEM and nonoverlapping initial/final energy estimates; actual area, membrane and feature checks | 284/285 returned endpoints have separated practical initial/final estimates; all 285 pass endpoint checks |
| The learned selector improves optimization | Five-method paid comparison on the same 45 known development starts; include 455 warm-training solves | No overall advantage over flux guidance; uniform and paid-pool controls also lose on descriptive average (all 225 equal-budget endpoints complete) |
| Better one-step ranking implies better long-run search | Separate paid four-choice diagnostic versus complete trajectories | Unsupported: initial-state regret improves while aggregate learned endpoints are worse than greedy flux |
| More complex membrane necessarily transfers more oxygen | Nine representative starts, actual P and Q, Bi sweep and exact-perimeter controls | Rejected as a general rule in this model; common perimeter cap is not equal perimeter |
| All Bi<1 geometries are nearly uniform | At Bi=0.01, initial eta400 is 0.949 for the rectangle and 0.525 for the Hilbert-like path | Unsupported at that parameter; the fixed-domain Bi-to-zero limit is distinct from assuming a uniform finite-Bi regime across shapes |
| The chosen endpoint is the best possible geometry | All internal states are coarse-grid searched; selected coarse-best endpoints receive fine checks and energy comparisons | Unsupported; report best observed returned endpoint and overlapping contenders only |
| Nature's acinus is optimal or suboptimal | Would require measured 3D anatomy, physiological parameter justification, ventilation, blood-side transport and biological constraints | Outside this study; proxies and a planar objective cannot establish it |
| High permeability gives a finite universal transport plateau | Independent strip eigenfunction series and source-corner analysis | False for touching ideal source/sink corners; finite-Bi behavior and the completed collar/analytical-strip results are reported |
| Agreement between FVM and FEM validates smooth or anatomical geometry | Same-staircase FEM, plus separate declared physical-polygon controls | Unsupported; solver error and geometry bias are distinct |
| Energy ranges rigorously certify rankings in floating arithmetic | Complementary estimates with residual correction and practical floating guards | Exact-arithmetic variational argument supported; no interval-arithmetic certification claimed |
| The Laplace/Robin optimization problem or the classical methods are novel | Close cooling-fin and shape-optimization prior work | Not claimed; candidate contribution is the reproducible benchmark and measured limitations |
| Earlier RL runs and the new response-guided study show one continuous algorithm improvement | Different membrane caps, search settings and evaluation designs | Unsupported; historical cap-4 work remains exploratory context, separate from cap-8 evidence |
| Five pairs measure independent geometry, search and training variance | One geometry variant paired with one search seed, one shared warm model | Unsupported; descriptive paired outcomes only, no crossed variance or training-replica estimate |

The paper can retain a failed learning hypothesis or an unfavorable geometry
comparison. It must not retain an unsupported interpretation merely to make
an earlier experiment appear successful. Historical exploratory outputs remain
in the versioned source snapshot; their conclusions are not silently promoted into the frozen
study. Changing protocols or budgets requires a separately identified experiment.

Project code and primary data are publicly released under MIT and CC BY 4.0
terms. Author identity is recorded in CITATION.cff. Human scientific and
mathematical review, final submission declarations, archival DOI deposition and
journal selection remain outstanding. A public artifact is not peer review or
an accepted article.

A post-run pandas index-alignment bug affected learning-curve rendering. The
corrected curves keep the actual initial and final solve counts; original
figures and source diffs are archived. No search/PDE result or reported endpoint
comparison changed. This presentation correction is distinct from numerical
validation and is covered by focused index-regression tests and a renewed gate.

The release regression passes **324 tests** locally and on Windows/Ubuntu.
[Validation records](https://github.com/nabil-stem/oxygen-shape-optimization/releases/tag/v0.1.0) accompany the full evidence.