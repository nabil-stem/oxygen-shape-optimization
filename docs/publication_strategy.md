# Research critique and publication strategy

Assessment date: 20 October 2026. This document reviews existing results and proposes future work. It does not report new PDE solves, an independent continuum validation, or a completed final test.

There is material for a computational research paper, but submission readiness and novelty are not yet established. The defensible scope is constrained two-dimensional Robin diffusion and the comparison of search strategies. A claim that reinforcement learning discovered an optimal pulmonary acinus is unsupported.

**What the reasoning got right**

The deterministic reference solver came before the learned optimizer. Nondimensionalization, a reactive Robin wall, conserved face fluxes, an analytical slab check, symmetry and permeability tests provide a sound starting point. The optimizer evaluates the global PDE objective rather than treating a dark region in a plot as disposable. Fixed source protection, connectivity, area and membrane constraints address obvious ways to exploit an unconstrained objective.

Several initial intuitions were tested and qualified instead of silently retained: deleting low-flux wall has counterexamples; increasing source amplitude cannot change normalized fields in this linear model; more training has not established a consistent advantage over the flux heuristic. Auditable checkpoints and recovery records are useful research infrastructure.

These are strengths of the research process. A large test count, many PDE solves and reliable crash recovery are not, by themselves, scientific novelty or evidence of physiological validity.

**Where our thinking needs correction**

1. We let the choice of RL precede a sufficiently sharp reason to need learning. The first decision should have been whether a cheap physics heuristic leaves predictable improvement available under a fixed budget. More episodes and a larger network cannot answer a poorly posed comparison. I should have prioritized that diagnosis before extending training.

2. We sometimes moved between three distinct claims: the linear system is solved accurately; the optimizer finds better admissible masks; the shapes explain lung design. Each requires separate evidence. Conservation supports the first, paired search comparisons address the second, and neither establishes the third.

3. The best individual geometry is not evidence that its generating policy is best. It was selected after inspecting 1,078 complete refined endpoint rows; rows include repeated shapes and shared starts. It is a useful illustration and a candidate for further validation, not a population-level result.

4. Membrane budget and membrane equality must be distinguished. The current optimization fixes area and caps Robin perimeter; it does not fix the amount of membrane actually used. Its winner uses more membrane than its own start.

| Quantity, Bi = 1 | Its starting geometry | Selected endpoint | Change |
|---|---:|---:|---:|
| Area | 0.24 | 0.24 | unchanged |
| Robin perimeter | 3.56 | 3.96 | +11.24% |
| Q at n = 400 | 0.536631373 | 0.567037493 | +5.67% |
| eta at n = 400 | 15.0739% | 14.3191% | -0.7548 percentage points |

This is a valid increase in total transfer under the cap of 4. It does not show improved membrane utilization or an advantage at equal membrane length. The exact relative eta decrease is 5.01%. Values combine the saved refined fluxes with the recorded perimeters; no new solve was performed. Source endpoint: `test_rectangle_242001_252001_rl_230002` in experiment `20261019T133653820304Z_pioneer_rl_2382a80e`.

5. Grid refinement of a raster shape is not automatically validation of a smooth physical shape. Exposed Cartesian faces define an orientation-dependent staircase perimeter. Subdividing the same mask can converge to the staircase-domain solution while retaining a different boundary measure from the intended smooth wall. Conservation computed with the same conductances cannot detect this modeling discrepancy. The exact linear slab is also insufficient to measure convergence order: it can be exact to roundoff.

   The selected winner exceeds the next recorded refined endpoint by about 0.001401 in Q, or 0.25%. Their respective n = 200 to 400 changes are about 0.000733 and 0.000746. These changes are not certified error bounds, but they show why a unique continuum ranking should not be inferred from the stored decimals. Operational morphological feature checks also need a precise definition; they are not a general proof of minimum neck width.

6. The physiological interpretation is presently too strong for the model. We omit breathing, advection, evolving blood oxygenation, hemoglobin binding, perfusion and mechanical feasibility. A prescribed blood-side concentration is an explicit idealized sink. Swan and Tawhai's coupled model predicts much smaller healthy-acinar oxygen heterogeneity when blood oxygenation dynamics are included. This is a reason to delimit our interpretation, not to declare either model universally wrong. [Original physiological modeling study](https://pmc.ncbi.nlm.nih.gov/articles/PMC3043789/).

7. The high-Bi narrative must follow the data. Our baseline Bi = 100 flux spread is approximately 0.0064%, whereas the final grid change is approximately 2.1%. Shared inlet and source-corner effects dominate those cases. These results do not establish that geometry sensitivity increases monotonically with Bi. Most optimizer comparisons are at Bi = 1, so cross-regime optimization performance is not known.

8. Thousands of solves do not create thousands of independent experimental cases. The duration study's 60 paired outcomes per checkpoint reuse six geometries, two rollout seeds and five trained replicas with shared controls. Old validation and test cases have now informed development. A fresh evaluation cohort is necessary for any final generalization claim. This follows the distinction between exploratory and confirmatory evaluation emphasized in [Empirical Design in Reinforcement Learning](https://jmlr.org/papers/v25/23-0183.html).

9. Equal PDE budgets do not mean equal computational budgets. Refill and screening consume raw proposals and geometry checks; training is an additional cost. Timing after a crash and resume is unsuitable for an unqualified causal speedup claim. The 96-episode study recorded all 480 training episodes and all 252 evaluations, but one training episode exhausted its raw-draw cap, leaving the aggregate status marked incomplete. That exception must remain visible.

10. Our current learner is a 24-feature linear Expected Sarsa candidate selector with a fixed flux prior. It is neither a PINN nor a deep geometry generator. True PDE evaluation and greedy acceptance remain essential. Whether delayed credit assignment helps more than a contextual bandit or supervised ranker has not been established. The retained unconditional exploration component also accounts for substantial invalid selections; additional training alone does not remove that design choice.

**Evidence we can and cannot claim**

| Proposed statement | Present assessment |
|---|---|
| The finite-volume implementation passes its recorded analytical and conservation checks | Supported within the documented discretization and test scope |
| Constrained paired disk edits can improve Q from particular starts | Supported for recorded raster cases |
| Removing low-flux boundary is a universally beneficial rule | Contradicted by recorded counterexamples |
| Source-amplitude scaling changes normalized concentration or geometry rankings | Contradicted under this homogeneous linear model; this is a sanity check, not new physics |
| Feasible pools increase the availability of meaningful choices | Supported in the measured experiment; their extra work must be counted |
| The current learned selector reliably outperforms the fixed flux selector | Not established |
| All RL methods are inferior for diffusion shape optimization | Not tested |
| The selected endpoint maximizes membrane efficiency | Unsupported; eta decreased from its own start |
| A global or smooth-domain optimum has been found | Unsupported |
| The optimized shape describes an optimal biological acinus | Unsupported |

At 96 episodes, the duration study gives 21 wins and 39 losses versus flux, with mean delta Q = -0.00265121 and median = -0.002817196. At 12, 24 and 48 episodes the corresponding win counts are 22, 21 and 18. These are descriptive development results with shared cases, not independent binomial trials. The earlier 24-episode test also failed to establish a dependable advantage over flux, despite producing the best single stored endpoint.

**The paper to aim for**

Working title: *Constraint-aware shape optimization for Robin diffusion: a reproducible comparison of physics-guided and learned proposal selection*.

Primary question: Under explicit geometric constraints and computational budgets, when does learned selection of admissible shape edits improve total diffusive transfer beyond a fixed physics heuristic?

Possible contributions are a transparent benchmark, a controlled analysis of proposal feasibility, and an explanation of the measured limits of the learned selector. A negative result becomes useful when its scope, mechanism and competitive controls are clear. Failure of one small learner alone is insufficient for a broad claim about RL.

Novelty remains a literature-review task. Screening and the diffusivity/permeability length scale are established in [Sapoval, Filoche and Weibel, 2002](https://doi.org/10.1073/pnas.122352499). The supplied [Fujita and Koshiyama, 2025 paper](https://doi.org/10.1080/15502287.2025.2463673) already develops polyhedral acinar structure generation with an optimization criterion and morphometric comparisons; its criterion should not be equated to our Robin-flux objective. Build a comparison matrix covering absorbing diffusion, constrained shape optimization, adjoint methods, gradient-free search, RL topology optimization and acinar modeling. Do not make a first-of-its-kind claim without that review.

**Required work before a full research submission, in priority order**

1. Independently check the numerical geometry model. Compare an independent body-fitted FEM or cut-cell implementation against the same physical boundaries on a rectangle, a curved chamber and representative optimized shapes. Separate the PDE error on a fixed polygon from the error of approximating a smooth boundary. Test orientation sensitivity with geometry and boundary conditions transformed consistently. State precisely how feature width is enforced. If the paper remains strictly a raster benchmark, limit its claims accordingly and quantify grid dependence.

2. Separate membrane amount from accessibility. Add a controlled membrane-matched comparison or Q-versus-perimeter Pareto analysis under the declared area constraints. Report Q, eta, area, actual perimeter and validity together. Evaluate a limited, frozen Bi sweep rather than imply that a Bi = 1 policy generalizes. Check the role of the protected inlet and where reactive wall begins before interpreting distal screening biologically.

3. Diagnose selection information before enlarging RL. On a small fixed training/development dataset, evaluate every admissible candidate in selected pools and measure the distribution of true delta Q. Count every solve. Test whether existing features can rank useful moves on unseen geometries. Compare a simple ranker or bandit if the claim requires RL. An oracle ranking is a diagnostic upper reference, not a free optimizer baseline.

4. Freeze the final protocol before generating fresh outcomes. Hold out geometry families and generation seeds, not merely unique masks from familiar families or starts chosen for how a favored method performs. A pilot of 12-20 starts across at least four families could estimate between-geometry variability; this is a planning example, not a justified final sample size. Choose replication and sample size from that variability and the smallest meaningful improvement.

5. Compare uniform, flux-guided and learned selection with at least one credible conventional search method appropriate to the same representation and resources. Declare the primary outcome, numerical resolution, stopping rule and analysis in advance. Report both matched-PDE and matched-time performance, including training and failed proposal costs. Estimate uncertainty at the geometry and training-seed levels; do not treat shared controls or duplicate endpoints as independent evidence. Retain truncated runs and account for interrupted work.

6. Complete the publication artifacts. Audit the new duration-study outputs, archive the exact sources, record an honest initial Git history and tagged release, choose a license, add a reproducible environment and CI, and provide one command to regenerate paper tables and figures. At this assessment there is no HEAD commit and no top-level license or citation file. Existing source hashes preserve useful provenance but are not a substitute for a public release. Do not invent retrospective commit history.

A physiology-focused paper would additionally require justified physical parameter ranges and validation of the blood-side and airway assumptions. The scoped methods paper does not require rushing into 3D or a PINN.

**Manuscript and figures**

Write the introduction around the constrained transport question and the gap in existing comparisons. Follow with the mathematical model and admissible set, solver verification and geometry limitations, candidate-generation and learning methods, frozen evaluation design, results, and discussion. Distinguish exploratory development from fresh final evaluation throughout. Explain the negative RL result at the level actually tested.

The main figures should show: boundary conditions and constraints; analytical and independent numerical validation; Q and eta versus Bi and membrane resources; choice availability and computational cost; paired performance and uncertainty versus the flux baseline; and representative geometry/flux maps with common, explicitly labeled color normalization. The post-hoc winner belongs in an illustrative panel with its selection rule stated. Full seeds, controls, per-case values and exclusions belong in the supplement and public data.

**Publication route**

PLOS ONE is a plausible candidate after the validation and novelty gaps are addressed. Its criteria explicitly allow negative and null findings while requiring original research, rigorous methods and justified conclusions. This is a possible fit, not an acceptance prediction. [Official publication criteria](https://journals.plos.org/plosone/s/criteria-for-publication).

JOSS is not an immediate route for this local prototype. Its current criteria require more than six months of active public development, research use and sustained open-source practices. It publishes software-focused papers rather than a substitute for the scientific results article. [Official JOSS submission requirements](https://joss.readthedocs.io/en/latest/submitting.html).

Before submission, obtain substantive human review of the mathematics, numerical comparisons and, if retained, physiological interpretation. Agree authorship based on actual contributions. Release code and the data behind every reported result in a persistent archive; PLOS requires the data necessary to replicate the findings and a Data Availability Statement. [Official data policy](https://journals.plos.org/plosone/s/data-availability).

AI assistance in this project must be described accurately, with human authors validating the outputs and owning the scientific decisions. PLOS requires disclosure of research/code assistance and does not allow AI authorship. Do not imply that review has happened when it has not. [Official AI policy](https://journals.plos.org/plosone/s/ethical-publishing-practice).

Then draft the full manuscript, obtain author approval, optionally post a preprint, and submit to one selected journal. PLOS permits preprints but prohibits simultaneous journal submission. Submission, public release and correspondence remain separate actions requiring the authors' decisions; no publication action was performed by this assessment.

**Local evidence reviewed**

- Baseline study: experiment `20261018T190141820531Z_baseline_shapes_11d19deb`.
- Low-flux deletion study: `20261018T232110846668Z_deletion_hypothesis_2e7be463`.
- Feasible-pool comparison: `20261019T122613613156Z_feasible_pool_b8b49d2b`.
- Audited 24-episode RL study: `20261019T133653820304Z_pioneer_rl_2382a80e`.
- Update/decision replay diagnostic: `20261020T161304514936Z_learning_diagnostic_ef1e29c9`.
- Training-duration study: `20261020T162617162903Z_learning_duration_789b687b`.
- Best-endpoint review: `20261020T223106280735Z_best_geometry_review_e827c658`.

At the time of the original assessment, the independent solver, matched-membrane
experiment, fresh final cohort and completed manuscript were still proposed work.

**Update, 21 October 2026**

The [new study](morphology_results.md) now implements an independently assembled
triangular FEM, physical-polygon controls, nine literature-inspired families,
45 distinct development starts, long trajectories and exact within-start
membrane controls. Complementary energy estimates, a complete solve/update
audit and source-wall sensitivity are implemented and tested. The combined
306-case local regression passes. The main production run, supplemental review and 90 additional paid classical
controls are complete, accounting for 50,721 solver calls under the separately
recorded protocols. The post-run curve-index correction preserves original
figures and leaves numerical results unchanged. A
[manuscript draft](manuscript_draft.md), [related-work assessment](related_work_assessment.md)
and [claim review](claim_evidence_review.md) now exist.

These improvements do not create a fresh confirmatory cohort, separate geometry
and search variance, or training replication for the new model. They also do
not supply a public release, license, authorship agreement or biological
validation. The paper is being scoped to its finite computational benchmark;
the PDE/objective and Robin energy shape optimization have close established
prior work. Record completed results in the linked evidence files rather than
treating this historical assessment as the final run status.
