# Local research-paper review packet

The computational study is complete. Start with the [manuscript](manuscript.md), [technical supplement](supplement/manuscript_supplement.md), [evidence tables](report.md) and [claim review](supplement/claim_evidence_review.md).

The packet records 285 returned endpoints and 50,721 committed solver calls. The corrected full regression passes 324 tests. All search budgets complete. Practical energy estimates support 284 initial/final improvements and leave two leading masks unresolved. The tested learned ranker does not improve on flux guidance on descriptive average.

The leading observed endpoint has Q400=0.579831643 and fine-FEM Q=0.581930801, area 0.24 and membrane length 7.82. Its 27.55% FEM gain includes 30.33% additional membrane and proximal redistribution. It is not a globally optimal or validated natural acinus. [See its fields and redistribution](figures/selected_geometry.png).

PNG previews and vector PDFs are in figures/. Curve summaries retain every actual charged solve, including the initial and final values. The [post-run correction](post_run_curve_fix/README.md) preserves original plots and documents the pandas indexing fix; numerical results are unchanged.

Full journals and field caches remain in the sibling original [main study](../20261021T122626095890Z_morphology_98e4aafa/report.md) and [classical study](../20261021T141950978652Z_classical_72101312/report.md). input_manifest.json pins the reviewed fields; they are not duplicated in this lightweight packet. Executed sources and current analysis scripts are under software/. Exact historical replay uses archived executed sources, while current reporting includes the documented rendering correction.

From the repository root, regenerate the paper tables and figures without PDE solves using:

```text
python scripts/paper_tables.py experiments/20261021T122626095890Z_morphology_98e4aafa experiments/20261021T141950978652Z_classical_72101312 experiments/paper_review_20261021
python -m scripts.paper_figures experiments/paper_review_20261021 experiments/20261021T122626095890Z_morphology_98e4aafa
```

Author: Nabil ES-SAADI, independent researcher. Code is MIT; original data/figures and research text are CC BY 4.0. The versioned release is https://github.com/nabil-stem/oxygen-shape-optimization/releases/tag/v0.1.0. Full data archives supplement this packet. The targeted novelty review and reviewer brief are in supplement/. No archival DOI, independent human review or journal submission is implied.

Sibling study links resolve after extracting all release data assets into the documented common folder. The GitHub source view contains the review packet, while full raw studies are release assets.
