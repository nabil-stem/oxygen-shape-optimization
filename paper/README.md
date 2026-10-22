# Paper figures and tables

Start with the [manuscript](../docs/manuscript_draft.md),
[technical supplement](../docs/manuscript_supplement.md) and
[evidence tables](tables/report.md). The manuscript is unreviewed.

- `figures/` contains the ten main figures as PNG previews and vector PDFs.
- `tables/` contains every returned endpoint and the reported comparisons,
  resource/validation tables and curve summaries.
- `public_view_manifest.json` records each source hash, display-copy hash and
  path-only transformation.

The [complete v0.1.0 evidence](https://github.com/nabil-stem/oxygen-shape-optimization/releases/tag/v0.1.0)
includes all primary journals, checked fields, executed code versions and original
manifests. This public view omits duplicated code, scratch logs and machine-specific
metadata. CSV copies drop the redundant `resolved_path` column; summary
copies omit the same location field. Text line endings are normalized to LF. Numerical values and figure bytes are
unchanged. [Reproduction instructions](../docs/reproducibility.md) identify the
full archive and explain regeneration.

The benchmark includes 285 returned endpoints. The tested response ranker has
no aggregate advantage over flux guidance. Two leading masks have overlapping
practical flux ranges; the selected high-Q design is not a global or anatomical
optimum. See the [claim review](../docs/claim_evidence_review.md).
