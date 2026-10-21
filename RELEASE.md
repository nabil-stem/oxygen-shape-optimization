# Research release v0.1.0

Author: **Nabil ES-SAADI — independent researcher**.

This is a software/data release with an **unreviewed working manuscript**.
It is not a journal publication, external peer-review outcome or biological
validation. Code is MIT; original data/figures/research text are CC BY 4.0.
Optional Triangle has separate terms; see THIRD_PARTY.md.

## Contents

- Current Python package, configs, tests and analysis scripts.
- All primary morphology and classical-control study data: 285 returned
  endpoints, 50,721 committed production solver calls, evaluated rejected
  edits, training/search journals and independent numerical checks.
- Executed source snapshots and original manifests, plus the documented
  post-run curve-index correction.
- Manuscript, vector figures, claim matrix, targeted novelty review and a
  scientific-review invitation draft. No invitation has been sent.

The tested response ranker has no overall advantage over flux guidance.
The highest observed returned Q400 is 0.579831643; two leading masks remain
unresolved by practical energy ranges. More membrane and proximal relocation
contribute to the selected high-flux design. No global or natural optimum is
claimed.

## Reproduce from downloaded assets

Download every numbered ZIP plus release_manifest.json. Copy
scripts/verify_release.py from this source release into the download folder if
you have not cloned the repository. With Python >=3.10:

```text
python verify_release.py DOWNLOAD_FOLDER --extract NEW_EMPTY_PARENT
```

All parts expand under `NEW_EMPTY_PARENT/oxygen_shape_release/`. From that folder:

```text
python -m pip install -e ".[dev]" -r requirements-validation.txt
python -m pytest -q
python scripts/paper_tables.py experiments/20261021T122626095890Z_morphology_98e4aafa experiments/20261021T141950978652Z_classical_72101312 experiments/regenerated_paper
python -m scripts.paper_figures experiments/regenerated_paper experiments/20261021T122626095890Z_morphology_98e4aafa
```

Table/figure regeneration consumes checked saved fields and spends no PDE
solves. The test suite performs its own small numerical checks. Exact historical
identity-guarded continuation uses archived executed sources in a separate
compatible environment; it does not use the corrected current reporter.
Cross-platform bitwise replay is not promised.

Read DATA_AVAILABILITY.md, docs/novelty_review_20261021.md and
docs/scientific_review_brief.md before using the manuscript scientifically.
No DOI, reviewer endorsement, author funding declaration or journal acceptance
has been fabricated.
