# Reproduce numerical checks and the morphology study

The numerical production predates the initial source commit. Source snapshots,
hashes, configuration and the exact installed environment preserve its identity;
no retrospective commit history is invented. The versioned research-release
instructions are in [RELEASE.md](../RELEASE.md). Author metadata and MIT/CC BY 4.0
license choices were supplied by Nabil ES-SAADI on 21 October 2026. No archival
DOI or independent human review is claimed.

## Numerical regression

In a clean Python >=3.10 environment:

```text
python -m pip install -e ".[dev]" -r requirements-validation.txt
python -m ruff check src tests
python -m pytest -q
```

The optional requirements file pins Triangle for independent FEM tests. These
checks include analytical diffusion, conservation, symmetry, constraints,
interruption/resume, exact learning-update replay, the small end-to-end study,
complementary energy estimates and the analytical 2D source-corner strip. They
do not rerun the full production search matrix.

The combined local gate passed 306 cases in 640.20 seconds on 21 October 2026.
Its XML is in `tmp/paper_gate_20261021/tests.xml`, with a copy and hash in launch
`20261021T140729Z_classical_61a372`. This includes paid conventional controls,
their interruption/recovery accounting and an end-to-end extension pipeline.

A local [CI workflow](../.github/workflows/numerics.yml) prepares the same gate
for Windows and Ubuntu with Python 3.10 and one numerical thread. It is not a
claim of a successful remote CI run; no repository is published or pushed here.
Its action versions are pinned to the v7 commit SHAs resolved from the official
[checkout](https://github.com/actions/checkout),
[setup-python](https://github.com/actions/setup-python) and
[upload-artifact](https://github.com/actions/upload-artifact) repositories on
21 October 2026. Test XML, rather than research experiment data, is the only
configured CI artifact.

## Exact study evidence

Run `20261021T122626095890Z_morphology_98e4aafa` retains config.yaml, the bank
manifest, all evaluated search states and selected features, model checkpoints,
source_manifest.json and source_snapshot/. environment.json records the actual
Python, package versions, platform and Git state. requirements-environment.txt
is that installed-version inventory, excluding the editable project itself.
For an exact replay use the saved source snapshot and compatible recorded
environment, rather than replacing it with newer package versions. Cross-platform
bitwise identity is not promised.

To launch the fixed protocol in this workspace:

```powershell
.\scripts\start_morphology_study.ps1
```

To resume the recorded experiment, preserve its source/config identities:

```powershell
.\scripts\start_morphology_study.ps1 -ResumeRoot "experiments\20261021T122626095890Z_morphology_98e4aafa"
```

Completed jobs and identity-verified fields are reused. Pending uncommitted work
is retained separately; do not describe a truncated or interrupted job as a
full-budget success. Two search workers and auxiliary prescribed validation
scheduling are recorded in [morphology_execution.md](morphology_execution.md).

After the main study finishes, regenerate its supplemental review with:

```powershell
$study = "experiments\20261021T122626095890Z_morphology_98e4aafa"
.\scripts\complete_morphology_review.ps1 -RunRoot $study -DriverId 0
.\.venv\Scripts\python.exe scripts\winner_mechanisms.py $study
.\.venv\Scripts\python.exe scripts\plot_morphology_catalog.py $study
```

DriverId 0 is used only for an already finished study; a running study requires
its actual live driver PID. The saved-field audit, energy estimates, mechanism
figure and catalog plots perform no PDE solves. The collar and analytical strip
checks spend their separately declared solves once, then reuse identity-verified
caches. Read their own cost summaries, not the frozen main summary, for these
additional evaluations. Script snapshots and input hashes are retained under
postprocess/ and in the launch records. Logs, incomplete cases and unfavorable
results are evidence to retain, not cleanup targets.

## Added classical controls

The [separate protocol](classical_control_protocol.md) identifies the post-hoc
development extension. It imports completed original comparison fields and pins
both their hashes and its own implementation. It adds no training. To queue it
while the original driver is still running, pass the actual driver PID:

```powershell
.\scripts\start_classical_controls.ps1 -DriverId 18064
```

For an already finished original study, omit DriverId. The launcher checks the
passing 306-case regression record and rejects duplicate live launches. Its
hidden driver waits for completed original reports before starting. Two search
workers each use one numerical thread; parent validation/audit is overlapped.
Read the returned experiment path from the launch stdout. To resume unfinished
controls, use the same source identity and its returned run directory:

```text
python -m scripts.conventional_study ORIGINAL_RUN --workers 2 --resume CLASSICAL_RUN
```

Do not edit pinned sources during either study. Completed jobs and verified
fields are reused; lost uncommitted work is bounded separately. New conventional
search and validation costs belong to its summary, with imported original
endpoints reported as reused. Remote CI execution remains unperformed.


## Post-run reporting correction and paper regeneration

After all numerical runs finished, figure QA found pandas label alignment
shifting learning-curve values by one solve. The shared gain_by_solve converter
uses positional arrays; four added tests preserve the initial zero, final gain
and actual integer budgets. Original figures and before/after reporting sources
are archived in paper_review_20261021/post_run_curve_fix. The renewed full gate
is tmp/paper_gate_corrected_20261021/tests.xml.

Current reporting sources therefore intentionally differ from the executed
historical source manifests. Exact identity-guarded historical replay/audit
uses the archived source_snapshot and control_sources in a separate compatible
checkout. Do not rewrite original manifests or silently resume under altered
sources. Only postprocessing changed after production; solver/search fields,
weights, histories, costs and endpoint tables remain unchanged.

From this repository root, the zero-PDE paper regeneration is:

```text
python scripts/paper_tables.py experiments/20261021T122626095890Z_morphology_98e4aafa experiments/20261021T141950978652Z_classical_72101312 experiments/paper_review_20261021
python -m scripts.paper_figures experiments/paper_review_20261021 experiments/20261021T122626095890Z_morphology_98e4aafa
```

The first command verifies reviewed input hashes and collects tables; the second
regenerates vector figures and the selected-geometry view from existing checked
fields. No missing field is silently replaced by a new PDE solve in that view.
The local review packet retains executed sources, current analysis code,
regression XML, a file manifest and links to the full original data directories.

Final corrected full regression: **310 passed in 511.75 seconds** on 21 October 2026.
The XML is retained in tmp/paper_gate_corrected_20261021/tests.xml and the review packet.

## Portable release analysis

The current paper collector resolves historical Windows and backslash-relative
references only inside its two explicitly supplied study roots, and still
verifies the original data hashes. This post-production portability correction
does not rewrite original manifests or change solver/search results. Added tests
reject unknown roots and path escapes. The release verifier checks archive and
individual payload hashes; its extraction target must be a new empty folder.

Production data, including all evaluated rejected candidates, are distributed
separately from Git source history. Original user-supplied papers, virtual
environments and old models are excluded. See DATA_AVAILABILITY.md for scope.
