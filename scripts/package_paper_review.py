"""Assemble a local human-review packet; do not publish or rerun experiments."""

import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def run(paper, gate):
    repository = Path(__file__).resolve().parents[1]
    paper, gate = paper.resolve(), gate.resolve()
    suites = ET.parse(gate).getroot().iter("testsuite")
    counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.get(key, 0))
    if counts["tests"] < 310 or counts["failures"] or counts["errors"] or counts["skipped"]:
        raise RuntimeError("The complete corrected 310-case gate must pass before packaging")
    summary = json.loads((paper / "summary.json").read_text())
    if (
        summary["total_committed_solver_calls"] != 50721
        or summary["unique_returned_endpoints"] != 285
    ):
        raise ValueError("Unexpected frozen evidence scope")
    manuscript = (repository / "docs/manuscript_draft.md").read_text(encoding="utf-8")
    manuscript = manuscript.replace("../experiments/paper_review_20261021/", "")
    for image_path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", manuscript):
        if not (paper / image_path).is_file():
            raise FileNotFoundError(image_path)
    (paper / "manuscript.md").write_text(manuscript, encoding="utf-8")
    supplement = paper / "supplement"
    supplement.mkdir(exist_ok=True)
    for name in (
        "manuscript_supplement.md",
        "derivations.md",
        "numerical_validation.md",
        "complementary_flux_bounds.md",
        "source_corner_limit.md",
        "collar_ablation.md",
        "permeability_response.md",
        "morphology_protocol.md",
        "morphology_limits.md",
        "classical_control_protocol.md",
        "reproducibility.md",
        "morphology_execution.md",
        "claim_evidence_review.md",
        "related_work_assessment.md",
        "novelty_review_20261021.md",
        "scientific_review_brief.md",
        "ai_assistance_disclosure.md",
        "morphology_literature.md",
        "paper_readiness_plan.md",
        "screening_distribution.md",
    ):
        source = repository / "docs" / name
        text = source.read_text(encoding="utf-8").replace("../experiments/", "../../")
        (supplement / name).write_text(text, encoding="utf-8")
    figure_assets = supplement / "figures"
    figure_assets.mkdir(exist_ok=True)
    for name in ("slab_errors.png", "rectangle_fields.png", "rectangle_refinement.png"):
        shutil.copyfile(repository / "docs/figures" / name, figure_assets / name)
    workflow = paper / ".github/workflows"
    workflow.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repository / ".github/workflows/numerics.yml", workflow / "numerics.yml")
    software = paper / "software"
    software.mkdir(exist_ok=True)
    main = repository / "experiments/20261021T122626095890Z_morphology_98e4aafa"
    classical = repository / "experiments/20261021T141950978652Z_classical_72101312"
    shutil.copytree(main / "source_snapshot", software / "executed_core", dirs_exist_ok=True)
    shutil.copytree(
        classical / "control_sources", software / "executed_controls", dirs_exist_ok=True
    )
    core = software / "executed_core"
    (core / "docs").mkdir(exist_ok=True)
    (core / "configs").mkdir(exist_ok=True)
    shutil.copyfile(main / "protocol.md", core / "docs/morphology_protocol.md")
    shutil.copyfile(main / "config.yaml", core / "configs/morphology_trajectories.yaml")
    shutil.copyfile(main / "requirements-validation.txt", core / "requirements-validation.txt")
    shutil.copyfile(main / "source_manifest.json", core / "source_manifest.json")
    (core / "README.md").write_text(
        "# Executed numerical sources\n\n"
        "The 92 files identified by source_manifest.json are unchanged executed sources. "
        "This supporting README and the copied configuration, protocol and optional "
        "requirements were added to make the archive inspectable/installable; they are "
        "not retrospective additions to the executed source manifest. The original plot "
        "indexing issue is retained here for historical identity. The corrected analysis "
        "and source diffs are supplied separately. Full data live in the original runs.\n",
        encoding="utf-8",
    )
    analysis = software / "analysis_current"
    analysis.mkdir(exist_ok=True)
    for name in (
        "paper_tables.py",
        "paper_figures.py",
        "response_diagnostics.py",
        "permeability_checks.py",
        "archive_curve_correction.py",
        "package_paper_review.py",
        "build_release.py",
        "verify_release.py",
        "winner_mechanisms.py",
        "collar_ablation.py",
        "postprocess_morphology_bounds.py",
        "flux_energy_bounds.py",
    ):
        shutil.copyfile(repository / "scripts" / name, analysis / name)
    shutil.copyfile(repository / "src/oxygen_shape/viz/history.py", analysis / "history.py")
    evidence_paths = repository / "src/oxygen_shape/experiments/evidence_paths.py"
    shutil.copyfile(evidence_paths, analysis / "evidence_paths.py")
    (software / "README.md").write_text(
        "# Source versions in this local packet\n\n"
        "executed_core mirrors the source snapshot used for numerical production. "
        "Its supporting configuration/protocol/requirements are copies from that run. "
        "The original 92 source hashes remain unchanged. executed_controls contains the "
        "extension's seven pinned source files; their original relative paths are in "
        "provenance/classical_control_source_manifest.json. Restore scripts under scripts/ "
        "and the protocol under docs/ when constructing a clean historical checkout.\n\n"
        "analysis_current supplies the post-run table/figure scripts and the corrected "
        "history converter. It is a distinct analysis version, not the executed optimizer. "
        "The converter belongs at src/oxygen_shape/viz/history.py in a current analysis "
        "checkout. See post_run_curve_fix for the exact before/after changes. The local "
        "repository commands in the packet README are tested against the existing data; "
        "this lightweight source packet does not embed all journals or claim a clean-room "
        "replay on another machine. Environment versions and original input hashes are retained.\n",
        encoding="utf-8",
    )
    provenance = paper / "provenance"
    provenance.mkdir(exist_ok=True)
    shutil.copyfile(gate, provenance / "corrected_tests.xml")
    for root, label, names in (
        (
            main,
            "main",
            [
                "config.yaml",
                "summary.json",
                "source_manifest.json",
                "environment.json",
                "requirements-environment.txt",
                "bank_manifest.json",
            ],
        ),
        (
            classical,
            "classical",
            ["summary.json", "control_source_manifest.json", "extension.json"],
        ),
    ):
        for name in names:
            shutil.copyfile(root / name, provenance / (label + "_" + name))
    for relative, destination in (
        ("postprocess/journal_audit/summary.json", "main_journal_audit.json"),
        ("postprocess/collar_ablation/summary.json", "collar_summary.json"),
        ("postprocess/source_corner/summary.json", "corner_summary.json"),
    ):
        shutil.copyfile(main / relative, provenance / destination)
    (paper / "README.md").write_text(
        "# Local research-paper review packet\n\n"
        "The computational study is complete. Start with the [manuscript](manuscript.md), "
        "[technical supplement](supplement/manuscript_supplement.md), "
        "[evidence tables](report.md) and [claim review](supplement/claim_evidence_review.md).\n\n"
        f"The packet records 285 returned endpoints and 50,721 committed solver calls. "
        f"The corrected full regression passes {counts['tests']} tests. All search budgets "
        "complete. Practical energy estimates support 284 initial/final improvements and "
        "leave two leading masks unresolved. The tested learned ranker does not improve "
        "on flux guidance on descriptive average.\n\n"
        "The leading observed endpoint has Q400=0.579831643 and fine-FEM Q=0.581930801, "
        "area 0.24 and membrane length 7.82. Its 27.55% FEM gain includes 30.33% additional "
        "membrane and proximal redistribution. It is not a globally optimal or validated "
        "natural acinus. [See its fields and redistribution](figures/selected_geometry.png).\n\n"
        "PNG previews and vector PDFs are in figures/. Curve summaries retain every actual "
        "charged solve, including the initial and final values. The "
        "[post-run correction](post_run_curve_fix/README.md) preserves original plots and "
        "documents the pandas indexing fix; numerical results are unchanged.\n\n"
        "Full journals and field caches remain in the sibling original "
        "[main study](../20261021T122626095890Z_morphology_98e4aafa/report.md) and "
        "[classical study](../20261021T141950978652Z_classical_72101312/report.md). "
        "input_manifest.json pins the reviewed fields; they are not duplicated in this "
        "lightweight packet. Executed sources and current analysis scripts are under "
        "software/. Exact historical replay uses archived executed sources, while current "
        "reporting includes the documented rendering correction.\n\n"
        "From the repository root, regenerate the paper tables and figures without "
        "PDE solves using:\n\n```text\n"
        "python scripts/paper_tables.py experiments/20261021T122626095890Z_morphology_98e4aafa "
        "experiments/20261021T141950978652Z_classical_72101312 experiments/paper_review_20261021\n"
        "python -m scripts.paper_figures experiments/paper_review_20261021 "
        "experiments/20261021T122626095890Z_morphology_98e4aafa\n```\n\n"
        "Author: Nabil ES-SAADI, independent researcher. Code is MIT; original data/figures "
        "and research text are CC BY 4.0. The versioned release is "
        "https://github.com/nabil-stem/oxygen-shape-optimization/releases/tag/v0.1.0. "
        "Full data archives supplement this packet. The targeted novelty review and reviewer "
        "brief are in supplement/. No archival DOI, independent human review or journal "
        "submission is implied.\n",
        encoding="utf-8",
    )
    manifest = {
        file.relative_to(paper).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in paper.rglob("*")
        if file.is_file() and file.name != "packet_manifest.json"
    }
    (paper / "packet_manifest.json").write_text(
        json.dumps(
            dict(regression=counts, scope="local human review, not public release", files=manifest),
            indent=2,
        )
        + "\n"
    )
    print(paper / "README.md")
    return paper


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path)
    parser.add_argument("gate", type=Path)
    args = parser.parse_args()
    run(args.paper, args.gate)
