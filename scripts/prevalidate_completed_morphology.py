"""Prefill prescribed validation caches for a fixed prefix of finished searches."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from oxygen_shape.experiments.config import load_config
from oxygen_shape.experiments.morphology_study import MorphologyStudyConfig, _verify_sources
from oxygen_shape.experiments.morphology_validation import validate_endpoint
from oxygen_shape.experiments.reporting import write_csv


def run(root, limit, offset=0):
    if limit < 1 or offset < 0:
        raise ValueError("Positive count and nonnegative offset required")
    _verify_sources(root)
    config = load_config(root / "config.yaml", MorphologyStudyConfig)
    jobs = json.loads((root / "jobs.json").read_text())[offset : offset + limit]
    if any(not (root / job["path"] / "summary.json").exists() for job in jobs):
        raise RuntimeError("Every job in the requested fixed prefix must already be complete")
    if (root / "search_metrics.csv").exists():
        raise RuntimeError("Main validation is already starting; do not race its caches")
    output = root / "postprocess/prefill_validation"
    if offset:
        output = output / f"range_{offset}_{offset + limit}"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    state = {
        "completed": 0,
        "requested": limit,
        "status": "running",
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    for index, job in enumerate(jobs):
        # The fixed prefix is substantially shorter than the unfinished matrix.
        # Exit if the main driver approaches validation; it owns the final report.
        unfinished = sum(
            not (root / j["path"] / "summary.json").exists()
            for j in json.loads((root / "jobs.json").read_text())
        )
        if unfinished < 20 or (root / "search_metrics.csv").exists():
            break
        rows.append(validate_endpoint(root, config, job))
        write_csv(output / "endpoint_metrics.csv", rows)
        state = {
            "completed": len(rows),
            "requested": limit,
            "status": "running",
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        (output / "status.json").write_text(json.dumps(state, indent=2) + "\n")
        print(f"Prescribed endpoint checks: {index + 1}/{limit}", flush=True)
    state.update(status="complete", utc=datetime.now(timezone.utc).isoformat())
    (output / "status.json").write_text(json.dumps(state, indent=2) + "\n")
    (output / "execution_note.json").write_text(
        json.dumps(
            {
                "scheduling": "One extra single-thread process while two search workers continue",
                "scope": (
                    "Identical predeclared checks; main driver reuses identity-verified caches"
                ),
                "search_algorithm_changes": False,
                "additional_checks_beyond_protocol": 0,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "prefix_requested": limit,
                "prefix_completed": len(rows),
                "job_offset": offset,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, default=90)
    parser.add_argument("--offset", type=int, default=0)
    arguments = parser.parse_args()
    run(arguments.root, arguments.limit, arguments.offset)
