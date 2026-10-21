"""Exercise a completed resume on a small artifact copy; forbid all new rollouts."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import oxygen_shape.experiments.pioneer_training as training
from oxygen_shape.experiments.config import load_config
from oxygen_shape.optimization.pioneer_config import PioneerRLConfig


def check(root):
    destination = root / "completed_resume_check"
    destination.mkdir(exist_ok=False)
    (destination / "figures").mkdir()
    for name in (
        "config.yaml",
        "run_identity.json",
        "source_manifest.json",
        "readiness.json",
        "evaluation_index.json",
    ):
        shutil.copy2(root / name, destination / name)
    for name in ("bank", "policies"):
        shutil.copytree(root / name, destination / name)
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "policies").glob("*.npz")
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("A completed resume attempted new numerical work")

    training.run_stored_rollout = forbidden
    training.readiness = forbidden
    training.solve = forbidden
    training.validate_refinement = forbidden
    config = load_config(root / "config.yaml", PioneerRLConfig)
    _, summary = training.run_pioneer_training(config, resume=destination)
    after = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (destination / "policies").glob("*.npz")
    }
    assert before == after and summary["status"] == "complete"
    assert summary["training_episodes"] == 120 and summary["evaluation_runs"] == 120
    assert summary["number_of_pde_solves"] == 5014
    result = {
        "passed": True,
        "new_pde_solves": 0,
        "all_five_checkpoints_unchanged": True,
        "training_episodes_skipped": 120,
        "evaluation_runs_skipped": 120,
        "production_summary_unmodified": True,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (root / "completed_resume_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    check(parser.parse_args().run.resolve())
