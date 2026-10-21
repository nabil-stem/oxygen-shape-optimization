"""Keep all audited research policies with identities; do not promote a winner."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from oxygen_shape.optimization.candidate_policy import CandidatePolicy


def publish(root):
    if root.name != "20261019T133653820304Z_pioneer_rl_2382a80e":
        raise ValueError("This publication helper is pinned to the completed 2382a80e study")
    repository = Path(__file__).resolve().parents[1]
    audit = json.loads((root / "artifact_audit.json").read_text())
    assert audit["passed"]
    assert json.loads((root / "completed_resume_verification.json").read_text())["passed"]
    identity = json.loads((root / "run_identity.json").read_text())["identity"]
    destination = repository / "models" / "refill_rl_2382a80e"
    destination.mkdir(exist_ok=False)
    policies = []
    for source in sorted((root / "policies").glob("*.npz")):
        policy, records = CandidatePolicy.load(source, identity)
        assert len(records) == 24
        shutil.copy2(source, destination / source.name)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256((destination / source.name).read_bytes()).hexdigest() == digest
        policies.append(
            {
                "file": source.name,
                "sha256": digest,
                "updates": policy.updates,
                "policy_digest": policy.digest(),
            }
        )
    for name in (
        "config.yaml",
        "run_identity.json",
        "source_manifest.json",
        "learning_diagnostics.json",
        "artifact_audit.json",
        "completed_resume_verification.json",
        "implementation_verification.json",
    ):
        shutil.copy2(root / name, destination / name)
    (destination / "metadata.json").write_text(
        json.dumps(
            {
                "status": "research_only",
                "identity": identity,
                "run_directory": str(root),
                "algorithm": "linear Expected Sarsa",
                "policies": policies,
                "selection": "all five final replicas; no promotion",
                "test_comparison_vs_flux": "17 wins, 0 ties, 23 losses; shared controls",
                "training_episodes": 120,
                "training_updates": 2573,
                "new_experiment_pde_solves": 5014,
            },
            indent=2,
        )
        + "\n"
    )
    (destination / "README.md").write_text(
        "# Audited feasible-pool RL research models\n\n"
        "All five final replicas are retained. Each uses 24 incumbent-only features, "
        "a fixed flux prior and learned additive residual weights. These are CPU "
        "linear Expected Sarsa policies, not neural networks. Checkpoints include "
        "the 24-episode history and the frozen config/bank/source identity.\n\n"
        "Load with `CandidatePolicy.load(checkpoint_path, metadata['identity'])` "
        "using the `identity` in metadata.json. `PioneerRLConfig` in config.yaml "
        "and `run_candidate_rollout` define their observation/action protocol. "
        "The old tabular `disk_q_policy.npz` is a different model format.\n\n"
        "The test comparison beats raw uniform in 29/40 shared comparisons but "
        "flux ranking in only 17/40. No policy is promoted or selected by test "
        "performance. The original experiment and source snapshot are identified "
        "in metadata.json. See docs/refill_rl_results.md for costs and limits.\n",
        encoding="utf-8",
    )
    helpers = root / "supplemental_scripts"
    helpers.mkdir(exist_ok=True)
    manifest = {}
    for name in (
        "prepare_refill_rl_config.py",
        "audit_refill_rl.py",
        "audit_feasible_pools.py",
        "check_completed_rl_resume.py",
        "summarize_refill_rl.py",
        "publish_refill_rl.py",
    ):
        source = repository / "scripts" / name
        shutil.copy2(source, helpers / name)
        manifest[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (helpers / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ("training.csv", "evaluations.csv", "paired_controls.csv"):
        shutil.copy2(root / name, repository / "docs/data" / f"refill_rl_{name}")
    print(f"Retained {len(policies)} audited models in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    publish(parser.parse_args().run.resolve())
