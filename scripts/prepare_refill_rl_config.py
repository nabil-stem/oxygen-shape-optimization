"""Freeze a fresh feasible-pool RL protocol before training or held-out outcomes."""

import hashlib
from pathlib import Path

import numpy as np
import yaml

from oxygen_shape.experiments.config import load_config
from oxygen_shape.optimization.pioneer_config import PioneerRLConfig


def main():
    root = Path(__file__).resolve().parents[1]
    known = set()
    historical = [
        "20261019T031120855645Z_pioneer_rl_4ad9c2ee",
        "20261019T111653094459Z_rl_revision_694225e7",
        "20261019T122613613156Z_feasible_pool_b8b49d2b",
    ]
    for name in historical:
        directory = root / "experiments" / name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        masks = [*directory.glob("bank/*.npy"), *directory.rglob("final_geometry.npy")]
        for path in masks:
            mask = np.load(path, allow_pickle=False)
            if mask.dtype != bool or mask.shape != (100, 100):
                continue
            known.add(hashlib.sha256(mask.tobytes()).hexdigest())
    value = yaml.safe_load((root / "configs/train_rl_candidates.yaml").read_text())
    value.update(
        seed=229000,
        replica_seeds=list(range(230000, 230005)),
        validation_generation_seeds=[241000],
        test_generation_seeds=[242000, 242001],
        validation_rollout_seeds=[251000, 251001],
        test_rollout_seeds=[252000, 252001],
        episodes=24,
        candidate_screening="full",
        refill_quota=4,
        refill_draw_limit=256,
        flux_prior_scale=1,
        discount=1,
        invalid_reward=0,
        excluded_geometry_hashes=sorted(known),
    )
    destination = root / "configs/train_rl_feasible.yaml"
    if destination.exists():
        raise FileExistsError("Refusing to overwrite a frozen RL protocol")
    destination.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    config = load_config(destination, PioneerRLConfig)
    print(f"Frozen {destination}; {len(known)} known masks excluded from new held-out selection")
    print(f"{len(config.replica_seeds)} replicas x {config.episodes} episodes")


if __name__ == "__main__":
    main()
