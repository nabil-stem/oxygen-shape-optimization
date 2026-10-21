"""Outcome-blind bank construction: validity and mask identity only, never Q."""

import hashlib
import json

from oxygen_shape.geometry.research_families import make_research_geometry
from oxygen_shape.optimization.guided_search import validate_candidate
from oxygen_shape.optimization.trajectory_search import _json


def prepare_bank(root, config):
    path = root / "bank_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        for row in manifest["training"] + manifest["evaluation"]:
            g = make_research_geometry(
                row["family"], config.n, config.target_area, seed=row["seed"]
            ).geometry
            if hashlib.sha256(g.inside.tobytes()).hexdigest() != row["mask_sha256"]:
                raise ValueError("Stored bank mask does not reproduce")
        return {
            family: [r["seed"] for r in manifest["evaluation"] if r["family"] == family]
            for family in config.families
        }
    excluded, training, evaluation, rejections = set(), [], [], []
    cfg = config.trajectory()
    for family in config.families:
        g = make_research_geometry(family, config.n, config.target_area, seed=0).geometry
        excluded.add(hashlib.sha256(g.inside.tobytes()).hexdigest())
    for family in config.training_families:
        g = make_research_geometry(
            family, config.n, config.target_area, seed=config.training_bank_seed
        ).geometry
        reason = validate_candidate(g, g, cfg)
        if reason:
            raise ValueError(f"Invalid declared training case {family}: {reason}")
        digest = hashlib.sha256(g.inside.tobytes()).hexdigest()
        training.append(dict(family=family, seed=config.training_bank_seed, mask_sha256=digest))
        excluded.add(digest)
    seeds = {}
    for family in config.families:
        seeds[family] = []
        for index in range(len(config.evaluation_seeds)):
            for attempt in range(100):
                seed = config.evaluation_bank_seed + index + 1000 * attempt
                g = make_research_geometry(family, config.n, config.target_area, seed=seed).geometry
                reason = validate_candidate(g, g, cfg)
                digest = hashlib.sha256(g.inside.tobytes()).hexdigest()
                if reason or digest in excluded:
                    rejections.append(
                        dict(
                            family=family,
                            index=index,
                            seed=seed,
                            reason=reason or "Duplicate canonical/training/evaluation mask",
                        )
                    )
                    continue
                seeds[family].append(seed)
                excluded.add(digest)
                evaluation.append(dict(family=family, index=index, seed=seed, mask_sha256=digest))
                break
            else:
                raise ValueError(f"Cannot construct five distinct valid variants of {family}")
    _json(
        path,
        {
            "training": training,
            "evaluation": evaluation,
            "rejections": rejections,
            "selection_rule": "Geometry validity and identity only; no PDE outcome is read",
            "evaluation_distinct_masks": len({r["mask_sha256"] for r in evaluation}),
            "training_evaluation_mask_overlap": 0,
        },
    )
    return seeds
