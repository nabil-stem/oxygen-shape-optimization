"""On-policy, linear Expected Sarsa for variable geometry-candidate pools."""

import hashlib
import json
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np

from oxygen_shape.experiments.study_worker import replace_with_retry

FEATURE_NAMES = (
    "bias",
    "pde_remaining",
    "raw_remaining",
    "relative_gain",
    "perimeter_fraction",
    "eta",
    "mean_u",
    "radius",
    "reallocated_area",
    "perimeter_delta",
    "add_x",
    "add_y",
    "donor_x",
    "donor_y",
    "add_flux",
    "donor_flux",
    "flux_product",
    "cheap_pass",
    "pde_x_perimeter_delta",
    "pde_x_flux_product",
    "area_x_perimeter",
    "add_x_eta",
    "donor_x_eta",
    "radius_x_perimeter",
)
FEATURE_COUNT = len(FEATURE_NAMES)


def mixed_probabilities(
    scores: np.ndarray, eligible: np.ndarray, epsilon: float, uniform_floor: float
) -> np.ndarray:
    scores, eligible = np.asarray(scores, float), np.asarray(eligible, bool)
    if (
        scores.ndim != 1
        or not scores.size
        or eligible.shape != scores.shape
        or not np.isfinite(scores).all()
        or not 0 <= epsilon <= 1
        or not 0 < uniform_floor <= 1
    ):
        raise ValueError("Invalid policy probabilities")
    indices = np.flatnonzero(eligible)
    p = np.zeros(len(scores))
    if not len(indices):
        p[0] = 1
        return p
    greedy = indices[scores[indices] >= scores[indices].max() - 1e-12]
    p[indices] = (1 - uniform_floor) * epsilon / len(indices)
    p[greedy] += (1 - uniform_floor) * (1 - epsilon) / len(greedy)
    p[0] += uniform_floor
    return p


@dataclass
class CandidatePolicy:
    weights: np.ndarray = field(default_factory=lambda: np.zeros(FEATURE_COUNT))
    updates: int = 0
    prior_weights: np.ndarray = field(default_factory=lambda: np.zeros(FEATURE_COUNT))

    @classmethod
    def with_flux_prior(cls, scale: float = 1) -> "CandidatePolicy":
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("Flux prior scale must be finite and nonnegative")
        prior = np.zeros(FEATURE_COUNT)
        prior[FEATURE_NAMES.index("flux_product")] = scale
        return cls(prior_weights=prior)

    def scores(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, float)
        if (
            features.ndim != 2
            or features.shape[1] != FEATURE_COUNT
            or not np.isfinite(features).all()
        ):
            raise ValueError("Invalid candidate features")
        return features @ (self.weights + self.prior_weights)

    def update(
        self,
        features: np.ndarray,
        reward: float,
        bootstrap: float,
        *,
        alpha: float,
        gamma: float,
        terminal: bool,
    ) -> float:
        features = np.asarray(features, float)
        if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
            raise ValueError("Invalid update features")
        target = reward + (0 if terminal else gamma * bootstrap)
        delta = float(target - features @ (self.weights + self.prior_weights))
        self.weights += alpha * delta * features / (1 + float(features @ features))
        self.updates += 1
        if not np.isfinite(self.weights).all() or np.max(np.abs(self.weights)) > 1e6:
            raise ArithmeticError("Expected Sarsa weights diverged")
        return delta

    def digest(self) -> str:
        payload = self.weights.tobytes() + str(self.updates).encode()
        if np.any(self.prior_weights):
            payload += b"flux_prior_v2" + self.prior_weights.tobytes()
        return hashlib.sha256(payload).hexdigest()

    def save(self, path: Path, identity: str, records: list[dict]) -> None:
        buffer = BytesIO()
        np.savez(
            buffer,
            schema=np.array(
                "candidate_expected_sarsa_v2"
                if np.any(self.prior_weights)
                else "candidate_expected_sarsa_v1"
            ),
            feature_names=np.array(FEATURE_NAMES),
            weights=self.weights,
            prior_weights=self.prior_weights,
            updates=self.updates,
            identity=np.array(identity),
            records=np.array(json.dumps(records, allow_nan=False)),
        )
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_bytes(buffer.getvalue())
            replace_with_retry(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path, identity: str) -> tuple["CandidatePolicy", list[dict]]:
        with np.load(path, allow_pickle=False) as data:
            weights = data["weights"].copy()
            prior = (
                data["prior_weights"].copy() if "prior_weights" in data else np.zeros(FEATURE_COUNT)
            )
            if (
                str(data["schema"])
                not in ("candidate_expected_sarsa_v1", "candidate_expected_sarsa_v2")
                or str(data["identity"]) != identity
                or tuple(data["feature_names"]) != FEATURE_NAMES
                or weights.shape != (FEATURE_COUNT,)
                or not np.isfinite(weights).all()
                or prior.shape != (FEATURE_COUNT,)
                or not np.isfinite(prior).all()
                or (str(data["schema"]) == "candidate_expected_sarsa_v1" and np.any(prior))
                or int(data["updates"]) < 0
            ):
                raise ValueError("Incompatible candidate-policy checkpoint")
            records = json.loads(str(data["records"]))
            if not isinstance(records, list):
                raise ValueError("Checkpoint episode records are invalid")
            return cls(weights, int(data["updates"]), prior), records
