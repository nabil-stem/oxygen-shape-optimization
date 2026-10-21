"""Online regression of measured signed delta Q, with source-access features.

This is a contextual-bandit/ranking baseline, not an RL value function. Targets
are observed global PDE responses; unselected candidates never become labels.
"""

from collections import deque

import numpy as np

from oxygen_shape.geometry.boundary import BoundaryKind, extract_boundary
from oxygen_shape.optimization.candidate_policy import FEATURE_NAMES

ACCESS_FEATURES = ("add_source_path", "donor_source_path", "mean_source_path", "max_source_path")
RANK_FEATURE_NAMES = (*FEATURE_NAMES, *ACCESS_FEATURES)


def source_path_distances(geometry) -> np.ndarray:
    distance = np.full(geometry.inside.shape, np.inf)
    queue = deque()
    for face in geometry.source_faces:
        cell = (face.row, face.col)
        if not np.isfinite(distance[cell]):
            distance[cell] = geometry.h / 2
            queue.append(cell)
    ny, nx = geometry.inside.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if (
                0 <= rr < ny
                and 0 <= cc < nx
                and geometry.inside[rr, cc]
                and not np.isfinite(distance[rr, cc])
            ):
                distance[rr, cc] = distance[r, c] + geometry.h
                queue.append((rr, cc))
    if not np.isfinite(distance[geometry.inside]).all():
        raise ValueError("Source path feature requires a connected domain")
    return distance


def append_access_features(base, geometry, prepared, rows, protected_length):
    distance = source_path_distances(geometry)
    faces = [
        f
        for f in extract_boundary(geometry)
        if f.kind == BoundaryKind.ROBIN
        and (f.key.col + 0.5) * geometry.h >= protected_length + 2 * geometry.h
    ]
    if len(faces) != len(prepared.points):
        raise ValueError("Source-path features and candidate faces do not align")
    boundary = np.array([distance[f.key.row, f.key.col] for f in faces])
    mean, maximum = float(distance[geometry.inside].mean()), float(distance[geometry.inside].max())
    extra = [[boundary[r["addition_face"]], boundary[r["donor_face"]], mean, maximum] for r in rows]
    return np.c_[base, extra]


class ResponseRanker:
    def __init__(self, ridge: float = 1.0):
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("Ridge must be positive and finite")
        self.weights = np.zeros(len(RANK_FEATURE_NAMES))
        self.inverse = np.eye(len(RANK_FEATURE_NAMES)) / ridge
        self.prior = np.zeros(len(RANK_FEATURE_NAMES))
        self.prior[FEATURE_NAMES.index("flux_product")] = 1.0
        self.updates = 0

    def scores(self, features, uncertainty: float = 0.0):
        features = np.asarray(features, float)
        variance = np.maximum(0, np.einsum("ij,jk,ik->i", features, self.inverse, features))
        return features @ (self.weights + self.prior) + uncertainty * np.sqrt(variance)

    def update(self, features, target: float):
        """Recursive ridge least squares, no forgetting or artificial reward clipping."""
        features = np.asarray(features, float)
        prediction = float(features @ (self.weights + self.prior))
        product = self.inverse @ features
        gain = product / (1 + float(features @ product))
        self.weights += gain * (target - prediction)
        self.inverse -= np.outer(gain, product)
        self.inverse = (self.inverse + self.inverse.T) / 2
        self.updates += 1
        if not np.isfinite(self.weights).all() or not np.isfinite(self.inverse).all():
            raise ArithmeticError("Response regression diverged")
        return prediction
