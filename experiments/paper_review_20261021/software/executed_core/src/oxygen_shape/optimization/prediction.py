"""Out-of-sample ranking diagnostics; no calibrated flux prediction is assumed."""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

SCORE_NAMES = ("spatial_sensitivity", "spatial_delta", "radius_sensitivity", "radius_delta")


def predict_scores(
    training_features: NDArray,
    training_delta: NDArray,
    training_area: NDArray,
    query_features: NDArray,
    spatial_bandwidth: float,
    radius_bandwidth: float,
) -> dict[str, NDArray]:
    """Features are [addition_x,y, donor_x,y, addition_radius]. Return signed sums."""
    train, query = (
        np.asarray(training_features, dtype=float),
        np.asarray(query_features, dtype=float),
    )
    delta, area = np.asarray(training_delta, dtype=float), np.asarray(training_area, dtype=float)
    if any(
        a.ndim != 2 or a.shape[1] != 5 or not len(a) or not np.isfinite(a).all()
        for a in (train, query)
    ):
        raise ValueError("Features must be nonempty finite (N,5) arrays")
    if (
        delta.shape != (len(train),)
        or area.shape != delta.shape
        or not np.isfinite(delta).all()
        or not np.isfinite(area).all()
        or np.any(area <= 0)
    ):
        raise ValueError("Training responses must be finite, with positive reallocated areas")
    if (
        not np.isfinite([spatial_bandwidth, radius_bandwidth]).all()
        or min(spatial_bandwidth, radius_bandwidth) <= 0
    ):
        raise ValueError("Kernel bandwidths must be positive and finite")
    distance = np.sum((query[:, None, :4] - train[None, :, :4]) ** 2, axis=2)
    kernel = np.exp(-distance / (2 * spatial_bandwidth**2))
    radial = kernel * np.exp(
        -((query[:, None, 4] - train[None, :, 4]) ** 2) / (2 * radius_bandwidth**2)
    )
    return dict(
        zip(
            SCORE_NAMES,
            (kernel @ (delta / area), kernel @ delta, radial @ (delta / area), radial @ delta),
            strict=True,
        )
    )


def rank_correlation(scores: NDArray, actual: NDArray) -> float | None:
    if len(scores) < 2 or np.ptp(scores) == 0 or np.ptp(actual) == 0:
        return None
    result = float(spearmanr(scores, actual).statistic)
    return result if np.isfinite(result) else None


def top_weights(scores: NDArray, fraction: float) -> NDArray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all() or not 0 < fraction < 1:
        raise ValueError("Top fraction needs a finite vector and a fraction in (0,1)")
    count = int(np.ceil(len(scores) * fraction))
    threshold = np.partition(scores, len(scores) - count)[len(scores) - count]
    weights = (scores > threshold).astype(float)
    ties = scores == threshold
    weights[ties] = (count - weights.sum()) / ties.sum()
    return weights / count


def ranking_metrics(
    signed_scores: NDArray, delta: NDArray, area: NDArray, top_fraction: float, tolerance: float
) -> dict:
    score, delta, area = (np.asarray(v, dtype=float) for v in (signed_scores, delta, area))
    if (
        score.ndim != 1
        or score.shape != delta.shape
        or area.shape != delta.shape
        or not score.size
        or not all(np.isfinite(v).all() for v in (score, delta, area))
        or np.any(area <= 0)
    ):
        raise ValueError(
            "Scores and actual responses need aligned finite vectors and positive areas"
        )
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Effect tolerance must be finite and positive")
    ranking = np.maximum(score, 0)
    weights = top_weights(ranking, top_fraction)
    resolved = np.abs(delta) > tolerance
    positive = delta > tolerance
    top_mean = float(weights @ delta)
    return {
        "heldout_count": len(delta),
        "spearman_delta_Q": rank_correlation(ranking, delta),
        "spearman_sensitivity": rank_correlation(ranking, delta / area),
        "top_mean_delta_Q": top_mean,
        "pool_mean_delta_Q": float(delta.mean()),
        "top_gain_over_pool_mean": top_mean - float(delta.mean()),
        "top_positive_fraction": float(weights @ positive.astype(float)),
        "resolved_count": int(resolved.sum()),
        "sign_accuracy": float(np.mean(np.sign(score[resolved]) == np.sign(delta[resolved])))
        if resolved.any()
        else None,
        "always_positive_accuracy": float(positive[resolved].mean()) if resolved.any() else None,
    }
