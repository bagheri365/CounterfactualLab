"""Oracle CATE metrics for synthetic experiments."""

from __future__ import annotations

import numpy as np


def cate_rmse(tau_true: np.ndarray, tau_hat: np.ndarray) -> float:
    """Root mean squared error of estimated conditional treatment effects."""

    truth, estimate = _validated_pair(tau_true, tau_hat)
    return float(np.sqrt(np.mean((truth - estimate) ** 2)))


def cate_correlation(tau_true: np.ndarray, tau_hat: np.ndarray) -> float:
    """Pearson correlation between true and estimated CATE."""

    truth, estimate = _validated_pair(tau_true, tau_hat)
    if np.std(truth) == 0.0 or np.std(estimate) == 0.0:
        raise ValueError("CATE correlation requires non-constant arrays")
    return float(np.corrcoef(truth, estimate)[0, 1])


def _validated_pair(
    tau_true: np.ndarray, tau_hat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(tau_true, dtype=float)
    estimate = np.asarray(tau_hat, dtype=float)

    if truth.ndim != 1 or estimate.ndim != 1:
        raise ValueError("CATE arrays must be one-dimensional")
    if truth.shape != estimate.shape:
        raise ValueError("true and estimated CATE must have the same shape")
    if len(truth) == 0:
        raise ValueError("CATE arrays must not be empty")
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ValueError("CATE arrays must contain only finite values")
    return truth, estimate
