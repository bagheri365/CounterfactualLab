"""Inverse-probability weighting for observed confounding."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ipw_weights(
    treatment: np.ndarray | pd.Series,
    propensity: np.ndarray | pd.Series,
) -> np.ndarray:
    """Return unstabilized ATE inverse-probability weights."""

    t = np.asarray(treatment, dtype=float)
    e = np.asarray(propensity, dtype=float)
    if t.shape != e.shape:
        raise ValueError("treatment and propensity must have the same shape")
    if np.any((e <= 0.0) | (e >= 1.0)):
        raise ValueError("propensities must lie strictly between 0 and 1")
    return t / e + (1.0 - t) / (1.0 - e)


def ipw_ate(
    data: pd.DataFrame,
    propensity: np.ndarray | pd.Series,
) -> float:
    """Estimate ATE with normalized (Hájek) inverse-probability weighting."""

    t = data["treatment"].to_numpy(dtype=float)
    y = data["outcome"].to_numpy(dtype=float)
    e = np.asarray(propensity, dtype=float)
    weights = ipw_weights(t, e)

    treated = t == 1.0
    control = ~treated
    mu1 = np.sum(weights[treated] * y[treated]) / np.sum(weights[treated])
    mu0 = np.sum(weights[control] * y[control]) / np.sum(weights[control])
    return float(mu1 - mu0)
