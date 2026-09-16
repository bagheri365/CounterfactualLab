"""Oracle policy evaluation utilities for synthetic experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Select exactly the k highest-scoring rows."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not 1 <= k <= len(scores):
        raise ValueError("k must be between 1 and the number of rows")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")

    # Stable sorting gives deterministic tie handling.
    order = np.argsort(-scores, kind="stable")
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:k]] = True
    return selected


def oracle_incremental_conversions(
    data: pd.DataFrame, selected: np.ndarray
) -> float:
    """Expected incremental conversions under a selected synthetic policy.

    Uses true CATE and is therefore simulation-only.
    """

    selected = np.asarray(selected, dtype=bool)
    if selected.shape != (len(data),):
        raise ValueError("selected mask must have one value per row")
    return float(data.loc[selected, "tau_true"].sum())


def incremental_conversions_per_1000(
    data: pd.DataFrame, selected: np.ndarray
) -> float:
    """Expected incremental conversions per 1,000 treatments."""

    selected = np.asarray(selected, dtype=bool)
    n_selected = int(selected.sum())
    if n_selected == 0:
        raise ValueError("policy must select at least one row")
    return oracle_incremental_conversions(data, selected) / n_selected * 1_000.0


def top_k_overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Fraction of selected rows shared by two equal-budget policies."""

    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    if left.shape != right.shape:
        raise ValueError("policy masks must have the same shape")
    left_n = int(left.sum())
    right_n = int(right.sum())
    if left_n == 0 or left_n != right_n:
        raise ValueError("policies must select the same positive number of rows")
    return float(np.logical_and(left, right).sum() / left_n)
