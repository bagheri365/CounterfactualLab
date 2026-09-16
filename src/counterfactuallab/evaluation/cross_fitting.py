"""Out-of-fold ranking scores for honest uplift evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from counterfactuallab.models.response import ResponseModel
from counterfactuallab.models.t_learner import TLearner


def cross_fitted_ranking_scores(
    data: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return out-of-fold response and T-Learner CATE scores.

    Each row is scored by models fit without that row. This separates learning
    the ranking from evaluating that ranking with the row's factual outcome.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > len(data):
        raise ValueError("n_splits cannot exceed the number of rows")

    response_scores = np.empty(len(data), dtype=float)
    cate_scores = np.empty(len(data), dtype=float)
    assigned = np.zeros(len(data), dtype=bool)

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(data):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        response_scores[test_idx] = ResponseModel().fit(train).predict_response(test)
        cate_scores[test_idx] = TLearner().fit(train).predict_cate(test)
        assigned[test_idx] = True

    if not assigned.all():
        raise RuntimeError("cross-fitting failed to score every row")

    return response_scores, cate_scores
