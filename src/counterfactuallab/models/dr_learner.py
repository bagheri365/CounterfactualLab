"""Doubly robust estimation with out-of-fold nuisance predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from counterfactuallab.data.synthetic import model_feature_columns


def cross_fitted_nuisance_predictions(
    data: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate e(X), mu0(X), and mu1(X) out of fold.

    Every row is predicted by models that were fit without that row.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    features = model_feature_columns(data)
    x = data[features].to_numpy()
    treatment = data["treatment"].to_numpy()
    outcome = data["outcome"].to_numpy()

    e_hat = np.empty(len(data))
    mu0_hat = np.empty(len(data))
    mu1_hat = np.empty(len(data))

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        t_train, y_train = treatment[train_idx], outcome[train_idx]

        propensity_model = LogisticRegression(max_iter=1_000)
        propensity_model.fit(x_train, t_train)
        e_hat[test_idx] = propensity_model.predict_proba(x_test)[:, 1]

        control_model = LogisticRegression(max_iter=1_000)
        control_model.fit(x_train[t_train == 0], y_train[t_train == 0])
        mu0_hat[test_idx] = control_model.predict_proba(x_test)[:, 1]

        treated_model = LogisticRegression(max_iter=1_000)
        treated_model.fit(x_train[t_train == 1], y_train[t_train == 1])
        mu1_hat[test_idx] = treated_model.predict_proba(x_test)[:, 1]

    return e_hat, mu0_hat, mu1_hat


def aipw_pseudo_outcome(
    data: pd.DataFrame,
    e_hat: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
    clip: float = 0.01,
) -> np.ndarray:
    """Return the augmented-IPW score whose mean estimates the ATE."""

    if not 0.0 < clip < 0.5:
        raise ValueError("clip must lie between 0 and 0.5")

    t = data["treatment"].to_numpy(dtype=float)
    y = data["outcome"].to_numpy(dtype=float)
    e = np.clip(np.asarray(e_hat, dtype=float), clip, 1.0 - clip)
    mu0 = np.asarray(mu0_hat, dtype=float)
    mu1 = np.asarray(mu1_hat, dtype=float)

    if not (len(t) == len(e) == len(mu0) == len(mu1)):
        raise ValueError("all predictions must have one value per row")

    return (
        mu1
        - mu0
        + t * (y - mu1) / e
        - (1.0 - t) * (y - mu0) / (1.0 - e)
    )


def cross_fitted_aipw_ate(
    data: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """Estimate the ATE using cross-fitted nuisance models and AIPW."""

    e_hat, mu0_hat, mu1_hat = cross_fitted_nuisance_predictions(
        data, n_splits=n_splits, seed=seed
    )
    scores = aipw_pseudo_outcome(data, e_hat, mu0_hat, mu1_hat)
    return float(scores.mean())
