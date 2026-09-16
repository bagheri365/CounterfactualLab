"""A minimal T-Learner for binary outcomes.

The T-Learner fits one response model in the treated group and another in the
control group, then estimates CATE by subtracting their predicted probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from counterfactuallab.data.synthetic import model_feature_columns


@dataclass
class TLearner:
    """Estimate heterogeneous treatment effects with two outcome models."""

    estimator: object | None = None

    def fit(self, data: pd.DataFrame) -> "TLearner":
        features = model_feature_columns(data)
        treated = data.loc[data["treatment"].eq(1)]
        control = data.loc[data["treatment"].eq(0)]

        if treated.empty or control.empty:
            raise ValueError("T-Learner requires both treatment groups")
        if treated["outcome"].nunique() < 2:
            raise ValueError("treated outcomes must contain both classes")
        if control["outcome"].nunique() < 2:
            raise ValueError("control outcomes must contain both classes")

        base = self.estimator or LogisticRegression(max_iter=1_000)
        self.treated_model_ = clone(base)
        self.control_model_ = clone(base)
        self.feature_columns_ = features

        self.treated_model_.fit(treated[features], treated["outcome"])
        self.control_model_.fit(control[features], control["outcome"])
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "treated_model_"):
            raise RuntimeError("fit must be called before prediction")

    def predict_mu1(self, data: pd.DataFrame) -> np.ndarray:
        """Estimate P(Y(1)=1 | X=x)."""

        self._require_fitted()
        return self.treated_model_.predict_proba(data[self.feature_columns_])[:, 1]

    def predict_mu0(self, data: pd.DataFrame) -> np.ndarray:
        """Estimate P(Y(0)=1 | X=x)."""

        self._require_fitted()
        return self.control_model_.predict_proba(data[self.feature_columns_])[:, 1]

    def predict_cate(self, data: pd.DataFrame) -> np.ndarray:
        """Estimate CATE as mu1_hat(x) - mu0_hat(x)."""

        return self.predict_mu1(data) - self.predict_mu0(data)
