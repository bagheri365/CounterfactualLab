"""Conventional response prediction for treated users.

This model answers a predictive question:

    P(Y = 1 | X=x, T=1)

It is intentionally *not* a treatment-effect estimator. M2 uses it to show
why high predicted response and high incremental effect are different targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from counterfactuallab.data.synthetic import model_feature_columns


@dataclass
class ResponseModel:
    """Predict factual response under treatment using treated observations."""

    estimator: object | None = None

    def fit(self, data: pd.DataFrame) -> "ResponseModel":
        features = model_feature_columns(data)
        treated = data.loc[data["treatment"].eq(1)]

        if treated.empty:
            raise ValueError("response model requires treated observations")
        if treated["outcome"].nunique() < 2:
            raise ValueError("treated outcomes must contain both classes")

        base = self.estimator or LogisticRegression(max_iter=1_000)
        self.estimator_ = clone(base)
        self.feature_columns_ = features
        self.estimator_.fit(treated[features], treated["outcome"])
        return self

    def predict_response(self, data: pd.DataFrame) -> np.ndarray:
        """Return estimated response probability if each row were treated."""

        if not hasattr(self, "estimator_"):
            raise RuntimeError("fit must be called before predict_response")
        return self.estimator_.predict_proba(data[self.feature_columns_])[:, 1]
