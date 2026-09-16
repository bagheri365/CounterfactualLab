"""Propensity-score estimation from observed pre-treatment covariates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from counterfactuallab.data.synthetic import model_feature_columns


class PropensityModel:
    """Estimate P(T=1 | X) with logistic regression."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1_000)
        self.feature_columns: list[str] | None = None

    def fit(self, data: pd.DataFrame) -> "PropensityModel":
        self.feature_columns = model_feature_columns(data)
        self.model.fit(data[self.feature_columns], data["treatment"])
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.feature_columns is None:
            raise RuntimeError("fit the propensity model before predicting")
        return self.model.predict_proba(data[self.feature_columns])[:, 1]
