"""Balance diagnostics after inverse-probability weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from counterfactuallab.data.synthetic import model_feature_columns
from counterfactuallab.evaluation.ipw import ipw_weights


def weighted_standardized_mean_differences(
    data: pd.DataFrame,
    propensity: np.ndarray | pd.Series,
) -> pd.Series:
    """Compute treated-minus-control SMDs after IPW."""

    t = data["treatment"].to_numpy()
    w = ipw_weights(t, propensity)
    result = {}

    for column in model_feature_columns(data):
        x = data[column].to_numpy(dtype=float)
        xt, xc = x[t == 1], x[t == 0]
        wt, wc = w[t == 1], w[t == 0]

        mt = np.average(xt, weights=wt)
        mc = np.average(xc, weights=wc)
        vt = np.average((xt - mt) ** 2, weights=wt)
        vc = np.average((xc - mc) ** 2, weights=wc)
        pooled_sd = np.sqrt((vt + vc) / 2.0)
        result[column] = 0.0 if pooled_sd == 0.0 else (mt - mc) / pooled_sd

    return pd.Series(result, name="weighted_smd")
