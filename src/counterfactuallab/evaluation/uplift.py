"""Observable uplift evaluation for randomized experiments."""
from __future__ import annotations
import numpy as np
import pandas as pd


def uplift_curve(data: pd.DataFrame, scores: np.ndarray, *, n_bins: int = 10) -> pd.DataFrame:
    """Estimate cumulative treatment effect down a ranking using factual outcomes."""
    scores = np.asarray(scores, dtype=float)
    if not {"treatment", "outcome"}.issubset(data.columns):
        raise ValueError("data must contain treatment and outcome")
    if scores.ndim != 1 or scores.shape != (len(data),):
        raise ValueError("scores must have one value per row")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not isinstance(n_bins, int) or not 1 <= n_bins <= len(data):
        raise ValueError("invalid n_bins")
    ranked=data[["treatment","outcome"]].copy(); ranked["score"]=scores
    ranked=ranked.sort_values("score",ascending=False,kind="stable").reset_index(drop=True)
    rows=[]; n=len(ranked)
    for step in range(1,n_bins+1):
        end=int(np.ceil(step*n/n_bins)); prefix=ranked.iloc[:end]
        yt=prefix.loc[prefix.treatment.eq(1),"outcome"]; yc=prefix.loc[prefix.treatment.eq(0),"outcome"]
        if yt.empty or yc.empty: raise ValueError("each prefix needs treated and control observations")
        effect=float(yt.mean()-yc.mean())
        rows.append({"fraction":end/n,"n":end,"treated_n":len(yt),"control_n":len(yc),"estimated_effect":effect,"incremental_outcomes":end*effect})
    return pd.DataFrame(rows)


def area_under_uplift_curve(curve: pd.DataFrame) -> float:
    """Trapezoidal area under cumulative incremental outcomes vs population share."""
    if not {"fraction","incremental_outcomes"}.issubset(curve.columns) or curve.empty:
        raise ValueError("invalid uplift curve")
    x=np.r_[0.0,curve["fraction"].to_numpy(float)]; y=np.r_[0.0,curve["incremental_outcomes"].to_numpy(float)]
    return float(np.trapezoid(y,x))
