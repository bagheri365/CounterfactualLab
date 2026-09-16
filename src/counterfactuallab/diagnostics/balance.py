"""Simple covariate-balance diagnostics."""
import numpy as np
import pandas as pd
from counterfactuallab.data.synthetic import model_feature_columns


def standardized_mean_differences(data: pd.DataFrame) -> pd.Series:
    treated=data[data.treatment.eq(1)]; control=data[data.treatment.eq(0)]
    if treated.empty or control.empty: raise ValueError("balance diagnostics require both treatment groups")
    result={}
    for c in model_feature_columns(data):
        sd=np.sqrt((treated[c].var(ddof=1)+control[c].var(ddof=1))/2)
        result[c]=0.0 if sd==0 else (treated[c].mean()-control[c].mean())/sd
    return pd.Series(result,name="smd")


def max_absolute_smd(data: pd.DataFrame) -> float:
    return float(standardized_mean_differences(data).abs().max())
