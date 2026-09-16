"""Oracle policy-value helpers for synthetic experiments."""
import numpy as np
import pandas as pd


def oracle_policy_value(data: pd.DataFrame, policy: np.ndarray) -> float:
    """Expected outcome under a fixed binary policy using synthetic mu0/mu1."""
    p=np.asarray(policy,dtype=bool)
    if len(p)!=len(data): raise ValueError("policy must have one decision per row")
    return float(np.where(p, data["mu1_true"], data["mu0_true"]).mean())


def oracle_incremental_value(data: pd.DataFrame, policy: np.ndarray) -> float:
    """Expected gain over treating nobody."""
    return oracle_policy_value(data,policy)-float(data["mu0_true"].mean())
