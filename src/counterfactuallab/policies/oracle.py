"""Simulation-only oracle treatment policy."""
import numpy as np
from counterfactuallab.evaluation.policy import top_k_mask


def oracle_policy(true_cate: np.ndarray, budget_fraction: float) -> np.ndarray:
    """Treat the budgeted fraction with the highest true synthetic CATE."""
    k = max(1, int(len(true_cate) * budget_fraction))
    return top_k_mask(true_cate, k)
