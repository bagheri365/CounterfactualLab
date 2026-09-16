"""Treatment policy based on estimated conditional treatment effects."""
import numpy as np
from counterfactuallab.evaluation.policy import top_k_mask


def uplift_policy(cate_scores: np.ndarray, budget_fraction: float) -> np.ndarray:
    """Treat the budgeted fraction with the highest estimated CATE."""
    k = max(1, int(len(cate_scores) * budget_fraction))
    return top_k_mask(cate_scores, k)
