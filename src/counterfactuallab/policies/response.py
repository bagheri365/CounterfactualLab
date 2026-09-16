"""Treatment policy based on predicted response under treatment."""
import numpy as np
from counterfactuallab.evaluation.policy import top_k_mask


def response_policy(scores: np.ndarray, budget_fraction: float) -> np.ndarray:
    """Treat the budgeted fraction with the highest response scores."""
    k = max(1, int(len(scores) * budget_fraction))
    return top_k_mask(scores, k)
