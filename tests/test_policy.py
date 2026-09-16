import numpy as np
import pandas as pd
import pytest

from counterfactuallab.evaluation.policy import (
    incremental_conversions_per_1000,
    oracle_incremental_conversions,
    top_k_mask,
    top_k_overlap,
)


def test_top_k_mask_selects_exact_highest_scores() -> None:
    scores = np.array([0.1, 0.9, 0.4, 0.8])
    selected = top_k_mask(scores, 2)
    np.testing.assert_array_equal(selected, [False, True, False, True])


def test_top_k_mask_has_deterministic_ties() -> None:
    selected = top_k_mask(np.array([0.5, 0.5, 0.2]), 1)
    np.testing.assert_array_equal(selected, [True, False, False])


@pytest.mark.parametrize("k", [0, 4])
def test_invalid_budget_is_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        top_k_mask(np.array([0.1, 0.2, 0.3]), k)


def test_oracle_incremental_conversions_and_rate() -> None:
    data = pd.DataFrame({"tau_true": [0.10, -0.05, 0.20]})
    selected = np.array([True, False, True])

    assert oracle_incremental_conversions(data, selected) == pytest.approx(0.30)
    assert incremental_conversions_per_1000(data, selected) == pytest.approx(150.0)


def test_top_k_overlap() -> None:
    left = np.array([True, True, False, False])
    right = np.array([False, True, True, False])
    assert top_k_overlap(left, right) == pytest.approx(0.5)
