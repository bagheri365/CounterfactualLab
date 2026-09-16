import numpy as np
import pytest

from counterfactuallab.evaluation.cate import cate_correlation, cate_rmse


def test_cate_rmse() -> None:
    truth = np.array([0.0, 0.1, 0.2])
    estimate = np.array([0.0, 0.2, 0.1])
    expected = np.sqrt((0.0**2 + 0.1**2 + 0.1**2) / 3.0)
    assert cate_rmse(truth, estimate) == pytest.approx(expected)


def test_cate_correlation_is_one_for_affine_ranking() -> None:
    truth = np.array([-0.1, 0.0, 0.1, 0.2])
    estimate = 2.0 * truth + 0.3
    assert cate_correlation(truth, estimate) == pytest.approx(1.0)


def test_cate_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        cate_rmse(np.array([0.1, 0.2]), np.array([0.1]))
