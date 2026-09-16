import numpy as np
import pytest

from counterfactuallab.data.synthetic import SyntheticConfig, generate_randomized_data
from counterfactuallab.models.response import ResponseModel


def test_response_model_fits_and_returns_probabilities() -> None:
    data = generate_randomized_data(SyntheticConfig(n_samples=2_000, seed=11))
    model = ResponseModel().fit(data)
    scores = model.predict_response(data)

    assert scores.shape == (len(data),)
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_response_model_uses_only_observed_x_features() -> None:
    data = generate_randomized_data(SyntheticConfig(n_samples=2_000, seed=12))
    model = ResponseModel().fit(data)

    assert model.feature_columns_ == ["x1", "x2", "x3", "x4", "x5"]
    assert "tau_true" not in model.feature_columns_
    assert "mu1_true" not in model.feature_columns_
    assert "outcome" not in model.feature_columns_
    assert "treatment" not in model.feature_columns_


def test_predict_before_fit_is_rejected() -> None:
    data = generate_randomized_data(SyntheticConfig(n_samples=100))
    with pytest.raises(RuntimeError, match="fit"):
        ResponseModel().predict_response(data)
