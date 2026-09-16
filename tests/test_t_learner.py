import numpy as np
import pytest

from counterfactuallab.data.synthetic import SyntheticConfig, generate_randomized_data
from counterfactuallab.models.t_learner import TLearner


def test_t_learner_returns_valid_outcome_probabilities_and_cate() -> None:
    data = generate_randomized_data(SyntheticConfig(n_samples=3_000, seed=21))
    learner = TLearner().fit(data)

    mu1_hat = learner.predict_mu1(data)
    mu0_hat = learner.predict_mu0(data)
    tau_hat = learner.predict_cate(data)

    assert mu1_hat.shape == (len(data),)
    assert mu0_hat.shape == (len(data),)
    assert tau_hat.shape == (len(data),)
    assert ((mu1_hat >= 0.0) & (mu1_hat <= 1.0)).all()
    assert ((mu0_hat >= 0.0) & (mu0_hat <= 1.0)).all()
    np.testing.assert_allclose(tau_hat, mu1_hat - mu0_hat)


def test_t_learner_uses_only_observed_covariates() -> None:
    data = generate_randomized_data(SyntheticConfig(n_samples=2_000, seed=22))
    learner = TLearner().fit(data)

    assert learner.feature_columns_ == ["x1", "x2", "x3", "x4", "x5"]
    assert "tau_true" not in learner.feature_columns_
    assert "mu0_true" not in learner.feature_columns_
    assert "mu1_true" not in learner.feature_columns_


def test_t_learner_learns_nonconstant_effects() -> None:
    data = generate_randomized_data(SyntheticConfig(n_samples=5_000, seed=23))
    tau_hat = TLearner().fit(data).predict_cate(data)

    assert np.std(tau_hat) > 0.01


def test_t_learner_rejects_prediction_before_fit() -> None:
    data = generate_randomized_data(SyntheticConfig(n_samples=100))
    with pytest.raises(RuntimeError, match="fit"):
        TLearner().predict_cate(data)
