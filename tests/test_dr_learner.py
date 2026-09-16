import numpy as np

from counterfactuallab.data.synthetic import (
    SyntheticConfig,
    difference_in_means,
    generate_synthetic_data,
    true_ate,
)
from counterfactuallab.models.dr_learner import (
    aipw_pseudo_outcome,
    cross_fitted_aipw_ate,
    cross_fitted_nuisance_predictions,
)


def test_cross_fitted_nuisance_predictions_have_valid_shapes_and_ranges() -> None:
    data = generate_synthetic_data(
        SyntheticConfig(n_samples=4_000, scenario="confounded", seed=61)
    )
    e_hat, mu0_hat, mu1_hat = cross_fitted_nuisance_predictions(
        data, n_splits=4, seed=61
    )
    for prediction in (e_hat, mu0_hat, mu1_hat):
        assert prediction.shape == (len(data),)
        assert np.isfinite(prediction).all()
        assert ((prediction > 0.0) & (prediction < 1.0)).all()


def test_aipw_with_oracle_nuisances_is_close_to_true_ate() -> None:
    data = generate_synthetic_data(
        SyntheticConfig(n_samples=50_000, scenario="confounded", seed=62)
    )
    scores = aipw_pseudo_outcome(
        data,
        data["propensity_true"].to_numpy(),
        data["mu0_true"].to_numpy(),
        data["mu1_true"].to_numpy(),
    )
    assert abs(scores.mean() - true_ate(data)) < 0.02


def test_cross_fitted_dr_reduces_naive_confounding_bias() -> None:
    data = generate_synthetic_data(
        SyntheticConfig(n_samples=40_000, scenario="confounded", seed=63)
    )
    truth = true_ate(data)
    naive_error = abs(difference_in_means(data) - truth)
    dr_error = abs(cross_fitted_aipw_ate(data, n_splits=5, seed=63) - truth)
    assert dr_error < naive_error
    assert dr_error < 0.03


def test_aipw_demonstrates_double_robustness_with_one_oracle_nuisance() -> None:
    data = generate_synthetic_data(
        SyntheticConfig(n_samples=60_000, scenario="confounded", seed=64)
    )
    truth = true_ate(data)
    n = len(data)

    # Correct propensity + deliberately wrong constant outcome regressions.
    wrong_mu0 = np.full(n, 0.5)
    wrong_mu1 = np.full(n, 0.5)
    score_correct_e = aipw_pseudo_outcome(
        data,
        data["propensity_true"].to_numpy(),
        wrong_mu0,
        wrong_mu1,
    )

    # Correct outcome regressions + deliberately wrong constant propensity.
    wrong_e = np.full(n, 0.5)
    score_correct_mu = aipw_pseudo_outcome(
        data,
        wrong_e,
        data["mu0_true"].to_numpy(),
        data["mu1_true"].to_numpy(),
    )

    assert abs(score_correct_e.mean() - truth) < 0.03
    assert abs(score_correct_mu.mean() - truth) < 0.03
