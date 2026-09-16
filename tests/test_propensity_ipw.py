import numpy as np

from counterfactuallab.data.synthetic import (
    SyntheticConfig,
    difference_in_means,
    generate_synthetic_data,
    true_ate,
)
from counterfactuallab.diagnostics.balance import max_absolute_smd
from counterfactuallab.diagnostics.weighted_balance import (
    weighted_standardized_mean_differences,
)
from counterfactuallab.evaluation.ipw import ipw_ate, ipw_weights
from counterfactuallab.models.propensity import PropensityModel


def test_propensity_model_uses_observed_features_not_oracles() -> None:
    data = generate_synthetic_data(
        SyntheticConfig(n_samples=2_000, scenario="confounded", seed=51)
    )
    model = PropensityModel().fit(data)
    assert model.feature_columns == ["x1", "x2", "x3", "x4", "x5"]
    assert "propensity_true" not in model.feature_columns


def test_ipw_weights_match_definition() -> None:
    treatment = np.array([1, 0, 1, 0])
    propensity = np.array([0.8, 0.8, 0.2, 0.2])
    np.testing.assert_allclose(
        ipw_weights(treatment, propensity),
        [1.25, 5.0, 5.0, 1.25],
    )


def test_ipw_reduces_ate_bias_under_observed_confounding() -> None:
    data = generate_synthetic_data(
        SyntheticConfig(n_samples=40_000, scenario="confounded", seed=52)
    )
    propensity = PropensityModel().fit(data).predict(data)
    truth = true_ate(data)
    naive_error = abs(difference_in_means(data) - truth)
    ipw_error = abs(ipw_ate(data, propensity) - truth)
    assert ipw_error < naive_error
    assert ipw_error < 0.03


def test_ipw_improves_covariate_balance() -> None:
    data = generate_synthetic_data(
        SyntheticConfig(n_samples=30_000, scenario="confounded", seed=53)
    )
    propensity = PropensityModel().fit(data).predict(data)
    before = max_absolute_smd(data)
    after = weighted_standardized_mean_differences(data, propensity).abs().max()
    assert after < before
    assert after < 0.10
