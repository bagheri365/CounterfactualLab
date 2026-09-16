import numpy as np

from counterfactuallab.data.hidden_confounding import generate_hidden_confounding_data
from counterfactuallab.data.synthetic import model_feature_columns, true_ate
from counterfactuallab.diagnostics.weighted_balance import weighted_standardized_mean_differences
from counterfactuallab.evaluation.ipw import ipw_ate
from counterfactuallab.models.dr_learner import cross_fitted_aipw_ate
from counterfactuallab.models.propensity import PropensityModel


def test_hidden_confounder_is_excluded_from_model_features():
    data = generate_hidden_confounding_data(2_000, 81)
    assert "u_hidden" in data.columns
    assert "u_hidden" not in model_feature_columns(data)


def test_observed_balance_can_improve_while_hidden_imbalance_remains():
    data = generate_hidden_confounding_data(20_000, 82)
    e = PropensityModel().fit(data).predict(data)
    smd = weighted_standardized_mean_differences(data, e)
    assert np.max(np.abs(smd)) < .10

    t = data.treatment.to_numpy()
    u = data.u_hidden.to_numpy()
    assert abs(u[t == 1].mean() - u[t == 0].mean()) > .50


def test_observed_adjustment_does_not_recover_true_ate():
    data = generate_hidden_confounding_data(40_000, 83)
    truth = true_ate(data)
    e = PropensityModel().fit(data).predict(data)
    ipw_error = abs(ipw_ate(data, e) - truth)
    dr_error = abs(cross_fitted_aipw_ate(data, seed=83) - truth)
    assert ipw_error > .03
    assert dr_error > .03
