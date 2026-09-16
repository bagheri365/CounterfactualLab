"""M9: show that observed-covariate adjustment cannot remove hidden confounding."""

import numpy as np

from counterfactuallab.data.hidden_confounding import generate_hidden_confounding_data
from counterfactuallab.data.synthetic import difference_in_means, true_ate
from counterfactuallab.diagnostics.balance import standardized_mean_differences
from counterfactuallab.diagnostics.weighted_balance import weighted_standardized_mean_differences
from counterfactuallab.evaluation.ipw import ipw_ate, ipw_weights
from counterfactuallab.models.dr_learner import cross_fitted_aipw_ate
from counterfactuallab.models.propensity import PropensityModel


def weighted_mean(x, w):
    return float(np.sum(x*w) / np.sum(w))


def main() -> None:
    data = generate_hidden_confounding_data()
    truth = true_ate(data)
    propensity = PropensityModel().fit(data).predict(data)
    weights = ipw_weights(data["treatment"], propensity)

    before = standardized_mean_differences(data)
    after = weighted_standardized_mean_differences(data, propensity)

    t = data["treatment"].to_numpy()
    u = data["u_hidden"].to_numpy()
    u_before = u[t == 1].mean() - u[t == 0].mean()
    u_after = weighted_mean(u[t == 1], weights[t == 1]) - weighted_mean(
        u[t == 0], weights[t == 0]
    )

    print("HIDDEN CONFOUNDING")
    print(f"  True ATE:                    {truth:.4f}")
    print(f"  Naive difference in means:   {difference_in_means(data):.4f}")
    print(f"  Observed-X IPW ATE:          {ipw_ate(data, propensity):.4f}")
    print(f"  Observed-X AIPW/DR ATE:      {cross_fitted_aipw_ate(data):.4f}")
    print()
    print("BALANCE")
    print(f"  Max |observed X SMD| before: {np.max(np.abs(before)):.3f}")
    print(f"  Max |observed X SMD| after:  {np.max(np.abs(after)):.3f}")
    print(f"  Hidden U mean diff before:    {u_before:+.3f}")
    print(f"  Hidden U mean diff after:     {u_after:+.3f}")
    print()
    print("u_hidden is oracle-only: the fitted propensity and DR models never see it.")
    print("Balancing observed X does not establish that hidden confounding is absent.")


if __name__ == "__main__":
    main()
