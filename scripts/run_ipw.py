"""M6: estimate propensity scores and correct observed confounding with IPW."""

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
from counterfactuallab.evaluation.ipw import ipw_ate
from counterfactuallab.models.propensity import PropensityModel


def main() -> None:
    data = generate_synthetic_data(
        SyntheticConfig(n_samples=20_000, scenario="confounded", seed=42)
    )

    model = PropensityModel().fit(data)
    estimated_propensity = model.predict(data)

    truth = true_ate(data)
    naive = difference_in_means(data)
    weighted = ipw_ate(data, estimated_propensity)
    weighted_smd = weighted_standardized_mean_differences(
        data, estimated_propensity
    )

    print("CONFOUNDED DATA")
    print(f"  True ATE:                       {truth:.4f}")
    print(f"  Naive difference in means:      {naive:.4f}")
    print(f"  Estimated-propensity IPW ATE:   {weighted:.4f}")
    print()
    print("BALANCE")
    print(f"  Max |SMD| before weighting:     {max_absolute_smd(data):.3f}")
    print(f"  Max |SMD| after weighting:      {weighted_smd.abs().max():.3f}")
    print()
    print("ESTIMATED PROPENSITY")
    print(f"  min: {estimated_propensity.min():.3f}")
    print(f"  max: {estimated_propensity.max():.3f}")
    print(f"  mean: {estimated_propensity.mean():.3f}")


if __name__ == "__main__":
    main()
