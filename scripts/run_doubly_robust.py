"""M7: compare naive, IPW, and cross-fitted doubly robust ATE estimates."""

from counterfactuallab.data.synthetic import (
    SyntheticConfig,
    difference_in_means,
    generate_synthetic_data,
    true_ate,
)
from counterfactuallab.evaluation.ipw import ipw_ate
from counterfactuallab.models.dr_learner import (
    cross_fitted_aipw_ate,
    cross_fitted_nuisance_predictions,
)
from counterfactuallab.models.propensity import PropensityModel


def main() -> None:
    data = generate_synthetic_data(
        SyntheticConfig(n_samples=20_000, scenario="confounded", seed=42)
    )

    truth = true_ate(data)
    naive = difference_in_means(data)

    propensity = PropensityModel().fit(data).predict(data)
    ipw = ipw_ate(data, propensity)

    e_hat, mu0_hat, mu1_hat = cross_fitted_nuisance_predictions(
        data, n_splits=5, seed=42
    )
    dr = cross_fitted_aipw_ate(data, n_splits=5, seed=42)

    print("CONFOUNDED DATA")
    print(f"  True ATE:                     {truth:.4f}")
    print(f"  Naive difference in means:    {naive:.4f}")
    print(f"  Estimated-propensity IPW:     {ipw:.4f}")
    print(f"  Cross-fitted AIPW/DR ATE:     {dr:.4f}")
    print()
    print("OUT-OF-FOLD NUISANCE PREDICTIONS")
    print(f"  propensity range:             {e_hat.min():.3f} to {e_hat.max():.3f}")
    print(f"  mu0 range:                    {mu0_hat.min():.3f} to {mu0_hat.max():.3f}")
    print(f"  mu1 range:                    {mu1_hat.min():.3f} to {mu1_hat.max():.3f}")
    print()
    print("All nuisance predictions above are out of fold.")
    print("DR still assumes no unobserved confounding and adequate overlap.")


if __name__ == "__main__":
    main()
