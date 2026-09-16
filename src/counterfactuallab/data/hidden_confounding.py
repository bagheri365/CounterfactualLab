"""Synthetic scenario with an intentionally unobserved confounder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from counterfactuallab.data.synthetic import SyntheticConfig, generate_synthetic_data


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_hidden_confounding_data(
    n_samples: int = 20_000, seed: int = 42
) -> pd.DataFrame:
    """Generate data where latent U affects both treatment and outcome.

    ``u_hidden`` is retained only because this is a simulation. Standard model
    feature selection still sees x1..x5 only.
    """

    base = generate_synthetic_data(
        SyntheticConfig(n_samples=n_samples, scenario="randomized", seed=seed)
    )
    rng = np.random.default_rng(seed + 10_000)
    u = rng.normal(size=n_samples)

    x1 = base["x1"].to_numpy()
    x2 = base["x2"].to_numpy()
    x3 = base["x3"].to_numpy()
    x4 = base["x4"].to_numpy()
    x5 = base["x5"].to_numpy()

    # U raises baseline response and treatment probability. Observed X also
    # confounds treatment, so balancing X alone is useful but insufficient.
    baseline_logit = -1.0 + 0.9*x1 - 0.7*x2 + 0.25*x5 + 1.0*u
    mu0 = _sigmoid(baseline_logit)
    desired_effect = 0.06 + 0.16*np.tanh(0.9*x3 - 0.7*x4)
    mu1 = np.clip(mu0 + desired_effect, 0.01, 0.99)
    tau = mu1 - mu0

    propensity = _sigmoid(0.9*x1 - 0.7*x2 + 0.25*x5 + 1.2*u)
    treatment = rng.binomial(1, propensity)
    y0 = rng.binomial(1, mu0)
    y1 = rng.binomial(1, mu1)
    outcome = np.where(treatment == 1, y1, y0)

    data = base.copy()
    data["treatment"] = treatment
    data["outcome"] = outcome
    data["mu0_true"] = mu0
    data["mu1_true"] = mu1
    data["tau_true"] = tau
    data["propensity_true"] = propensity
    data["y0"] = y0
    data["y1"] = y1
    data["u_hidden"] = u
    data["scenario"] = "hidden_confounder"
    return data
