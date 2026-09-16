"""Synthetic potential-outcomes data for CounterfactualLab.

The simulator deliberately keeps the causal world simple enough to inspect.
Models should train only on observed pre-treatment features plus factual
treatment/outcome as appropriate. Columns ending in ``_true`` and potential
outcomes are oracle information used only for simulation evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_COLUMNS = tuple(f"x{i}" for i in range(1, 6))
OBSERVED_COLUMNS = (*FEATURE_COLUMNS, "treatment", "outcome")
ORACLE_COLUMNS = (
    "mu0_true",
    "mu1_true",
    "tau_true",
    "propensity_true",
    "y0",
    "y1",
)


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for the randomized M1 data-generating process."""

    n_samples: int = 20_000
    n_features: int = 5
    treatment_probability: float = 0.5
    scenario: str = "randomized"
    seed: int = 42

    def validate(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.n_features != 5:
            raise ValueError("M1 currently requires exactly 5 features")
        if not 0.0 < self.treatment_probability < 1.0:
            raise ValueError("treatment_probability must be strictly between 0 and 1")
        if self.scenario not in {"randomized", "confounded", "weak_overlap"}:
            raise ValueError("scenario must be randomized, confounded, or weak_overlap")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_synthetic_data(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Generate a randomized binary-treatment, binary-outcome dataset.

    The DGP separates baseline response from treatment-effect heterogeneity.
    This becomes important in M2, where response ranking and uplift ranking
    will intentionally disagree.
    """

    config = config or SyntheticConfig()
    config.validate()
    rng = np.random.default_rng(config.seed)

    x = rng.normal(size=(config.n_samples, config.n_features))
    x1, x2, x3, x4, x5 = x.T

    # Untreated response probability. x1/x2 primarily drive baseline risk.
    baseline_logit = -1.0 + 0.9 * x1 - 0.7 * x2 + 0.25 * x5
    mu0 = _sigmoid(baseline_logit)

    # Heterogeneous treatment effect on the probability scale. x3/x4 primarily
    # drive treatment response, helping keep baseline risk and uplift distinct.
    raw_effect = 0.16 * np.tanh(0.9 * x3 - 0.7 * x4)
    # A small positive average effect makes the randomized ATE easy to inspect.
    desired_effect = 0.06 + raw_effect

    # Clip treated probabilities, then define tau from the realized probabilities
    # so the oracle identity tau == mu1 - mu0 is exact.
    mu1 = np.clip(mu0 + desired_effect, 0.01, 0.99)
    tau = mu1 - mu0

    if config.scenario == "randomized":
        propensity = np.full(config.n_samples, config.treatment_probability)
    elif config.scenario == "confounded":
        propensity = _sigmoid(0.9 * x1 - 0.7 * x2 + 0.25 * x5)
    else:
        # Deliberately weak overlap: assignment is much more deterministic.
        propensity = _sigmoid(2.8 * x1 - 2.2 * x2 + 0.8 * x5)
    treatment = rng.binomial(1, propensity)

    y0 = rng.binomial(1, mu0)
    y1 = rng.binomial(1, mu1)
    outcome = np.where(treatment == 1, y1, y0)

    data = pd.DataFrame(x, columns=[f"x{i}" for i in range(1, config.n_features + 1)])
    data.insert(0, "user_id", np.arange(config.n_samples))
    data["treatment"] = treatment
    data["outcome"] = outcome
    data["mu0_true"] = mu0
    data["mu1_true"] = mu1
    data["tau_true"] = tau
    data["propensity_true"] = propensity
    data["y0"] = y0
    data["y1"] = y1
    data["scenario"] = config.scenario
    data["seed"] = config.seed
    return data


def true_ate(data: pd.DataFrame) -> float:
    """Return the oracle average conditional treatment effect."""

    return float(data["tau_true"].mean())


def difference_in_means(data: pd.DataFrame) -> float:
    """Estimate ATE by treated-minus-control factual outcome means."""

    treated = data.loc[data["treatment"] == 1, "outcome"]
    control = data.loc[data["treatment"] == 0, "outcome"]
    if treated.empty or control.empty:
        raise ValueError("both treatment groups must contain observations")
    return float(treated.mean() - control.mean())


def model_feature_columns(data: pd.DataFrame) -> list[str]:
    """Return observed pre-treatment covariates and reject oracle leakage.

    The helper intentionally returns only x-columns. Treatment and outcome are
    factual variables used separately by estimators; they are not covariates.
    """

    oracle_present = set(ORACLE_COLUMNS).intersection(data.columns)
    # Oracle columns are expected in simulation data; they are explicitly not
    # selected. Keeping the set operation visible documents that boundary.
    _ = oracle_present
    return [column for column in data.columns if column.startswith("x") and column[1:].isdigit()]



def generate_randomized_data(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Backward-compatible helper that always uses randomized assignment."""
    config = config or SyntheticConfig()
    return generate_synthetic_data(SyntheticConfig(
        n_samples=config.n_samples, n_features=config.n_features,
        treatment_probability=config.treatment_probability,
        scenario="randomized", seed=config.seed,
    ))
