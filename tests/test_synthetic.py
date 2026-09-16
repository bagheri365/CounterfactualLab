import numpy as np
import pandas as pd
import pytest

from counterfactuallab.data.synthetic import (
    ORACLE_COLUMNS,
    SyntheticConfig,
    difference_in_means,
    generate_randomized_data,
    model_feature_columns,
    true_ate,
)


def test_generation_is_reproducible() -> None:
    config = SyntheticConfig(n_samples=500, seed=7)
    first = generate_randomized_data(config)
    second = generate_randomized_data(config)
    pd.testing.assert_frame_equal(first, second)


def test_probabilities_and_oracle_identity_are_valid() -> None:
    data = generate_randomized_data(SyntheticConfig(n_samples=1_000))

    for column in ("mu0_true", "mu1_true", "propensity_true"):
        assert data[column].between(0.0, 1.0).all()

    np.testing.assert_allclose(
        data["tau_true"],
        data["mu1_true"] - data["mu0_true"],
        rtol=0.0,
        atol=1e-12,
    )


def test_observed_outcome_is_factual_potential_outcome() -> None:
    data = generate_randomized_data(SyntheticConfig(n_samples=1_000))
    expected = np.where(data["treatment"].eq(1), data["y1"], data["y0"])
    np.testing.assert_array_equal(data["outcome"].to_numpy(), expected)


def test_randomized_treatment_is_approximately_balanced() -> None:
    data = generate_randomized_data(SyntheticConfig(n_samples=20_000, seed=42))
    assert data["treatment"].mean() == pytest.approx(0.5, abs=0.02)


def test_randomized_difference_in_means_recovers_true_ate() -> None:
    errors = []
    for seed in range(20):
        data = generate_randomized_data(
            SyntheticConfig(n_samples=20_000, seed=seed)
        )
        errors.append(difference_in_means(data) - true_ate(data))

    # Repeated randomization should center the observable estimator near truth.
    assert abs(float(np.mean(errors))) < 0.01
    assert float(np.mean(np.abs(errors))) < 0.02


def test_model_features_exclude_oracle_and_post_assignment_columns() -> None:
    data = generate_randomized_data(SyntheticConfig(n_samples=100))
    features = model_feature_columns(data)

    assert features == ["x1", "x2", "x3", "x4", "x5"]
    assert not set(ORACLE_COLUMNS).intersection(features)
    assert "treatment" not in features
    assert "outcome" not in features


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_samples": 0}, "n_samples"),
        ({"n_features": 4}, "exactly 5"),
        ({"treatment_probability": 0.0}, "strictly between"),
        ({"treatment_probability": 1.0}, "strictly between"),
    ],
)
def test_invalid_config_is_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate_randomized_data(SyntheticConfig(**kwargs))
