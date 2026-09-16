import numpy as np
from counterfactuallab.data.synthetic import SyntheticConfig,difference_in_means,generate_synthetic_data,true_ate
from counterfactuallab.diagnostics.balance import max_absolute_smd


def test_confounded_propensity_varies():
    d=generate_synthetic_data(SyntheticConfig(n_samples=5000,scenario="confounded",seed=31)); assert d.propensity_true.std()>.10


def test_outcome_world_is_unchanged():
    a=generate_synthetic_data(SyntheticConfig(n_samples=2000,scenario="randomized",seed=32)); b=generate_synthetic_data(SyntheticConfig(n_samples=2000,scenario="confounded",seed=32))
    for c in ["mu0_true","mu1_true","tau_true","y0","y1"]: np.testing.assert_allclose(a[c],b[c])


def test_confounding_creates_imbalance():
    a=generate_synthetic_data(SyntheticConfig(n_samples=20000,scenario="randomized",seed=33)); b=generate_synthetic_data(SyntheticConfig(n_samples=20000,scenario="confounded",seed=33)); assert max_absolute_smd(a)<.10; assert max_absolute_smd(b)>.40


def test_naive_difference_is_biased():
    d=generate_synthetic_data(SyntheticConfig(n_samples=40000,scenario="confounded",seed=34)); assert abs(difference_in_means(d)-true_ate(d))>.10
