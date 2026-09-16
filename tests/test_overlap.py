import numpy as np
from counterfactuallab.data.synthetic import SyntheticConfig,generate_synthetic_data
from counterfactuallab.diagnostics.overlap import effective_sample_size,overlap_summary
from counterfactuallab.models.propensity import PropensityModel


def test_equal_weights_have_full_ess(): assert effective_sample_size(np.ones(100))==100.0

def test_weak_overlap_has_more_extreme_propensities():
    a=generate_synthetic_data(SyntheticConfig(n_samples=20000,scenario="confounded",seed=71)); b=generate_synthetic_data(SyntheticConfig(n_samples=20000,scenario="weak_overlap",seed=71)); ae=((a.propensity_true<.05)|(a.propensity_true>.95)).mean(); be=((b.propensity_true<.05)|(b.propensity_true>.95)).mean(); assert be>ae+.20

def test_weak_overlap_reduces_effective_sample_size():
    a=generate_synthetic_data(SyntheticConfig(n_samples=20000,scenario="confounded",seed=72)); b=generate_synthetic_data(SyntheticConfig(n_samples=20000,scenario="weak_overlap",seed=72)); ea=PropensityModel().fit(a).predict(a); eb=PropensityModel().fit(b).predict(b); sa=overlap_summary(a,ea); sb=overlap_summary(b,eb); assert sb["effective_sample_fraction"]<sa["effective_sample_fraction"]; assert sb["max_weight"]>sa["max_weight"]

def test_outcome_world_is_unchanged():
    a=generate_synthetic_data(SyntheticConfig(n_samples=2000,scenario="confounded",seed=73)); b=generate_synthetic_data(SyntheticConfig(n_samples=2000,scenario="weak_overlap",seed=73));
    for c in ["mu0_true","mu1_true","tau_true","y0","y1"]: assert a[c].equals(b[c])
