import numpy as np
from counterfactuallab.data.synthetic import SyntheticConfig,generate_synthetic_data
from counterfactuallab.evaluation.policy_value import oracle_incremental_value,oracle_policy_value
from counterfactuallab.models.response import ResponseModel
from counterfactuallab.models.t_learner import TLearner
from counterfactuallab.policies.oracle import oracle_policy
from counterfactuallab.policies.response import response_policy
from counterfactuallab.policies.uplift import uplift_policy


def test_policy_helpers_respect_budget():
    scores=np.arange(100,dtype=float)
    assert response_policy(scores,.2).sum()==20
    assert uplift_policy(scores,.2).sum()==20
    assert oracle_policy(scores,.2).sum()==20


def test_treat_none_has_zero_incremental_value():
    d=generate_synthetic_data(SyntheticConfig(n_samples=2000,seed=91)); p=np.zeros(len(d),dtype=bool)
    assert abs(oracle_incremental_value(d,p))<1e-12


def test_oracle_value_matches_policy_potential_outcomes():
    d=generate_synthetic_data(SyntheticConfig(n_samples=2000,seed=92)); p=np.arange(len(d))%2==0
    expected=np.where(p,d.mu1_true,d.mu0_true).mean(); assert np.isclose(oracle_policy_value(d,p),expected)


def test_uplift_policy_beats_response_policy_in_learning_dgp():
    d=generate_synthetic_data(SyntheticConfig(n_samples=20000,scenario="randomized",seed=42)); r=ResponseModel().fit(d).predict_response(d); c=TLearner().fit(d).predict_cate(d)
    rp=response_policy(r,.2); up=uplift_policy(c,.2)
    assert oracle_incremental_value(d,up)>oracle_incremental_value(d,rp)
