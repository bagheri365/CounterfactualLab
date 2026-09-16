"""M10: turn response and CATE scores into treatment policies."""
from counterfactuallab.data.synthetic import SyntheticConfig,generate_synthetic_data
from counterfactuallab.evaluation.policy import top_k_overlap
from counterfactuallab.evaluation.policy_value import oracle_incremental_value,oracle_policy_value
from counterfactuallab.models.response import ResponseModel
from counterfactuallab.models.t_learner import TLearner
from counterfactuallab.policies.oracle import oracle_policy
from counterfactuallab.policies.response import response_policy
from counterfactuallab.policies.uplift import uplift_policy


def main():
    data=generate_synthetic_data(SyntheticConfig(n_samples=20000,scenario="randomized",seed=42))
    response=ResponseModel().fit(data).predict_response(data)
    cate=TLearner().fit(data).predict_cate(data)
    truth=data["tau_true"].to_numpy()
    print("CATE -> TARGETING POLICY (randomized synthetic data)")
    for budget in (.10,.20,.40):
        rp=response_policy(response,budget); up=uplift_policy(cate,budget); op=oracle_policy(truth,budget)
        print(); print(f"Budget: {budget:.0%} ({rp.sum()} users)")
        for name,p in [("Response",rp),("T-Learner uplift",up),("Oracle uplift",op)]:
            value=oracle_policy_value(data,p); gain=oracle_incremental_value(data,p)
            print(f"  {name:<17} policy value: {value:.4f} | gain/user: {gain:.4f}")
        print(f"  Uplift/oracle overlap: {top_k_overlap(up,op):.1%}")
    print(); print("Oracle policy/value use simulation truth only; deployable policies do not see tau_true.")


if __name__=="__main__": main()
