"""M4: observable uplift evaluation in a randomized experiment."""
import numpy as np
from counterfactuallab.data.synthetic import SyntheticConfig, generate_randomized_data
from counterfactuallab.evaluation.policy import incremental_conversions_per_1000, top_k_mask
from counterfactuallab.evaluation.uplift import area_under_uplift_curve, uplift_curve
from counterfactuallab.models.response import ResponseModel
from counterfactuallab.models.t_learner import TLearner


def main():
    data=generate_randomized_data(SyntheticConfig(n_samples=20_000,seed=42))
    rankings={"Response":ResponseModel().fit(data).predict_response(data),"T-Learner":TLearner().fit(data).predict_cate(data),"Random":np.random.default_rng(42).random(len(data))}
    print("Observable randomized-experiment evaluation")
    print("(uses treatment + factual outcome; does not use tau_true)\n")
    for name,scores in rankings.items():
        curve=uplift_curve(data,scores,n_bins=10)
        print(f"{name:10s} AUUC: {area_under_uplift_curve(curve):8.2f} | top 20% estimated effect: {curve.iloc[1]['estimated_effect']:.4f}")
    print("\nOracle sanity check (simulation only; uses tau_true)")
    k=int(.2*len(data))
    for name,scores in rankings.items():
        v=incremental_conversions_per_1000(data,top_k_mask(scores,k))
        print(f"{name:10s} true incremental conversions / 1,000: {v:.2f}")

if __name__=="__main__": main()
