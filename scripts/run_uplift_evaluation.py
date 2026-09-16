"""M4: honest observable uplift evaluation in a randomized experiment."""
import numpy as np
from counterfactuallab.data.synthetic import SyntheticConfig, generate_randomized_data
from counterfactuallab.evaluation.cross_fitting import cross_fitted_ranking_scores
from counterfactuallab.evaluation.policy import incremental_conversions_per_1000, top_k_mask
from counterfactuallab.evaluation.uplift import area_under_uplift_curve, uplift_curve


def main():
    data=generate_randomized_data(SyntheticConfig(n_samples=20_000,seed=42))
    response,cate=cross_fitted_ranking_scores(data,n_splits=5,seed=42)
    rankings={"Response":response,"T-Learner":cate,"Random":np.random.default_rng(42).random(len(data))}
    print("Observable randomized-experiment evaluation")
    print("(response and T-Learner scores are out of fold; evaluation uses factual outcomes)\n")
    for name,scores in rankings.items():
        curve=uplift_curve(data,scores,n_bins=10)
        print(f"{name:10s} AUUC: {area_under_uplift_curve(curve):8.2f} | top 20% estimated effect: {curve.iloc[1]['estimated_effect']:.4f}")
    print("\nOracle sanity check (simulation only; uses tau_true)")
    k=int(.2*len(data))
    for name,scores in rankings.items():
        v=incremental_conversions_per_1000(data,top_k_mask(scores,k))
        print(f"{name:10s} true incremental conversions / 1,000: {v:.2f}")

if __name__=="__main__": main()
