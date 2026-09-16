"""M11: evaluate uplift on the real randomized Hillstrom experiment."""
import numpy as np
from counterfactuallab.data.hillstrom import load_hillstrom
from counterfactuallab.evaluation.uplift import area_under_uplift_curve,uplift_curve
from counterfactuallab.models.response import ResponseModel
from counterfactuallab.models.t_learner import TLearner


def main():
    data=load_hillstrom("data/raw/hillstrom.csv")
    response=ResponseModel().fit(data).predict_response(data)
    cate=TLearner().fit(data).predict_cate(data)
    rng=np.random.default_rng(42); random_scores=rng.random(len(data))
    print("HILLSTROM: MEN'S EMAIL VS NO EMAIL | outcome=conversion")
    print(f"Users: {len(data):,} | treated: {data.treatment.sum():,} | control: {(1-data.treatment).sum():,}")
    print(f"Observed experiment effect: {data.loc[data.treatment.eq(1),'outcome'].mean()-data.loc[data.treatment.eq(0),'outcome'].mean():.4f}")
    print(); print("Observable uplift evaluation (no tau_true)")
    for name,scores in [("Response",response),("T-Learner",cate),("Random",random_scores)]:
        curve=uplift_curve(data,scores,n_bins=10); auuc=area_under_uplift_curve(curve)
        top=curve.iloc[1]["estimated_effect"]
        print(f"  {name:<9} AUUC: {auuc:8.2f} | top 20% estimated effect: {top:.4f}")
    print(); print("Real data provide factual randomized outcomes, not individual counterfactuals.")
    print("Therefore tau_true, PEHE, and oracle policy value are unavailable.")


if __name__=="__main__": main()
