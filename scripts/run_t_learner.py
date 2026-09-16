"""M3: estimate CATE with a T-Learner and use it for targeting."""

from __future__ import annotations

from counterfactuallab.data.synthetic import SyntheticConfig, generate_randomized_data
from counterfactuallab.evaluation.cate import cate_correlation, cate_rmse
from counterfactuallab.evaluation.policy import (
    incremental_conversions_per_1000,
    top_k_mask,
    top_k_overlap,
)
from counterfactuallab.models.response import ResponseModel
from counterfactuallab.models.t_learner import TLearner


def main() -> None:
    data = generate_randomized_data(SyntheticConfig(n_samples=20_000, seed=42))
    tau_true = data["tau_true"].to_numpy()

    response_score = ResponseModel().fit(data).predict_response(data)
    learner = TLearner().fit(data)
    tau_hat = learner.predict_cate(data)

    k = int(0.20 * len(data))
    response_policy = top_k_mask(response_score, k)
    learned_policy = top_k_mask(tau_hat, k)
    oracle_policy = top_k_mask(tau_true, k)

    print(f"Users: {len(data):,}")
    print(f"Treatment budget: {k:,} ({k / len(data):.0%})")
    print()
    print("CATE estimation")
    print(f"  RMSE: {cate_rmse(tau_true, tau_hat):.4f}")
    print(f"  Correlation: {cate_correlation(tau_true, tau_hat):.3f}")
    print()
    print("Incremental conversions / 1,000 treatments")
    print(
        "  Response targeting: "
        f"{incremental_conversions_per_1000(data, response_policy):.2f}"
    )
    print(
        "  T-Learner targeting: "
        f"{incremental_conversions_per_1000(data, learned_policy):.2f}"
    )
    print(
        "  Oracle uplift:       "
        f"{incremental_conversions_per_1000(data, oracle_policy):.2f}"
    )
    print()
    print(
        "T-Learner / oracle top-K overlap: "
        f"{top_k_overlap(learned_policy, oracle_policy):.1%}"
    )


if __name__ == "__main__":
    main()
