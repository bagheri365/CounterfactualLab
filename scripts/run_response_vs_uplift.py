"""M2: compare response targeting with oracle uplift targeting."""

from __future__ import annotations

import numpy as np

from counterfactuallab.data.synthetic import SyntheticConfig, generate_randomized_data
from counterfactuallab.evaluation.policy import (
    incremental_conversions_per_1000,
    top_k_mask,
    top_k_overlap,
)
from counterfactuallab.models.response import ResponseModel


def main() -> None:
    data = generate_randomized_data(SyntheticConfig(n_samples=20_000, seed=42))

    model = ResponseModel().fit(data)
    response_score = model.predict_response(data)
    uplift_score = data["tau_true"].to_numpy()

    # Equal 20% treatment budget makes the ranking comparison concrete.
    k = int(0.20 * len(data))
    response_policy = top_k_mask(response_score, k)
    oracle_uplift_policy = top_k_mask(uplift_score, k)

    response_value = incremental_conversions_per_1000(data, response_policy)
    uplift_value = incremental_conversions_per_1000(data, oracle_uplift_policy)

    response_baseline = float(data.loc[response_policy, "mu0_true"].mean())
    uplift_baseline = float(data.loc[oracle_uplift_policy, "mu0_true"].mean())

    print(f"Users: {len(data):,}")
    print(f"Treatment budget: {k:,} ({k / len(data):.0%})")
    print(f"Top-K overlap: {top_k_overlap(response_policy, oracle_uplift_policy):.1%}")
    print()
    print("Response targeting")
    print(f"  Mean untreated baseline: {response_baseline:.4f}")
    print(f"  Incremental conversions / 1,000: {response_value:.2f}")
    print()
    print("Oracle uplift targeting")
    print(f"  Mean untreated baseline: {uplift_baseline:.4f}")
    print(f"  Incremental conversions / 1,000: {uplift_value:.2f}")
    print()
    print(f"Oracle uplift advantage / 1,000: {uplift_value - response_value:.2f}")

    corr = np.corrcoef(response_score, uplift_score)[0, 1]
    print(f"Response-score / true-uplift correlation: {corr:.3f}")


if __name__ == "__main__":
    main()
