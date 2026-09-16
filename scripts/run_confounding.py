"""M5: demonstrate confounding bias."""
from counterfactuallab.data.synthetic import SyntheticConfig,difference_in_means,generate_synthetic_data,true_ate
from counterfactuallab.diagnostics.balance import max_absolute_smd,standardized_mean_differences


def report(scenario):
    d=generate_synthetic_data(SyntheticConfig(scenario=scenario,seed=42)); truth=true_ate(d); naive=difference_in_means(d)
    print(scenario.upper()); print(f"  True ATE:                    {truth:.4f}"); print(f"  Naive difference in means:   {naive:.4f}"); print(f"  Bias (naive - true):         {naive-truth:+.4f}"); print(f"  Max |covariate SMD|:         {max_absolute_smd(d):.3f}")
    for f,v in standardized_mean_differences(d).items(): print(f"    {f}: {v:+.3f}")


def main(): report("randomized"); print(); report("confounded")
if __name__ == "__main__": main()
