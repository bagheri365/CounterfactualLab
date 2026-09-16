"""M8: demonstrate weak-overlap instability."""
import numpy as np
from counterfactuallab.data.synthetic import SyntheticConfig,generate_synthetic_data,true_ate
from counterfactuallab.diagnostics.overlap import overlap_summary
from counterfactuallab.evaluation.ipw import ipw_ate
from counterfactuallab.models.dr_learner import cross_fitted_aipw_ate
from counterfactuallab.models.propensity import PropensityModel


def estimates(scenario,seed=42,n=20000):
    d=generate_synthetic_data(SyntheticConfig(n_samples=n,scenario=scenario,seed=seed)); e=PropensityModel().fit(d).predict(d); return d,e,ipw_ate(d,e),cross_fitted_aipw_ate(d,seed=seed)


def report(scenario):
    d,e,ipw,dr=estimates(scenario); s=overlap_summary(d,e)
    print(scenario.upper()); print(f"  True ATE:                    {true_ate(d):.4f}"); print(f"  IPW ATE:                     {ipw:.4f}"); print(f"  Cross-fitted AIPW/DR ATE:    {dr:.4f}"); print(f"  Propensity range:             {s['min_propensity']:.4f} to {s['max_propensity']:.4f}"); print(f"  Fraction e<.05 or e>.95:      {s['extreme_fraction']:.1%}"); print(f"  Largest IPW weight:           {s['max_weight']:.1f}"); print(f"  Effective sample size:        {s['effective_sample_size']:.0f} / {len(d)} ({s['effective_sample_fraction']:.1%})")
    vals=[estimates(scenario,i,10000)[2:] for i in range(10)]; print(f"  Across 10 seeds, IPW SD:      {np.std([v[0] for v in vals],ddof=1):.4f}"); print(f"  Across 10 seeds, DR SD:       {np.std([v[1] for v in vals],ddof=1):.4f}")


def main():
    report("confounded"); print(); report("weak_overlap"); print(); print("Better prediction cannot create missing treatment support.")
if __name__=="__main__": main()
