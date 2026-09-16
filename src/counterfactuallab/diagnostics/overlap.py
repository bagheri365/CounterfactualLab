"""Diagnostics for overlap and inverse-weight concentration."""
import numpy as np
from counterfactuallab.evaluation.ipw import ipw_weights


def effective_sample_size(weights):
    w=np.asarray(weights,dtype=float)
    if w.ndim!=1 or len(w)==0: raise ValueError("weights must be a non-empty one-dimensional array")
    if np.any(w<0): raise ValueError("weights must be nonnegative")
    return float(w.sum()**2/np.sum(w**2))


def overlap_summary(data, propensity, threshold=.05):
    e=np.asarray(propensity,dtype=float)
    if len(e)!=len(data): raise ValueError("propensity must have one value per row")
    w=ipw_weights(data["treatment"],e); ess=effective_sample_size(w); extreme=(e<threshold)|(e>1-threshold)
    return {"min_propensity":float(e.min()),"max_propensity":float(e.max()),"extreme_fraction":float(extreme.mean()),"max_weight":float(w.max()),"effective_sample_size":ess,"effective_sample_fraction":ess/len(data)}
