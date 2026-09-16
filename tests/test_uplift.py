import numpy as np
import pandas as pd
import pytest
from counterfactuallab.evaluation.uplift import area_under_uplift_curve, uplift_curve


def test_curve_ignores_oracle_columns():
    d=pd.DataFrame({"treatment":[1,0,1,0,1,0,1,0],"outcome":[1,0,1,0,0,0,0,0],"tau_true":[999.]*8}); s=np.arange(8,0,-1.)
    pd.testing.assert_frame_equal(uplift_curve(d,s,n_bins=2),uplift_curve(d.drop(columns="tau_true"),s,n_bins=2))


def test_known_prefix_effect():
    d=pd.DataFrame({"treatment":[1,0,1,0,1,0,1,0],"outcome":[1,0,1,0,0,0,0,0]}); c=uplift_curve(d,np.arange(8,0,-1.),n_bins=2)
    assert c.iloc[0].estimated_effect==pytest.approx(1.0); assert c.iloc[1].estimated_effect==pytest.approx(.5)


def test_auuc():
    c=pd.DataFrame({"fraction":[.5,1.],"incremental_outcomes":[4.,4.]}); assert area_under_uplift_curve(c)==pytest.approx(3.)


def test_missing_group_rejected():
    d=pd.DataFrame({"treatment":[1,1,0,0],"outcome":[1,0,0,0]})
    with pytest.raises(ValueError,match="treated and control"): uplift_curve(d,np.array([4.,3.,2.,1.]),n_bins=2)
