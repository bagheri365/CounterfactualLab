import pandas as pd
import pytest
from counterfactuallab.data.hillstrom import prepare_hillstrom
from counterfactuallab.data.synthetic import model_feature_columns


def fixture():
    return pd.DataFrame({"recency":[1,2,3,4],"history":[10.,20.,30.,40.],"mens":[1,0,1,0],"womens":[0,1,0,1],"zip_code":["Urban","Rural","Urban","Rural"],"newbie":[0,1,0,1],"channel":["Web","Phone","Web","Phone"],"segment":["Mens E-Mail","No E-Mail","Womens E-Mail","Mens E-Mail"],"conversion":[1,0,1,0]})


def test_prepare_hillstrom_keeps_requested_arm_and_control():
    d=prepare_hillstrom(fixture())
    assert len(d)==3
    assert d.treatment.tolist()==[1,0,1]
    assert d.outcome.tolist()==[1,0,0]


def test_hillstrom_features_are_pre_treatment_model_columns():
    d=prepare_hillstrom(fixture())
    features=model_feature_columns(d)
    assert features and all(c.startswith("x") for c in features)
    assert "treatment" not in features and "outcome" not in features


def test_prepare_hillstrom_has_no_synthetic_oracles():
    d=prepare_hillstrom(fixture())
    assert not {"tau_true","mu0_true","mu1_true","propensity_true","y0","y1"}.intersection(d.columns)


def test_prepare_hillstrom_requires_schema():
    with pytest.raises(ValueError): prepare_hillstrom(pd.DataFrame({"segment":["No E-Mail"]}))
