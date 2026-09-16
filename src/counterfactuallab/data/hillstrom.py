"""Load and prepare the randomized Hillstrom email experiment."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

HILLSTROM_URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"


def prepare_hillstrom(raw: pd.DataFrame, *, treatment_segment: str = "Mens E-Mail") -> pd.DataFrame:
    """Create a binary email-vs-control learning dataset with conversion outcome."""
    required={"recency","history","mens","womens","zip_code","newbie","channel","segment","conversion"}
    missing=required-set(raw.columns)
    if missing: raise ValueError(f"missing Hillstrom columns: {sorted(missing)}")
    keep=raw.loc[raw["segment"].isin([treatment_segment,"No E-Mail"])].copy()
    if keep.empty: raise ValueError("requested treatment/control arms are absent")
    covariates=keep[["recency","history","mens","womens","zip_code","newbie","channel"]]
    encoded=pd.get_dummies(covariates,columns=["zip_code","channel"],dtype=float)
    encoded.columns=[f"x{i+1}" for i in range(encoded.shape[1])]
    data=encoded.reset_index(drop=True)
    data["treatment"]=keep["segment"].eq(treatment_segment).astype(int).to_numpy()
    data["outcome"]=keep["conversion"].astype(int).to_numpy()
    data["scenario"]="hillstrom_randomized"
    return data


def load_hillstrom(path: str | Path | None = None, *, treatment_segment: str = "Mens E-Mail") -> pd.DataFrame:
    """Read Hillstrom from a local CSV or its original public challenge URL."""
    raw=pd.read_csv(path if path is not None else HILLSTROM_URL)
    return prepare_hillstrom(raw,treatment_segment=treatment_segment)
