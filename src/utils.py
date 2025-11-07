# src/utils.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

def load_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path)

def save_model(model, path):
    joblib.dump(model, path)

def compute_readmit_label(df, col="readmitted"):
    """
    Convert readmitted values (NO, <30, >30) to binary label:
    1 if readmitted within 30 days (<30), else 0.
    """
    s = df[col].astype(str).str.strip()
    return (s == "<30").astype(int)

def compute_los(df, admit_col="admittime", discharge_col="dischtime", los_col="los"):
    """
    Compute LOS (days) if not present. Expects parseable datetime strings.
    If 'los' exists, return as float.
    """
    if los_col in df.columns and not df[los_col].isnull().all():
        return df[los_col].astype(float)
    # parse and compute
    ad = pd.to_datetime(df[admit_col], errors="coerce")
    dis = pd.to_datetime(df[discharge_col], errors="coerce")
    los = (dis - ad).dt.total_seconds() / (24 * 3600)
    return los.fillna(-1)

def report_missing(df, max_rows=10):
    miss = df.isna().mean().sort_values(ascending=False)
    return miss[miss > 0].head(max_rows)

def try_parse_ids(df, candidates):
    """
    Return first column name that looks numeric and has many unique values from candidates.
    """
    for c in candidates:
        if c in df.columns:
            nunique = df[c].nunique(dropna=True)
            if nunique > 10:
                return c
    return None
