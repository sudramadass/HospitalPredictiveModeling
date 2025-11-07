# src/preprocessing.py
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

def split_num_cat(df, exclude=[]):
    df2 = df.drop(columns=exclude, errors="ignore")
    num = df2.select_dtypes(include=["number"]).columns.tolist()
    cat = df2.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num, cat

def _make_onehot(handle_unknown="ignore"):
    """
    Construct a OneHotEncoder instance that works across scikit-learn versions.
    Newer sklearn uses 'sparse_output', older versions use 'sparse'.
    """
    try:
        # try modern argument name
        return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
    except TypeError:
        # fallback for older sklearn
        return OneHotEncoder(handle_unknown=handle_unknown, sparse=False)

def build_tabular_pipeline(numeric_features, categorical_features):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", _make_onehot())
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ], remainder="drop", sparse_threshold=0)
    return preprocessor

def preprocess_for_readmission(df, id_cols=None, drop_cols=None, target_col="readmitted"):
    """
    Returns preprocessor, X, y
    """
    if id_cols is None: id_cols = ["encounter_id", "patient_nbr"]
    if drop_cols is None: drop_cols = []
    df = df.copy()
    # create binary label
    from .utils import compute_readmit_label
    y = compute_readmit_label(df, col=target_col)
    # drop target and ids
    X = df.drop(columns=[target_col] + id_cols + drop_cols, errors="ignore")
    num, cat = split_num_cat(X)
    pre = build_tabular_pipeline(num, cat)
    return pre, X, y

def preprocess_for_los(df, id_cols=None, drop_cols=None, los_col="los"):
    if id_cols is None: id_cols = ["subject_id", "hadm_id", "stay_id"]
    if drop_cols is None: drop_cols = []
    df = df.copy()
    from .utils import compute_los
    y = compute_los(df, los_col=los_col)
    X = df.drop(columns=[los_col] + id_cols + drop_cols, errors="ignore")
    num, cat = split_num_cat(X)
    pre = build_tabular_pipeline(num, cat)
    return pre, X, y
