# src/modeling.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             mean_squared_error, r2_score)
from sklearn.pipeline import Pipeline
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# imbalanced-learn SMOTE
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

def _wrap_pipeline(preprocessor, fitted_clf):
    """Return Pipeline([('pre', pre), ('clf', fitted_clf)]) but without re-fitting"""
    pipe = Pipeline([("pre", preprocessor), ("clf", fitted_clf)])
    return pipe

def train_classifiers(preprocessor, X, y, random_state=42, use_xgb=True, smote_k=5):
    """
    Train models with SMOTE applied on the preprocessed numeric/categorical arrays.
    Returns dict: name -> (pipeline, X_test, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    models = {}

    # Fit preprocessor on X_train
    pre = preprocessor
    pre.fit(X_train)

    # transform training data
    try:
        X_train_t = pre.transform(X_train)
    except Exception as e:
        # If transform returns numpy array or sparse, ensure dense ndarray
        X_train_t = pre.transform(X_train).toarray() if hasattr(pre.transform(X_train), "toarray") else np.asarray(pre.transform(X_train))

    # Apply SMOTE if available and there is class imbalance
    if SMOTE is not None:
        try:
            sm = SMOTE(random_state=random_state, k_neighbors=smote_k)
            X_res, y_res = sm.fit_resample(X_train_t, y_train)
        except Exception as e:
            # fallback: no resampling if SMOTE fails
            X_res, y_res = X_train_t, y_train
    else:
        X_res, y_res = X_train_t, y_train

    # ------------- Logistic Regression (with class_weight as fallback) -------------
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    try:
        lr.fit(X_res, y_res)
    except Exception:
        # If LR can't fit dense arrays (rare), try without resampling fit on original
        lr.fit(X_train_t, y_train)
    models["logistic"] = (_wrap_pipeline(pre, lr), X_test, y_test)

    # ------------- Random Forest -------------
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state)
    rf.fit(X_res, y_res)
    models["random_forest"] = (_wrap_pipeline(pre, rf), X_test, y_test)

    # ------------- Optional XGBoost -------------
    if use_xgb:
        try:
            from xgboost import XGBClassifier
            xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=4, random_state=random_state)
            xgb.fit(X_res, y_res)
            models["xgboost"] = (_wrap_pipeline(pre, xgb), X_test, y_test)
        except Exception as e:
            print("XGBoost not available or failed to train:", e)

    return models

def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)[:,1]
    except Exception:
        # try decision_function
        try:
            y_proba = model.decision_function(X_test)
        except Exception:
            y_proba = None
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }

# ---------- Regression helpers left mostly unchanged ----------
def train_regressors(preprocessor, X, y, random_state=42, use_lgb=True):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    models = {}
    pipe_rf = Pipeline([("pre", preprocessor), ("reg", RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1))])
    pipe_rf.fit(X_train, y_train)
    models["random_forest"] = (pipe_rf, X_test, y_test)

    if use_lgb:
        try:
            from lightgbm import LGBMRegressor
            pipe_lgb = Pipeline([("pre", preprocessor), ("reg", LGBMRegressor(n_estimators=200, random_state=random_state))])
            pipe_lgb.fit(X_train, y_train)
            models["lightgbm"] = (pipe_lgb, X_test, y_test)
        except Exception as e:
            print("LightGBM not available:", e)
    return models

def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "r2": r2_score(y_test, y_pred),
        "mae": np.mean(np.abs(y_test - y_pred))
    }
