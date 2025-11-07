# src/interpretability.py
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def explain_model_shap(pipeline, X_raw, out_path="outputs/shap_summary.png", sample_size=200, max_display=20):
    """
    pipeline: sklearn Pipeline with named_steps 'pre' (preprocessor) and final estimator 'clf' or 'reg'
    X_raw: raw dataframe (or sample) to use for explanation (untransformed)
    Saves a SHAP summary plot to out_path and also tries to render inline.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # extract parts
    pre = pipeline.named_steps.get("pre", None)
    # final estimator can be named differently; pick last step
    last_step_name = list(pipeline.named_steps.keys())[-1]
    model = pipeline.named_steps.get(last_step_name)

    # prepare a sample
    if isinstance(X_raw, pd.DataFrame):
        X_sample = X_raw.sample(min(len(X_raw), sample_size), random_state=42)
    else:
        X_sample = pd.DataFrame(X_raw)

    # transform sample (for TreeExplainer we may prefer original features for some models,
    # but since our model was trained on preprocessed arrays, we explain on transformed space)
    try:
        X_trans = pre.transform(X_sample)
        # convert to dense array if needed
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
    except Exception as e:
        print("Preprocessor transform failed; trying to explain on raw X_sample:", e)
        X_trans = None

    try:
        # Tree models -> TreeExplainer on model and transformed matrix if available
        if X_trans is not None and (hasattr(model, "feature_importances_") or model.__class__.__name__.lower().startswith(("xgb", "lgb", "lightgbm", "randomforest"))):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)
            # shap.summary_plot expects raw or transformed arrays depending on explainer;
            plt.figure()
            shap.summary_plot(shap_values, X_trans, show=False, max_display=max_display)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            print("Saved SHAP summary to", out_path)
        else:
            # fallback to model-agnostic explainer on raw dataframe (may be slower)
            explainer = shap.Explainer(model, X_sample)
            sv = explainer(X_sample)
            plt.figure()
            shap.summary_plot(sv, X_sample, show=False, max_display=max_display)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            print("Saved SHAP summary to", out_path)
    except Exception as e:
        print("SHAP explanation failed:", e)
