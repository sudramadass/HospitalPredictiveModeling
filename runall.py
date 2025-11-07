# run_all.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from src import utils, preprocessing, modeling, evaluation, interpretability

def main(diabetic_csv, merged_csv, out_dir="outputs"):
    out = Path(out_dir); out.mkdir(exist_ok=True, parents=True)

    print("Loading datasets...")
    df_diab = utils.load_csv(diabetic_csv)
    df_merged = utils.load_csv(merged_csv)

    print("diabetic shape:", df_diab.shape)
    print("merged shape:", df_merged.shape)

    # --- Attempt to find matching patient id to merge ---
    print("Checking possible patient id overlap...")
    cand_diab = ["patient_nbr", "patient_id", "subject_id"]
    cand_merge = ["subject_id", "patient_nbr", "hadm_id", "hadm_id_x", "hadm_id_y"]
    diab_id = utils.try_parse_ids(df_diab, cand_diab)
    merged_id = utils.try_parse_ids(df_merged, cand_merge)
    print("diabetic id suggestion:", diab_id)
    print("merged id suggestion:", merged_id)

    merged = None
    if diab_id and merged_id and diab_id in df_diab.columns and merged_id in df_merged.columns:
        # check intersection size
        set_diab = set(df_diab[diab_id].dropna().unique())
        set_merged = set(df_merged[merged_id].dropna().unique())
        inter = set_diab.intersection(set_merged)
        print(f"Intersection of ids: {len(inter)}")
        if len(inter) > 0:
            print("Merging on", diab_id, "and", merged_id)
            merged = df_diab.merge(df_merged, left_on=diab_id, right_on=merged_id, how="left")
            print("merged shape:", merged.shape)
        else:
            print("No overlap — continuing with separate pipelines.")
    else:
        print("Could not find sensible join keys — continuing with separate pipelines.")

    # ---------------- Readmission pipeline (classification) -----------------
    print("\n=== READMISSION PIPELINE ===")
    pre_read, X_read, y_read = preprocessing.preprocess_for_readmission(df_diab)
    print("Building and training classifiers...")
    models = modeling.train_classifiers(pre_read, X_read, y_read, use_xgb=True)
    for name, (pipe, X_test, y_test) in models.items():
        print(f"\nModel: {name}")
        metrics = modeling.evaluate_classification(pipe, X_test, y_test)
        print(metrics)

        # simple plots
        preds = pipe.predict(X_test)
        try:
            evaluation.plot_confusion(y_test, preds, title=f"{name} confusion")
            evaluation.plot_roc(pipe, X_test, y_test)
        except Exception as e:
            print("Plotting failed:", e)

        # save model
        model_path = Path(out) / f"readmit_{name}.joblib"
        joblib.dump(pipe, model_path)
        print(f"Model saved to {model_path}")

        # 🔍 SHAP explainability for RandomForest
        if name == "random_forest":
            try:
                print("Generating SHAP explanation for RandomForest (this may take ~10–30s)...")
                # take a small random subset for SHAP analysis
                try:
                    X_for_shap = X_test.sample(min(200, len(X_test)), random_state=42)
                except Exception:
                    X_for_shap = X_test.iloc[:min(200, len(X_test))]

                from src import interpretability
                interpretability.explain_model_shap(
                    pipe,
                    X_for_shap,
                    out_path=f"{out}/shap_summary_{name}.png"
                )
            except Exception as e:
                print("SHAP generation failed:", e)

        # save model
        joblib.dump(pipe, Path(out)/f"readmit_{name}.joblib")

    # ---------------- LOS pipeline (regression) -----------------
    print("\n=== LOS PIPELINE ===")
    # If merged dataset has 'los' use it, otherwise compute
    df_los = df_merged.copy()
    pre_los, X_los, y_los = preprocessing.preprocess_for_los(df_los, id_cols=None)
    # filter invalid LOS
    valid_mask = y_los >= 0
    X_los = X_los[valid_mask]
    y_los = y_los[valid_mask]
    if len(y_los) < 20:
        print("Not enough LOS rows for training:", len(y_los))
    else:
        regs = modeling.train_regressors(pre_los, X_los, y_los, use_lgb=True)
        for name, (pipe, X_test, y_test) in regs.items():
            print(f"\nRegressor: {name}")
            metrics = modeling.evaluate_regression(pipe, X_test, y_test)
            print(metrics)
            joblib.dump(pipe, Path(out)/f"los_{name}.joblib")

    print("\nDone. Models and outputs saved to", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diabetic", required=True, help="Path to diabetic_data CSV")
    parser.add_argument("--merged", required=True, help="Path to merged_admissions_patients CSV")
    parser.add_argument("--out", default="outputs", help="Output directory")
    args = parser.parse_args()
    main(args.diabetic, args.merged, args.out)

    # ---------------- FAIRNESS ANALYSIS -----------------
print("\n=== FAIRNESS ANALYSIS ===")
try:
    from fairlearn.metrics import MetricFrame, accuracy_score, selection_rate, true_positive_rate

    # pick the RandomForest model for fairness assessment
    if "random_forest" in models:
        pipe, X_test, y_test = models["random_forest"]

        # we’ll use 'gender' and 'race' if available
        sensitive_features = {}
        for attr in ["gender", "race"]:
            if attr in X_test.columns:
                sensitive_features[attr] = X_test[attr]

        if len(sensitive_features) == 0:
            print("No sensitive columns (e.g., gender, race) found in test set.")
        else:
            y_pred = pipe.predict(X_test)

            for attr, values in sensitive_features.items():
                print(f"\nFairness by {attr}:")
                mf = MetricFrame(
                    metrics={"accuracy": accuracy_score,
                             "tpr": true_positive_rate,
                             "selection_rate": selection_rate},
                    y_true=y_test,
                    y_pred=y_pred,
                    sensitive_features=values
                )
                print(mf.by_group)

except Exception as e:
    print("Fairness analysis skipped:", e)

# --- save fairness results ---
import pandas as pd
all_frames = []
for attr, values in sensitive_features.items():
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score,
                 "tpr": true_positive_rate,
                 "selection_rate": selection_rate},
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=values
    )
    df_attr = mf.by_group.reset_index().rename(columns={"index": attr})
    df_attr.insert(0, "attribute", attr)
    all_frames.append(df_attr)

if all_frames:
    fairness_df = pd.concat(all_frames)
    fairness_path = Path(out) / "fairness_metrics.csv"
    fairness_df.to_csv(fairness_path, index=False)
    print(f"Fairness metrics saved to {fairness_path}")
