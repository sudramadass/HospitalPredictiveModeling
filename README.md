# 🏥 Hospital Readmission & Length-of-Stay Prediction

This project develops and evaluates **machine learning models** to predict **hospital readmission** and **length of stay (LOS)**.  
It also explores **why** models make certain predictions using explainability tools such as **SHAP**, **feature importance**, and **fairness analysis**.

---

## 📘 Project Overview

Hospitals face challenges in managing readmissions and understanding patient risk factors.  
This project builds a robust predictive pipeline that:

1. **Preprocesses patient data**
   - Handles missing values
   - Encodes categorical variables
   - Balances imbalanced data
   - Performs feature scaling and engineering  

2. **Trains and compares multiple models**
   - Logistic Regression, Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost)
   - Evaluates using F1, AUC, and calibration metrics
   - Performs hyperparameter optimization with Optuna

3. **Explains predictions**
   - Global feature importance
   - SHAP plots for local and global interpretability
   - Fairness analysis across sensitive attributes (e.g., age, gender)

---

## 🧱 Project Structure

```
hospital-readmission/
├── src/
│   ├── preprocessing.py        # Data cleaning, encoding, scaling
│   ├── modeling.py             # Model training, tuning, evaluation
│   ├── interpretability.py     # SHAP, feature importance, explainability
│   ├── evaluation.py           # Confusion matrix, ROC, calibration plots
│   └── utils.py                # Helper functions
│
├── data/                       # Raw and processed datasets (excluded in .gitignore)
├── reports/                    # Saved plots, metrics, feature importances
├── environment.yml             # Conda environment specification
├── requirements.txt            # Optional pip-based dependency list
└── README.md                   # Project documentation (this file)
```

---

## ⚙️ Environment Setup

### 1. Create and activate the environment
```bash
conda env create -f environment.yml
conda activate hospital-readmission
```

### 2. Verify installation
```bash
python -c "import sklearn, shap, pandas; print('Environment OK!')"
```

If you don’t have `conda`, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.

---

## 📦 Dependencies

Core packages included:
- **pandas**, **numpy**, **scipy**, **scikit-learn**, **matplotlib**, **seaborn**

Additional ML / Explainability tools:
- **xgboost**, **lightgbm**, **catboost**
- **imbalanced-learn**
- **shap**, **eli5**
- **fairlearn**
- **optuna**
- **statsmodels**, **tqdm**

All are captured in `environment.yml` for reproducibility.

---

## 🧮 Workflow

### **1️⃣ Data Preprocessing**
- Load data → handle missing values (`SimpleImputer`, `KNNImputer`)
- Encode categorical variables (OneHot, Label)
- Scale features (StandardScaler / MinMaxScaler)
- Handle imbalance (SMOTE / class weights)
- Split into train / validation / test

### **2️⃣ Model Development**
- Train Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- Use consistent evaluation metrics (Accuracy, F1, AUC)
- Optimize hyperparameters (GridSearchCV or Optuna)
- Save best model with `joblib`

### **3️⃣ Evaluation**
- Plot confusion matrix, ROC, Precision-Recall, and calibration curves
- Analyze false positives/negatives
- Compare models across metrics

### **4️⃣ Model Interpretability**
- Compute feature importances (tree-based, permutation)
- Generate SHAP summary, beeswarm, and dependency plots
- Local explanations for individual patients

### **5️⃣ Fairness & Ethics**
- Use `fairlearn` to assess bias across subgroups
- Report disparate impact or performance gaps
- Discuss ethical implications in healthcare ML

---

## 📊 Example Results (Expected)

| Model | Accuracy | F1 | AUC | Key Drivers |
|--------|----------|----|-----|--------------|
| Logistic Regression | 0.71 | 0.66 | 0.73 | Age, Comorbidity Index |
| Random Forest | 0.78 | 0.74 | 0.81 | Lab Abnormalities, Prior Visits |
| XGBoost | 0.80 | 0.76 | 0.85 | Length of stay, Medications Count |

(Results will vary depending on dataset.)

---

## 📈 Example Visuals

- SHAP Summary Plot → global feature importance  
- Confusion Matrix → model performance overview  
- Fairness Metrics → performance by subgroup  

_All saved under `/reports/`._

---

## 🧪 Reproducibility

To exactly reproduce the working environment:
```bash
conda remove --name hospital-readmission --all -y
conda env create -f environment.yml
conda activate hospital-readmission
```

To update dependencies later:
```bash
conda env export --no-builds > environment.yml
```

---

## 🧑‍⚖️ Ethics Note

This project involves **healthcare prediction**, so:
- Sensitive attributes (age, gender, race) should be handled responsibly.
- Results are for **research and educational use only**, not clinical deployment.

---

## 🤝 Contributing

Pull requests are welcome!  
If you add new dependencies, please re-export the environment:
```bash
conda env export --no-builds > environment.yml
```

---

## 🧾 License

This project is released under the **MIT License**.  
See `LICENSE` for details.

