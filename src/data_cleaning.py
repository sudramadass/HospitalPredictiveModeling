"""
Complete Data Cleaning and Feature Selection Pipeline
1. Load raw diabetic_data.csv
2. Clean data according to standard data mining procedures
3. Run chi-squared analysis on all features vs 'readmitted'
4. Keep only statistically significant features
5. Save fully processed dataset
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# STEP 1: Load Raw Data
# ============================================================================
df = pd.read_csv('../data/raw/diabetic_data.csv')

# ============================================================================
# STEP 2: Data Cleaning
# ============================================================================

# 2.1: Remove low-quality columns
# Remove weight (97% missing), payer_code (redundant), medical_specialty (high cardinality)
# Remove individual diagnosis codes (too granular)
remove_cols = ['weight', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3']
remove_cols = [col for col in remove_cols if col in df.columns]
df = df.drop(columns=remove_cols)

# 2.2: Remove individual medication columns (keep summary features instead)
medication_cols = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
]
medication_cols = [col for col in medication_cols if col in df.columns]
df = df.drop(columns=medication_cols)

# 2.3: Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].replace('?', 'Unknown')
        df[col] = df[col].replace('', 'Unknown')
        df[col] = df[col].fillna('Unknown')

# 2.4: Create binary indicators for lab results with high missing rates
df['has_glu_serum'] = (~df['max_glu_serum'].isin(['Unknown', ''])).astype(int)
df['has_A1C'] = (~df['A1Cresult'].isin(['Unknown', ''])).astype(int)

# ============================================================================
# STEP 3: Chi-Squared Feature Selection
# ============================================================================

target = 'readmitted'
keep_cols = ['patient_nbr', target]
feature_cols = [col for col in df.columns if col not in keep_cols]

# Prepare data for chi-squared test
df_test = df.copy()
for col in df_test.columns:
    if df_test[col].dtype == 'object':
        df_test[col] = df_test[col].fillna('Missing').replace('?', 'Missing').replace('', 'Missing')
    else:
        df_test[col] = df_test[col].fillna(-999)

# Run chi-squared test for each feature
results = []
for col in feature_cols:
    try:
        if df_test[col].dtype in ['int64', 'float64']:
            n_bins = min(10, df_test[col].nunique())
            if n_bins > 1:
                col_binned = pd.qcut(df_test[col], q=n_bins, duplicates='drop')
            else:
                col_binned = df_test[col]
        else:
            col_binned = df_test[col]
        
        contingency_table = pd.crosstab(col_binned, df_test[target])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        results.append({
            'feature': col,
            'chi2': chi2,
            'p_value': p_value,
            'cramers_v': cramers_v,
            'significant': p_value < 0.05
        })
        
    except Exception as e:
        results.append({
            'feature': col,
            'chi2': np.nan,
            'p_value': np.nan,
            'cramers_v': np.nan,
            'significant': False
        })

results_df = pd.DataFrame(results)
significant_features = results_df[results_df['significant'] == True]['feature'].tolist()

# Keep only significant features
columns_to_keep = keep_cols + significant_features
df_filtered = df[columns_to_keep].copy()

# Sort by patient_nbr
df_filtered = df_filtered.sort_values('patient_nbr').reset_index(drop=True)

# ============================================================================
# STEP 4: Encode Categorical Variables
# ============================================================================

df_encoded = df_filtered.copy()

# Ordinal encoding for age (maintain order)
age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
if 'age' in df_encoded.columns:
    df_encoded['age'] = df_encoded['age'].map({age: i for i, age in enumerate(age_order)})

# Binary encoding for gender
if 'gender' in df_encoded.columns:
    df_encoded['gender'] = df_encoded['gender'].map({'Male': 1, 'Female': 0, 'Unknown/Invalid': -1, 'Unknown': -1})

# Label encoding for remaining categorical variables
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object' and col not in keep_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# ============================================================================
# STEP 5: Display Correlation Results
# ============================================================================

print("\n" + "="*80)
print("CHI-SQUARED CORRELATION ANALYSIS RESULTS")
print("="*80)
print(f"\nTotal features tested: {len(results_df)}")
print(f"Significant features (p < 0.05): {len(significant_features)}")
print(f"Non-significant features: {len(results_df) - len(significant_features)}")

print("\n" + "="*80)
print("ALL FEATURES - RANKED BY CORRELATION STRENGTH (Cramér's V)")
print("="*80)
results_sorted = results_df.sort_values('cramers_v', ascending=False)
print(f"\n{'Feature':<30} {'Chi²':>12} {'p-value':>12} {'Cramér\'s V':>12} {'Significant':>12}")
print("-"*80)
for idx, row in results_sorted.iterrows():
    sig_marker = '✓' if row['significant'] else '✗'
    print(f"{row['feature']:<30} {row['chi2']:>12.2f} {row['p_value']:>12.6f} {row['cramers_v']:>12.4f} {sig_marker:>12}")

# ============================================================================
# STEP 6: Save Final Processed Dataset
# ============================================================================

output_path = '../data/processed/diabetic_data_final.csv'
df_encoded.to_csv(output_path, index=False)

# Save feature selection results
results_path = '../data/processed/feature_selection_results.csv'
results_sorted.to_csv(results_path, index=False)

print("\n" + "="*80)
print(f"Saved: {output_path}")
print(f"Saved: {results_path}")
print("="*80)
