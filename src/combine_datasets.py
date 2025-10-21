"""
Combine Datasets Script
Combines diabetic_data_cleaned.csv and admissions_cleaned.csv into a single unified dataset.
"""

import pandas as pd
import numpy as np


diabetic_df = pd.read_csv('../data/processed/diabetic_data_cleaned.csv')
admissions_df = pd.read_csv('../data/processed/admissions_cleaned.csv')

# Unify column names so they can be merged properly
admissions_df = admissions_df.rename(columns={
    'subject_id': 'patient_id',
    'hadm_id': 'encounter_id',
    'length_of_stay_days': 'time_in_hospital'
})


diabetic_df = diabetic_df.rename(columns={
    'patient_nbr': 'patient_id'
})


admission_type_map = {
    1: 'Emergency',
    2: 'Urgent',
    3: 'Elective',
    4: 'Newborn',
    5: 'Not Available',
    6: 'NULL',
    7: 'Trauma Center',
    8: 'Not Mapped'
}

diabetic_df['admission_type'] = diabetic_df['admission_type_id'].map(admission_type_map)


admissions_type_map = {
    'EW EMER.': 'Emergency',
    'URGENT': 'Urgent',
    'ELECTIVE': 'Elective',
    'DIRECT EMER.': 'Emergency',
    'EU OBSERVATION': 'Emergency',
    'OBSERVATION ADMIT': 'Emergency',
    'AMBULATORY OBSERVATION': 'Emergency',
    'DIRECT OBSERVATION': 'Emergency',
    'SURGICAL SAME DAY ADMISSION': 'Elective'
}

admissions_df['admission_type'] = admissions_df['admission_type'].map(
    lambda x: admissions_type_map.get(x, x) if pd.notna(x) else x
)


# Group by patient_id and sort by encounter_id to identify readmissions
admissions_df = admissions_df.sort_values(['patient_id', 'encounter_id'])

# For simplicity, mark patients with multiple admissions
patient_admission_counts = admissions_df.groupby('patient_id').size()
multiple_admissions = patient_admission_counts[patient_admission_counts > 1].index

admissions_df['readmitted'] = admissions_df['patient_id'].apply(
    lambda x: '>30' if x in multiple_admissions else 'NO'
)

diabetic_df['time_in_hospital'] = pd.to_numeric(diabetic_df['time_in_hospital'], errors='coerce')
admissions_df['time_in_hospital'] = pd.to_numeric(admissions_df['time_in_hospital'], errors='coerce')

# Round admissions time_in_hospital to match diabetic data (whole days)
admissions_df['time_in_hospital'] = admissions_df['time_in_hospital'].round()


# Common columns that exist in both datasets
common_cols = ['patient_id', 'encounter_id', 'race', 'admission_type', 
               'time_in_hospital', 'readmitted']

# Get columns unique to diabetic data
diabetic_only_cols = ['gender', 'age', 'admission_type_id', 'discharge_disposition_id', 
                      'admission_source_id', 'num_lab_procedures', 'num_procedures', 
                      'num_medications', 'number_outpatient', 'number_emergency', 
                      'number_inpatient', 'number_diagnoses', 'max_glu_serum', 
                      'A1Cresult', 'insulin', 'change', 'diabetesMed']

# Get columns unique to admissions data
admissions_only_cols = ['admission_location', 'discharge_location', 'insurance',
                        'language', 'marital_status', 'hospital_expire_flag']

# Add missing columns with NaN values

for col in admissions_only_cols:
    if col not in diabetic_df.columns:
        diabetic_df[col] = np.nan

for col in diabetic_only_cols:
    if col not in admissions_df.columns:
        admissions_df[col] = np.nan


diabetic_df['data_source'] = 'diabetic.csv'
admissions_df['data_source'] = 'admissions.csv'

#combination of datasets

# Select all columns for combining
all_columns = (common_cols + diabetic_only_cols + admissions_only_cols + ['data_source'])

# Select columns from each dataset
diabetic_selected = diabetic_df[all_columns]
admissions_selected = admissions_df[all_columns]

# Concatenate vertically
combined_df = pd.concat([diabetic_selected, admissions_selected], axis=0, ignore_index=True)

print(f"Combined dataset shape: {combined_df.shape}")

output_file = '../data/processed/combined_data.csv'
combined_df.to_csv(output_file, index=False)

print(f"\n✓ Combined dataset saved to: {output_file}")


