

import pandas as pd
import numpy as np
from datetime import datetime

diabetic_df = pd.read_csv('../data/raw/diabetic_data.csv')
admissions_df = pd.read_csv('../data/raw/admissions.csv')

# Relevant features for readmission and length of stay prediction from diabetic data:
diabetic_features = [
    'encounter_id',           # Unique identifier
    'patient_nbr',            # Patient identifier
    'race',                   # Demographic
    'gender',                 # Demographic
    'age',                    # Demographic - important predictor
    'admission_type_id',      # Type of admission
    'discharge_disposition_id', # Discharge disposition
    'admission_source_id',    # Source of admission
    'time_in_hospital',       # TARGET for length of stay
    'num_lab_procedures',     # Clinical indicator
    'num_procedures',         # Clinical indicator
    'num_medications',        # Clinical indicator
    'number_outpatient',      # Healthcare utilization
    'number_emergency',       # Healthcare utilization
    'number_inpatient',       # Healthcare utilization
    'number_diagnoses',       # Clinical complexity
    'max_glu_serum',          # Lab result
    'A1Cresult',              # Lab result
    'insulin',                # Medication
    'change',                 # Medication change
    'diabetesMed',            # Diabetes medication
    'readmitted'              # TARGET for readmission
]

# Select only relevant columns from diabetic data
diabetic_clean = diabetic_df[diabetic_features].copy()

# Calculate length of stay from admissions data
admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])
admissions_df['dischtime'] = pd.to_datetime(admissions_df['dischtime'])
admissions_df['length_of_stay_days'] = (admissions_df['dischtime'] - admissions_df['admittime']).dt.total_seconds() / 86400

# Relevant features from admissions:
admissions_features = [
    'subject_id',              # Patient identifier (can be merged with patient_nbr)
    'hadm_id',                 # Hospital admission ID
    'admission_type',          # Type of admission (text version)
    'admission_location',      # Where patient came from
    'discharge_location',      # Where patient went - important for readmission
    'insurance',               # Insurance type
    'language',                # Demographic
    'marital_status',          # Demographic
    'race',                    # Demographic
    'hospital_expire_flag',    # Death indicator - important
    'length_of_stay_days'      # Calculated length of stay
]

admissions_clean = admissions_df[admissions_features].copy()

# Save cleaned diabetic data
diabetic_clean.to_csv('../data/processed/diabetic_data_cleaned.csv', index=False)

# Save cleaned admissions data
admissions_clean.to_csv('../data/processed/admissions_cleaned.csv', index=False)