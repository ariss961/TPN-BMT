# -*- coding: utf-8 -*-
"""
RL Pipeline for TPN Optimization using Implicit Q-Learning (IQL)
Offline RL approach without simulation model
This version loads and processes raw data files directly
"""

# =========================================================================
# === GLOBAL PARAMETERS AND CONFIGURATION ===
# =========================================================================

import os
import math
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ConstantInputWarning, entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import gc
import traceback
import datetime
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
import random
import json
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask and attn_mask is deprecated.*", module="torch.nn.functional")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- GLOBAL EXPERIMENT CONFIGURATION ---
EXPERIMENTS_ROOT_DIR = "/remote/home/ariss01/TPN/Final_file/Transformer/Final_RL_implicit_q_learning_v2"
os.makedirs(EXPERIMENTS_ROOT_DIR, exist_ok=True)
EXPERIMENT_NAME = "TPN_IQL_Offline_DirectData"

# Create output directories
IQL_OUTPUT_DIR = os.path.join(EXPERIMENTS_ROOT_DIR, EXPERIMENT_NAME)
os.makedirs(IQL_OUTPUT_DIR, exist_ok=True)
print(f"IQL outputs will be saved to: {IQL_OUTPUT_DIR}")

# --- REWARD FUNCTION PARAMETERS ---
# Simplified reward focusing only on range and mean proximity
W_IN_RANGE_BASE = 0.2      # Per-lab: Base reward for being in range
W_IN_RANGE_PROXIMITY = 0.05  # Per-lab: Additional reward for being close to mean
W_OUT_OF_RANGE_PENALTY = 0.02 # Per-lab: Penalty multiplier for out-of-range values

# New parameters for count-based rewards/penalties
W_COUNT_IN_RANGE_BONUS = 0.02      # Global: Bonus multiplier for each lab in range count
W_COUNT_OUT_OF_RANGE_PENALTY = 0.02 # Global: Penalty multiplier for each lab out of range count

# --- ENVIRONMENT PARAMETERS ---
PAD_VALUE = 2.0
MAX_EPISODE_STEPS_FOR_EVALUATION = None  # None for full episodes

# --- IQL TRAINING PARAMETERS ---
# IQL hyperparameters
IQL_BATCH_SIZE = 256
IQL_BUFFER_SIZE = 2000000
IQL_GAMMA = 0.99
IQL_TAU = 0.002
IQL_EXPECTILE = 0.7  # Key IQL parameter for advantage weighting
IQL_TEMPERATURE = 3.0  # For advantage weighting in policy
IQL_LR_ACTOR = 5e-6  # Learning rate for all networks
IQL_RL_CRITIC = 1e-5
# Training parameters
N_EPOCHS = 150
N_STEPS_PER_EPOCH = 1000

SAVE_INTERVAL = 50

# --- PHYSIOLOGICAL RANGES ---
MEASUREMENT_RANGES = {
    "Potassium [Moles/volume] in Serum or Plasma": (3.5, 5.2),
    "Sodium [Moles/volume] in Serum or Plasma": (135.0, 145.0),
    "Chloride [Moles/volume] in Serum or Plasma": (98.0, 108.0),
    "Bicarbonate [Moles/volume] in Specimen": (22.0, 29.0),
    "Calcium [Mass/volume] in Serum or Plasma": (8.6, 10.3),
    "Phosphate [Mass/volume] in Serum or Plasma": (2.5, 4.5),
    "Magnesium [Mass/volume] in Serum or Plasma": (1.7, 2.2),
    "Glucometer blood glucose": (70.0, 140.0),
    "Albumin [Mass/volume] in Serum or Plasma": (3.5, 5.5),
    "Protein [Mass/volume] in Serum or Plasma": (6.0, 8.3),
    "Urea nitrogen [Mass/volume] in Serum or Plasma": (7.0, 20.0)
}

TARGET_MEAN_VALUES = {
    'Potassium [Moles/volume] in Serum or Plasma': 4.0,
    'Sodium [Moles/volume] in Serum or Plasma': 140.0,
    'Chloride [Moles/volume] in Serum or Plasma': 100.0,
    'Bicarbonate [Moles/volume] in Specimen': 24.0,
    'Calcium [Mass/volume] in Serum or Plasma': 9.5,
    'Phosphate [Mass/volume] in Serum or Plasma': 3.5,
    'Magnesium [Mass/volume] in Serum or Plasma': 2.0,
    'Glucometer blood glucose': 100.0,
    'Albumin [Mass/volume] in Serum or Plasma': 4.0,
    'Protein [Mass/volume] in Serum or Plasma': 7.0,
    'Urea nitrogen [Mass/volume] in Serum or Plasma': 15.0
}


# =========================================================================
# === HELPER FUNCTIONS ===
# =========================================================================

def concordance_correlation_coefficient(y_true, y_pred):
    """Calculate concordance correlation coefficient."""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid_mask.sum() < 2:
        return np.nan
    y_true, y_pred = y_true[valid_mask], y_pred[valid_mask]
    if len(y_true) < 2:
        return np.nan
    try:
        cor, _ = pearsonr(y_true, y_pred)
    except ValueError:
        return np.nan
    if np.isnan(cor):
        return np.nan
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    sd_true, sd_pred = np.std(y_true), np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    if denominator < 1e-9:
        return 1.0 if abs(mean_true - mean_pred) < 1e-9 and abs(sd_true - sd_pred) < 1e-9 else 0.0
    return np.clip(numerator / denominator, -1.0, 1.0)


# =========================================================================
# === HYBRID REWARD FUNCTION ===
# =========================================================================

def calculate_hybrid_reward_v2(
    predicted_measurement_vector: np.ndarray,
    measurement_cols: List[str],
    measurement_ranges: Dict[str, Tuple[float, float]],
    target_mean_values_dict: Dict[str, float],
    w_in_range_base: float,
    w_in_range_proximity: float,
    w_out_of_range_penalty: float,
    w_count_in_range_bonus: float,
    w_count_out_of_range_penalty: float,
    epsilon: float = 1e-6
) -> float:
    """
    Hybrid reward function incorporating:
    1. Per-lab in-range reward (base + proximity to mean).
    2. Per-lab out-of-range penalty (proportional to deviation).
    3. Global bonus proportional to the number of labs in range.
    4. Global penalty proportional to the number of labs out of range.
    """
    if predicted_measurement_vector.ndim != 1 or \
       predicted_measurement_vector.size != len(measurement_cols):
        raise ValueError("predicted_measurement_vector shape mismatch")

    num_measurements = len(measurement_cols)
    individual_lab_rewards_sum = 0.0
    in_range_count = 0
    out_of_range_count = 0

    for val, col in zip(predicted_measurement_vector, measurement_cols):
        if np.isnan(val):
            # Treat NaNs as slightly out of range with small penalty
            individual_lab_rewards_sum += -w_out_of_range_penalty
            out_of_range_count += 1
            continue

        lo, hi = measurement_ranges[col]
        target = target_mean_values_dict[col]
        width = hi - lo + epsilon
        
        if width <= epsilon:
            width = 1.0

        if lo <= val <= hi:
            # Component 1: Per-lab In-Range Reward (with Proximity Bonus)
            base_reward = w_in_range_base
            
            # Simplified proximity bonus: linear decrease from target
            distance_from_target = abs(val - target)
            max_distance = max(abs(target - lo), abs(hi - target))
            if max_distance > 0:
                proximity_factor = 1.0 - (distance_from_target / max_distance)
            else:
                proximity_factor = 1.0
            
            proximity_bonus = w_in_range_proximity * proximity_factor
            r_i = base_reward + proximity_bonus
            individual_lab_rewards_sum += r_i
            in_range_count += 1
        else:
            # Component 2: Per-lab Out-of-Range Penalty (Proportional to Deviation)
            if val < lo:
                distance = lo - val
            else:  # val > hi
                distance = val - hi

            # Normalize distance by width
            normalized_distance = distance / width if width > 0 else distance
            r_i = -w_out_of_range_penalty * normalized_distance
            individual_lab_rewards_sum += r_i
            out_of_range_count += 1

    # Component 3: Bonus for Number of Measurements In Range
    global_in_range_bonus = w_count_in_range_bonus * in_range_count

    # Component 4: Penalty for Number of Measurements Out of Range
    global_out_of_range_penalty = w_count_out_of_range_penalty * out_of_range_count
    
    # Total Reward
    final_reward = individual_lab_rewards_sum + global_in_range_bonus - global_out_of_range_penalty

    return float(final_reward)


# =========================================================================
# === SCALER WRAPPER CLASS ===
# =========================================================================

class MinMaxScalerWrapper:
    def __init__(self, feature_range=(-1, 1), constant_scale_value=0.0):
        self.feature_range = feature_range
        self.constant_scale_value = constant_scale_value
        self._scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
        self._num_features = 0
        self._constant_features_mask = None
        self._constant_values = None
        self.feature_names = None
        
    def fit(self, data, feature_names=None):
        if data is None or data.size == 0:
            self._is_fitted = False
            return self
            
        original_shape = data.shape
        if data.ndim == 3:
            self._num_features = data.shape[2]
            data_2d = data.reshape(-1, self._num_features)
        elif data.ndim == 2:
            self._num_features = data.shape[1]
            data_2d = data
        else:
            raise ValueError(f"Unsupported data dimension for scaler: {data.ndim}.")
            
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self._num_features)]
        elif len(feature_names) == self._num_features:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(self._num_features)]
            print("Warning: feature_names length mismatch.")
            
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                std_devs = np.nanstd(data_2d, axis=0)
                self._constant_features_mask = std_devs < 1e-9
                all_nan_columns = np.all(np.isnan(data_2d), axis=0)
                self._constant_features_mask = np.logical_or(self._constant_features_mask, all_nan_columns)
        except Exception:
            self._constant_features_mask = np.zeros(self._num_features, dtype=bool)
            
        self._constant_features_mask = np.asarray(self._constant_features_mask, dtype=bool)
        
        if np.any(self._constant_features_mask):
            self._constant_values = np.nan_to_num(data_2d[0, self._constant_features_mask], nan=0.0, copy=True)
        else:
            self._constant_values = None
            
        non_constant_mask = ~self._constant_features_mask
        non_constant_data = data_2d[:, non_constant_mask]
        
        if non_constant_data.shape[1] > 0 and np.any(np.isfinite(non_constant_data)):
            try:
                self._scaler.fit(non_constant_data)
                self._is_fitted = True
            except ValueError as e_fit:
                print(f"Warning: Scaler fit failed: {e_fit}")
                self._is_fitted = False
        else:
            self._is_fitted = True
            
        return self
        
    def transform(self, data):
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted.")
        if data is None or data.size == 0:
            return data
            
        original_shape = data.shape
        original_ndim = data.ndim
        
        if original_ndim == 3:
            if data.shape[2] != self._num_features:
                raise ValueError(f"Feature mismatch (3D): Expected {self._num_features}, got {data.shape[2]}.")
            data_2d = data.reshape(-1, self._num_features)
        elif original_ndim == 2:
            if data.shape[1] != self._num_features:
                raise ValueError(f"Feature mismatch (2D): Expected {self._num_features}, got {data.shape[1]}.")
            data_2d = data
        else:
            raise ValueError(f"Unsupported input dimension: {original_ndim}.")
            
        if not np.issubdtype(data_2d.dtype, np.floating):
            data_2d = data_2d.astype(float)
            
        scaled_data_2d = np.full_like(data_2d, self.constant_scale_value, dtype=float)
        scaler_has_params = hasattr(self._scaler, 'scale_') and self._scaler.scale_ is not None
        
        if self._constant_features_mask is not None:
            non_constant_mask = ~self._constant_features_mask
            if np.any(non_constant_mask):
                if scaler_has_params:
                    try:
                        scaled_data_2d[:, non_constant_mask] = self._scaler.transform(data_2d[:, non_constant_mask])
                    except ValueError as e_trans:
                        print(f"Warning: Scaler transform failed for non-constant: {e_trans}")
        elif scaler_has_params:
            try:
                scaled_data_2d = self._scaler.transform(data_2d)
            except ValueError as e_trans:
                print(f"Warning: Scaler transform failed for all: {e_trans}")
            
        return scaled_data_2d.reshape(original_shape) if original_ndim == 3 else scaled_data_2d
        
    def inverse_transform(self, data):
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted.")
        if data is None or data.size == 0:
            return data
            
        original_shape = data.shape
        original_ndim = data.ndim
        
        if original_ndim == 3:
            if data.shape[2] != self._num_features:
                raise ValueError(f"Feature mismatch (3D inv): Expected {self._num_features}, got {data.shape[2]}.")
            data_2d = data.reshape(-1, self._num_features)
        elif original_ndim == 2:
            if data.shape[1] != self._num_features:
                raise ValueError(f"Feature mismatch (2D inv): Expected {self._num_features}, got {data.shape[1]}.")
            data_2d = data
        else:
            raise ValueError(f"Unsupported input dimension: {original_ndim}.")
            
        if not np.issubdtype(data_2d.dtype, np.floating):
            data_2d = data_2d.astype(float)
            
        inv_scaled_data_2d = np.zeros_like(data_2d, dtype=float)
        scaler_has_params = hasattr(self._scaler, 'scale_') and self._scaler.scale_ is not None
        const_approx = 0.0
        
        if self._constant_features_mask is not None and np.any(self._constant_features_mask):
            non_constant_mask = ~self._constant_features_mask
            if np.any(non_constant_mask):
                if scaler_has_params:
                    try:
                        inv_scaled_data_2d[:, non_constant_mask] = self._scaler.inverse_transform(data_2d[:, non_constant_mask])
                    except ValueError as e_inv:
                        inv_scaled_data_2d[:, non_constant_mask] = const_approx
                        print(f"Warning: Scaler inv_transform failed (non-const): {e_inv}")
                else:
                    inv_scaled_data_2d[:, non_constant_mask] = const_approx
            inv_scaled_data_2d[:, self._constant_features_mask] = const_approx
        elif scaler_has_params:
            try:
                inv_scaled_data_2d = self._scaler.inverse_transform(data_2d)
            except ValueError as e_inv:
                inv_scaled_data_2d.fill(const_approx)
                print(f"Warning: Scaler inv_transform failed (all cols): {e_inv}")
        else:
            inv_scaled_data_2d.fill(const_approx)
            
        if self._constant_features_mask is not None and self._constant_values is not None:
            if np.any(self._constant_features_mask):
                num_const_mask = np.sum(self._constant_features_mask)
                if self._constant_values.shape[0] == num_const_mask:
                    inv_scaled_data_2d[:, self._constant_features_mask] = self._constant_values
                    
        return inv_scaled_data_2d.reshape(original_shape) if original_ndim == 3 else inv_scaled_data_2d
        
    @property
    def is_fitted_(self):
        return self._is_fitted


# =========================================================================
# === PART 3: DATA PREPARATION ===
# =========================================================================

print("\n\n" + "="*70)
print("=== PART 3: Data Preparation for Offline RL ===")
print("="*70 + "\n")

# Define data paths directly
TPN_EMB_DATA_PATH = '/remote/home/ariss01/TPN/cohort_tpn_data_original_cleaned__decomposed_pivoted_with_Volume_lipids_corrected_measurement_v3_motor_embeddings_FIXED_Final_sorted.csv'
MEASUREMENTS_DATA_PATH = '/remote/home/ariss01/TPN/Isolated_Measurements_Filtered_selected_features_pivoted.csv'
MEAS_TIMESTAMP_COL = 'measurement_datetime'
TPN_WINDOW_HOURS = 4  # ±4 hours around each TPN dose to collect measurements

# Define feature columns
TPN_DOSE_COLS = [
    'Parenteral Amino Acids (g)', 'Acetate (mEq)', 'Calcium (mEq)', 'Chloride (mEq)', 'Chromium (mcg)',
    'Copper (mg)', 'Dextrose (g)', 'Famotidine (mg)', 'Folic Acid (mg)', 'Heparin (units)',
    'Insulin (units)', 'Levocarnitine (mg)', 'Magnesium (mEq)', 'Manganese (mg)', 'Multivitamins (units)',
    'Phosphate (mmol)', 'Potassium (mEq)', 'Selenium (mcg)', 'Sodium (mEq)', 'Thiamine (mg)',
    'Pyridoxine (mg)', 'Ascorbic Acid (Vitamin C) (mg)', 'Phytonadione (mg)', 'Zinc (mg)', 'lipids(g)'
]
EMBEDDING_COLS = [f'embedding_{i}' for i in range(769)]
MEASUREMENT_FEATURE_COLS = [
    "Potassium [Moles/volume] in Serum or Plasma", "Sodium [Moles/volume] in Serum or Plasma",
    "Chloride [Moles/volume] in Serum or Plasma", "Bicarbonate [Moles/volume] in Specimen",
    "Calcium [Mass/volume] in Serum or Plasma", "Phosphate [Mass/volume] in Serum or Plasma",
    "Magnesium [Mass/volume] in Serum or Plasma", "Glucometer blood glucose",
    "Protein [Mass/volume] in Serum or Plasma", "Urea nitrogen [Mass/volume] in Serum or Plasma",
]
TARGET_COLS = MEASUREMENT_FEATURE_COLS.copy()
RL_AGENT_OBSERVATION_COLS = EMBEDDING_COLS.copy()
ENCODER_INPUT_COLS = TPN_DOSE_COLS + EMBEDDING_COLS

# Load and process raw data files
print("\n--- Loading and Processing Raw Data Files ---")

# Load TPN/Embedding Data
print(f"Loading TPN/Embedding data: {TPN_EMB_DATA_PATH}")
df_main = pd.read_csv(TPN_EMB_DATA_PATH, low_memory=False)
df_main['order_inst'] = pd.to_datetime(df_main['order_inst'], errors='coerce')
df_main['person_id'] = pd.to_numeric(df_main['person_id'], errors='coerce').astype('Int64')
df_main.dropna(subset=['order_inst', 'person_id'], inplace=True)
print(f"  Loaded df_main shape: {df_main.shape}")

# Load Measurement Data
print(f"Loading measurement data: {MEASUREMENTS_DATA_PATH}")
df_meas = pd.read_csv(MEASUREMENTS_DATA_PATH, low_memory=False)
df_meas[MEAS_TIMESTAMP_COL] = pd.to_datetime(df_meas[MEAS_TIMESTAMP_COL], errors='coerce')
df_meas['person_id'] = pd.to_numeric(df_meas['person_id'], errors='coerce').astype('Int64')
df_meas.dropna(subset=[MEAS_TIMESTAMP_COL, 'person_id'], inplace=True)
df_meas.drop_duplicates(subset=['person_id', MEAS_TIMESTAMP_COL], keep='first', inplace=True)
print(f"  Loaded df_meas shape: {df_meas.shape}")


print(f"Filtering measurements to ±{TPN_WINDOW_HOURS}h windows around TPN doses...")

def filter_measurements_by_tpn_windows(patient_id):
    """Vectorized calculation to keep only measurements within ±4h of any TPN dose for this patient"""
    patient_tpn = df_main[df_main['person_id'] == patient_id]['order_inst'].dropna()
    patient_meas = df_meas[df_meas['person_id'] == patient_id].copy()
    
    if patient_tpn.empty or patient_meas.empty:
        return pd.DataFrame()
    
    # Vectorized calculation for time windows
    tpn_times = pd.to_datetime(patient_tpn.values).values
    meas_times = pd.to_datetime(patient_meas[MEAS_TIMESTAMP_COL].values).values

    diffs = np.abs(meas_times[:, None] - tpn_times[None, :])
    min_diffs = np.min(diffs, axis=1)
    
    # Convert hours to nanoseconds for numpy timedelta64 comparison
    window_ns = TPN_WINDOW_HOURS * 3600 * 1e9
    within_window_mask = min_diffs <= np.timedelta64(int(window_ns), 'ns')
    
    return patient_meas[within_window_mask]

# Apply filtering
filtered_meas_list = []
common_patients = set(df_main['person_id'].unique()) & set(df_meas['person_id'].unique())
print(f"  Found {len(common_patients)} patients with both TPN and measurement data")

for patient_id in tqdm(common_patients, desc="Filtering measurements"):
    filtered_patient_meas = filter_measurements_by_tpn_windows(patient_id)
    if not filtered_patient_meas.empty:
        filtered_meas_list.append(filtered_patient_meas)

if filtered_meas_list:
    df_meas = pd.concat(filtered_meas_list, ignore_index=True)
    print(f"  Measurements after filtering: {len(df_meas)}")
else:
    raise ValueError("No measurements found within ±4h windows of TPN doses")

# Filter TPN data to only include patients with measurements
valid_patient_ids = df_meas['person_id'].unique()
df_main = df_main[df_main['person_id'].isin(valid_patient_ids)].copy()

# Process TPN/Embedding columns
for col in TPN_DOSE_COLS:
    if col in df_main.columns:
        df_main[col] = pd.to_numeric(df_main[col], errors='coerce').fillna(0.0)
        df_main[col] = df_main[col] / 24.0  # Convert to hourly rates

for col in EMBEDDING_COLS:
    if col in df_main.columns:
        df_main[col] = pd.to_numeric(df_main[col], errors='coerce').fillna(0.0)

# Process measurement columns
for col in MEASUREMENT_FEATURE_COLS:
    if col in df_meas.columns:
        df_meas[col] = pd.to_numeric(df_meas[col], errors='coerce')

# Create aligned dataframe
main_cols = ['person_id', 'order_inst'] + TPN_DOSE_COLS + EMBEDDING_COLS
df_main_processed = df_main[main_cols].copy()
df_main_processed.rename(columns={'order_inst': 'event_time'}, inplace=True)
df_main_processed['is_tpn_event'] = 1

meas_cols = ['person_id', MEAS_TIMESTAMP_COL] + MEASUREMENT_FEATURE_COLS
df_meas_processed = df_meas[meas_cols].copy()
df_meas_processed.rename(columns={MEAS_TIMESTAMP_COL: 'event_time'}, inplace=True)
df_meas_processed['is_measurement_event'] = 1

# Merge and align data
print("Merging TPN and measurement data...")
df_aligned = pd.concat([df_main_processed, df_meas_processed], ignore_index=True, sort=False)
df_aligned.sort_values(['person_id', 'event_time', 'is_tpn_event'], inplace=True, ascending=[True, True, False])

# Forward fill TPN context
print("Forward-filling TPN/Embedding context...")
context_cols = TPN_DOSE_COLS + EMBEDDING_COLS
df_aligned[context_cols] = df_aligned.groupby('person_id')[context_cols].ffill()
df_aligned[context_cols] = df_aligned[context_cols].fillna(0.0)

# ── Impute measurements ──────────────────────────────────────────────
print("Imputing measurements...")
df_aligned['tpn_interval_marker_time'] = df_aligned.loc[df_aligned['is_tpn_event'] == 1, 'event_time']
df_aligned['tpn_interval_marker_time'] = (
    df_aligned.groupby('person_id')['tpn_interval_marker_time'].ffill()
)
df_aligned['tpn_interval_marker_time'].fillna(pd.Timestamp.min, inplace=True)

# 1️⃣ forward-fill within each person × TPN-interval
df_aligned[MEASUREMENT_FEATURE_COLS] = (
    df_aligned.groupby(['person_id', 'tpn_interval_marker_time'])[MEASUREMENT_FEATURE_COLS]
              .transform(lambda x: x.ffill())
)

# 2️⃣ patient-mean fallback (leak-free)
df_aligned[MEASUREMENT_FEATURE_COLS] = (
    df_aligned.groupby('person_id')[MEASUREMENT_FEATURE_COLS]
              .transform(lambda x: x.fillna(x.mean()))
)

# 3️⃣ last-resort padding only for labs never measured in that patient
df_aligned[MEASUREMENT_FEATURE_COLS] = (
    df_aligned[MEASUREMENT_FEATURE_COLS].fillna(PAD_VALUE)
)

# ── tidy up ──────────────────────────────────────────────────────────
df_aligned.drop(columns=['tpn_interval_marker_time'], inplace=True, errors='ignore')

# <<< MODIFICATION START: Added Train/Validation/Test Split >>>
print("\n--- Splitting patients into Train, Validation, and Test sets ---")
all_patient_ids = df_aligned['person_id'].unique()
if len(all_patient_ids) < 3:
    raise ValueError("Insufficient patients for train/validation/test split. Need at least 3.")

# Split into training+validation (85%) and a separate test set (15%)
train_val_ids, TEST_IDS_P2 = train_test_split(
    all_patient_ids, test_size=0.15, random_state=42
)

# Split the training+validation set into a final training set and a validation set.
# We'll use 15% of the original data for validation, which is 0.15 / 0.85 of the train_val_ids.
validation_size_ratio = 0.15 / 0.85
TRAIN_IDS_P2, VALIDATION_IDS_P2 = train_test_split(
    train_val_ids, test_size=validation_size_ratio, random_state=42
)
print(f"Total patients: {len(all_patient_ids)}")
print(f"  - Training patients:   {len(TRAIN_IDS_P2)}")
print(f"  - Validation patients: {len(VALIDATION_IDS_P2)}")
print(f"  - Test patients:       {len(TEST_IDS_P2)}")
# <<< MODIFICATION END >>>

# Create measurement data dictionary
measurement_data_dict_env = {}
for person_id in tqdm(all_patient_ids, desc="Creating measurement dictionary"):
    patient_data = df_aligned[(df_aligned['person_id'] == person_id) & (df_aligned['is_measurement_event'] == 1)].copy()
    if len(patient_data) > 0:
        patient_data = patient_data.sort_values('event_time')
        measurement_data_dict_env[int(person_id)] = {
            'timestamps': patient_data['event_time'].values.astype('datetime64[ns]'),
            'values_processed': patient_data[MEASUREMENT_FEATURE_COLS].values.astype(np.float32)
        }

# Create df_main_env
context_cols = TPN_DOSE_COLS + EMBEDDING_COLS
df_main_env = df_aligned[df_aligned['is_tpn_event'] == 1].copy()
df_main_env = df_main_env[['person_id', 'event_time'] + context_cols]
df_main_env = df_main_env.set_index(['person_id', 'event_time'])
print(f"df_main_env shape: {df_main_env.shape}")

# =========================================================================
# === OFFLINE DATASET CREATION ===
# =========================================================================

print("\n\n" + "="*70)
print("=== Creating Offline Dataset for IQL ===")
print("="*70 + "\n")

def create_offline_dataset_for_iql(
        df_main_env          : pd.DataFrame,
        measurement_data_dict: Dict[int, Dict[str, np.ndarray]],
        patient_ids          : List[int],
        rl_observation_cols  : List[str],    # embedding cols
        measurement_cols     : List[str],    # lab feature names
        target_cols          : List[str],    # (same set, keeps API)
        dose_cols            : List[str],    # action features
        pad_value            : float = PAD_VALUE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:

    obs_list, act_list, rew_list, next_obs_list, done_list = [], [], [], [], []

    emb_dim       = len(rl_observation_cols)
    meas_dim      = len(measurement_cols)
    action_dim    = len(dose_cols)
    tgt2meas_idx  = [measurement_cols.index(c) for c in target_cols]

    # <<< MODIFICATION START: Added counters for pair validation >>>
    total_tpn_events = 0
    found_pairs = 0
    # <<< MODIFICATION END >>>

    for pid in tqdm(patient_ids, desc="Building (s,a,s',r) tuples"):
        # --- pull this patient's data ---------------------------------
        if pid not in df_main_env.index.get_level_values('person_id'):
            continue
        tpn_df  = df_main_env.loc[pid].sort_index()        # (T rows)
        labs    = measurement_data_dict.get(pid)
        if labs is None or labs['values_processed'].shape[0] < 2:
            continue
        labs_df = pd.DataFrame(labs['values_processed'],
                               index=pd.to_datetime(labs['timestamps']),
                               columns=measurement_cols).sort_index()

        # --- iterate over every TPN event ----------------------------
        for t_idx, (tpn_time, tpn_row) in enumerate(tpn_df.iterrows()):
            
            # <<< MODIFICATION START: Increment total event counter >>>
            total_tpn_events += 1
            # <<< MODIFICATION END >>>

            # ➊ find the lab just before and strictly after this TPN
            prev_lab = labs_df.loc[labs_df.index <= tpn_time].iloc[-1:]      # t−
            next_lab = labs_df.loc[labs_df.index > tpn_time].iloc[0:1]       # Strictly after to prevent identical states

            if prev_lab.empty or next_lab.empty:
                continue  # need both to form a sample
            
            # <<< MODIFICATION START: Increment found pair counter >>>
            found_pairs += 1
            # <<< MODIFICATION END >>>

            # ➋ STATE  = embedding @t  + previous labs
            state = np.concatenate([
                tpn_row[rl_observation_cols].values.astype(np.float32),
                prev_lab.iloc[0].values.astype(np.float32)
            ])

            # ➌ ACTION = TPN dose row (already hourly-rate)
            action = tpn_row[dose_cols].values.astype(np.float32)

            # ➍ NEXT-STATE = same embedding + post-TPN labs
            next_state = np.concatenate([
                tpn_row[rl_observation_cols].values.astype(np.float32),
                next_lab.iloc[0].values.astype(np.float32)
            ])

            # ➎ REWARD on post-TPN labs
            next_labs_vec = next_lab.iloc[0, tgt2meas_idx].values.astype(np.float32)
            reward = calculate_hybrid_reward_v2(
                predicted_measurement_vector = next_labs_vec,
                measurement_cols             = target_cols,
                measurement_ranges           = MEASUREMENT_RANGES,
                target_mean_values_dict      = TARGET_MEAN_VALUES,
                w_in_range_base              = W_IN_RANGE_BASE,
                w_in_range_proximity         = W_IN_RANGE_PROXIMITY,
                w_out_of_range_penalty       = W_OUT_OF_RANGE_PENALTY,
                w_count_in_range_bonus       = W_COUNT_IN_RANGE_BONUS,
                w_count_out_of_range_penalty = W_COUNT_OUT_OF_RANGE_PENALTY
            )
            # Clip reward to prevent exploding gradients
            reward = np.clip(reward, -10.0, 10.0)

            # ➏ DONE?  only for the very last TPN of the patient
            done = (t_idx == len(tpn_df) - 1)

            # ➐ append
            obs_list.append(state)
            act_list.append(action)
            rew_list.append(reward)
            next_obs_list.append(next_state)
            done_list.append(done)

    # --- to numpy -----------------------------------------------------
    observations      = np.array(obs_list,      dtype=np.float32)
    actions           = np.array(act_list,      dtype=np.float32)
    rewards           = np.array(rew_list,      dtype=np.float32)
    next_observations = np.array(next_obs_list, dtype=np.float32)
    dones             = np.array(done_list,     dtype=np.float32)

    # <<< MODIFICATION START: Print confirmation log >>>
    print("\n--- Measurement Pair Confirmation ---")
    if total_tpn_events > 0:
        pair_percentage = (found_pairs / total_tpn_events) * 100
        print(f"Processed {total_tpn_events} TPN events.")
        print(f"Found {found_pairs} valid (pre, post) lab pairs ({pair_percentage:.2f}% coverage).")
    else:
        print("No TPN events were processed for this patient set.")
    # <<< MODIFICATION END >>>

    # quick log
    print(f"\nDataset built: {observations.shape[0]} transitions  | "
          f"obs_dim={observations.shape[1]}  act_dim={actions.shape[1]}")
    return observations, actions, rewards, next_observations, dones

# Create offline datasets for training, validation, and testing
print("\nCreating training dataset...")
train_obs, train_acts, train_rews, train_next_obs, train_dones = create_offline_dataset_for_iql(
    df_main_env=df_main_env,
    measurement_data_dict=measurement_data_dict_env,
    patient_ids=list(TRAIN_IDS_P2),
    rl_observation_cols=RL_AGENT_OBSERVATION_COLS,
    measurement_cols=MEASUREMENT_FEATURE_COLS,
    target_cols=TARGET_COLS,
    dose_cols=TPN_DOSE_COLS,
    pad_value=PAD_VALUE
)

# <<< MODIFICATION START: Create Validation Dataset >>>
print("\nCreating validation dataset...")
val_obs, val_acts, val_rews, val_next_obs, val_dones = create_offline_dataset_for_iql(
    df_main_env=df_main_env,
    measurement_data_dict=measurement_data_dict_env,
    patient_ids=list(VALIDATION_IDS_P2),
    rl_observation_cols=RL_AGENT_OBSERVATION_COLS,
    measurement_cols=MEASUREMENT_FEATURE_COLS,
    target_cols=TARGET_COLS,
    dose_cols=TPN_DOSE_COLS,
    pad_value=PAD_VALUE
)
# <<< MODIFICATION END >>>

print("\nCreating test dataset...")
test_obs, test_acts, test_rews, test_next_obs, test_dones = create_offline_dataset_for_iql(
    df_main_env=df_main_env,
    measurement_data_dict=measurement_data_dict_env,
    patient_ids=list(TEST_IDS_P2),
    rl_observation_cols=RL_AGENT_OBSERVATION_COLS,
    measurement_cols=MEASUREMENT_FEATURE_COLS,
    target_cols=TARGET_COLS,
    dose_cols=TPN_DOSE_COLS,
    pad_value=PAD_VALUE
)

# =========================================================================
# === DATA ARTIFACTS ===
# =========================================================================

IQL_RUN_NAME = f"iql_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
IQL_LOG_DIR = os.path.join(IQL_OUTPUT_DIR, IQL_RUN_NAME)
os.makedirs(IQL_LOG_DIR, exist_ok=True)
print(f"Artifacts will be saved to: {IQL_LOG_DIR}")

DATA_ARTIFACTS_DIR = os.path.join(IQL_LOG_DIR, "data_artifacts")
os.makedirs(DATA_ARTIFACTS_DIR, exist_ok=True)
print(f"Saving data artifacts to: {DATA_ARTIFACTS_DIR}")

# Save datasets as .npy files
np.save(os.path.join(DATA_ARTIFACTS_DIR, 'train_obs.npy'), train_obs)
np.save(os.path.join(DATA_ARTIFACTS_DIR, 'train_acts.npy'), train_acts)
np.save(os.path.join(DATA_ARTIFACTS_DIR, 'test_obs.npy'), test_obs)
np.save(os.path.join(DATA_ARTIFACTS_DIR, 'test_acts.npy'), test_acts)
np.save(os.path.join(DATA_ARTIFACTS_DIR, 'test_rews.npy'), test_rews)
np.save(os.path.join(DATA_ARTIFACTS_DIR, 'test_next_obs.npy'), test_next_obs)
np.save(os.path.join(DATA_ARTIFACTS_DIR, 'test_dones.npy'), test_dones)

# Save the test patient IDs
with open(os.path.join(DATA_ARTIFACTS_DIR, 'test_patient_ids.json'), 'w') as f:
    json.dump([int(i) for i in TEST_IDS_P2], f)

# Save the aligned dataframe required for trajectory plots
df_aligned.to_pickle(os.path.join(DATA_ARTIFACTS_DIR, 'df_aligned.pkl'))

print("All necessary data artifacts for standalone evaluation have been saved.")


# =========================================================================
# === PART 4: IQL TRAINING ===
# =========================================================================

print("\n\n" + "="*70)
print("=== PART 4: Implicit Q-Learning (IQL) Training ===")
print("="*70 + "\n")

# Import d3rlpy
try:
    import d3rlpy
    from d3rlpy.algos import IQLConfig
    from d3rlpy.dataset import MDPDataset
    from d3rlpy.logging import FileAdapterFactory
    from d3rlpy.models.encoders import VectorEncoderFactory
    from d3rlpy.preprocessing import StandardObservationScaler
    from d3rlpy.metrics import ContinuousActionDiffEvaluator
except ImportError as e:
    print(f"d3rlpy import error: {e}. Please ensure d3rlpy is installed.")
    raise

# Prepare action scaling
print("\n--- Preparing Action Scaling ---")
min_actions = np.min(train_acts, axis=0)
max_actions = np.max(train_acts, axis=0)

action_scaler = MinMaxScaler(feature_range=(-1, 1))
action_scaler.fit(train_acts)

train_acts_scaled = action_scaler.transform(train_acts)
val_acts_scaled = action_scaler.transform(val_acts)
test_acts_scaled = action_scaler.transform(test_acts)

# Prepare observation scaling
print("\n--- Preparing Observation Scaling ---")
obs_mean = np.mean(train_obs, axis=0)
obs_std = np.std(train_obs, axis=0)
obs_std[obs_std < 1e-8] = 1.0

observation_scaler = StandardObservationScaler(mean=obs_mean, std=obs_std)
print("Observation scaler prepared.")

# Create MDPDataset
print("\n--- Creating MDPDataset ---")
train_dataset = MDPDataset(
    observations=train_obs,
    actions=train_acts_scaled,
    rewards=train_rews,
    terminals=train_dones,
    action_space=d3rlpy.constants.ActionSpace.CONTINUOUS
)

val_dataset = MDPDataset(
    observations=val_obs,
    actions=val_acts_scaled,
    rewards=val_rews,
    terminals=val_dones
)

num_episodes      = train_dataset.size()
num_transitions = train_dataset.transition_count
print(f"Training dataset: {num_episodes} episodes / {num_transitions} transitions")

# Configure IQL
print("\n--- Configuring IQL Agent ---")
device_to_use = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device_to_use}")

actor_encoder = VectorEncoderFactory(hidden_units=[256, 256], activation='relu', use_batch_norm=False)
critic_encoder = VectorEncoderFactory(hidden_units=[256, 256], activation='relu', use_batch_norm=False)
value_encoder = VectorEncoderFactory(hidden_units=[256, 256], activation='relu', use_batch_norm=False)

iql_config = IQLConfig(
    batch_size=IQL_BATCH_SIZE,
    actor_learning_rate=IQL_LR_ACTOR,
    critic_learning_rate= IQL_RL_CRITIC,
    actor_encoder_factory=actor_encoder,
    critic_encoder_factory=critic_encoder,
    value_encoder_factory=value_encoder,
    observation_scaler=observation_scaler,
    gamma=IQL_GAMMA,
    tau=IQL_TAU,
    expectile=IQL_EXPECTILE,
    weight_temp=IQL_TEMPERATURE,
)

iql = iql_config.create(device=device_to_use)
print("IQL agent created.")

logger_adapter_factory = FileAdapterFactory(root_dir=IQL_OUTPUT_DIR)

# Training loop
print("\n--- Starting IQL Training ---")
training_successful = False
final_model_path = ""
try:
    start_time = time.time()
    
    iql.fit(
        train_dataset,
        n_steps=N_EPOCHS * N_STEPS_PER_EPOCH,
        n_steps_per_epoch=N_STEPS_PER_EPOCH,
        logger_adapter=logger_adapter_factory,
        experiment_name=IQL_RUN_NAME,
        show_progress=True,
        save_interval=SAVE_INTERVAL,
        evaluators={
            'td_error': d3rlpy.metrics.TDErrorEvaluator(),
            'value_scale': d3rlpy.metrics.AverageValueEstimationEvaluator(),
            'validation_value_scan': d3rlpy.metrics.InitialStateValueEstimationEvaluator(episodes=val_dataset.episodes),
        }
    )

    end_time = time.time()
    print(f"\nIQL training completed in {(end_time - start_time) / 3600:.2f} hours")
    
    print("\n--- Dynamically locating the exact log directory ---")
    search_pattern = os.path.join(IQL_OUTPUT_DIR, IQL_RUN_NAME + '*')
    possible_dirs = glob.glob(search_pattern)

    if not possible_dirs:
        print(f"FATAL: Could not find any log directory matching pattern: {search_pattern}")
        actual_log_dir = IQL_LOG_DIR
    else:
        actual_log_dir = max(possible_dirs, key=os.path.getmtime)
        print(f"Found actual log directory: {actual_log_dir}")

    IQL_LOG_DIR = actual_log_dir

    final_model_path = os.path.join(IQL_LOG_DIR, "iql_final_model.d3")
    iql.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    action_scaler_path = os.path.join(IQL_LOG_DIR, "action_scaler.joblib")
    joblib.dump(action_scaler, action_scaler_path)
    print(f"Action scaler saved to: {action_scaler_path}")

    training_successful = True
    
except Exception as e:
    print(f"\nError during IQL training: {e}")
    traceback.print_exc()

# =========================================================================
# === PART 5: OFFLINE EVALUATION ON SEPARATE TEST SET ===
# =========================================================================

print("\n\n" + "="*70)
print("=== PART 5: Offline Evaluation on Separate Test Set ===")
print("="*70 + "\n")

P5_OUT_DIR = os.path.join(IQL_LOG_DIR, "evaluation_results_TEST_SET")
os.makedirs(P5_OUT_DIR, exist_ok=True)
print(f"Evaluation outputs will be saved to: {P5_OUT_DIR}")

# Load trained IQL model
print("\n--- Loading trained IQL model ---")
try:
    iql_eval = d3rlpy.load_learnable(final_model_path, device=device_to_use)
    print("IQL model loaded successfully.")
except Exception as e:
    print(f"Error loading IQL model: {e}")
    try:
        latest_model_path = os.path.join(IQL_LOG_DIR, "model_latest.d3")
        iql_eval = d3rlpy.load_learnable(latest_model_path, device=device_to_use)
        print(f"Loaded latest model instead: {latest_model_path}")
    except Exception as e_latest:
        print(f"Could not load any model. Error: {e_latest}")
        raise

def evaluate_offline_policy(
    dataset_obs: np.ndarray,
    dataset_acts_scaled: np.ndarray,
    dataset_rews: np.ndarray,
    dataset_next_obs: np.ndarray,
    dataset_dones: np.ndarray,
    policy,
    action_scaler,
    policy_name: str = "Policy"
) -> Dict[str, Any]:
    """Evaluate a policy on offline data."""
    predicted_actions_scaled = policy.predict(dataset_obs)
    predicted_actions_unscaled = action_scaler.inverse_transform(predicted_actions_scaled)
    original_actions_unscaled = action_scaler.inverse_transform(dataset_acts_scaled)
    
    action_mse = np.mean((predicted_actions_unscaled - original_actions_unscaled) ** 2)
    action_mae = np.mean(np.abs(predicted_actions_unscaled - original_actions_unscaled))
    
    action_mse_per_dim = np.mean((predicted_actions_unscaled - original_actions_unscaled) ** 2, axis=0)
    action_mae_per_dim = np.mean(np.abs(predicted_actions_unscaled - original_actions_unscaled), axis=0)
    
    mean_reward = np.mean(dataset_rews)
    std_reward = np.std(dataset_rews)
    
    results = {
        'policy_name': policy_name,
        'num_transitions': len(dataset_obs),
        'action_mse': action_mse,
        'action_mae': action_mae,
        'action_mse_per_dim': action_mse_per_dim,
        'action_mae_per_dim': action_mae_per_dim,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'predicted_actions': predicted_actions_unscaled,
        'original_actions': original_actions_unscaled,
        'rewards': dataset_rews
    }
    
    return results

print("\n--- Evaluating IQL on the held-out TEST set ---")
test_results = evaluate_offline_policy(
    test_obs, test_acts_scaled, test_rews, test_next_obs, test_dones,
    iql_eval, action_scaler, "IQL"
)

print(f"\nTest Set Results:")
print(f"  Number of transitions: {test_results['num_transitions']}")
print(f"  Action MSE: {test_results['action_mse']:.4f}")
print(f"  Action MAE: {test_results['action_mae']:.4f}")
print(f"  Mean Reward: {test_results['mean_reward']:.4f}")
print(f"  Reward Std: {test_results['std_reward']:.4f}")

# Action comparison plots
print("\n--- Generating evaluation plots using TEST SET data ---")

# 1. Action Distribution Comparison
num_actions = len(TPN_DOSE_COLS)
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
axes = axes.flatten()

for i, dose_name in enumerate(TPN_DOSE_COLS):
    ax = axes[i]
    sns.histplot(test_results['original_actions'][:, i], bins=30, alpha=0.6,
                 label='Clinical (Expert)', color='green', ax=ax, stat='density', kde=True)
    sns.histplot(test_results['predicted_actions'][:, i], bins=30, alpha=0.6,
                 label='IQL Agent', color='blue', ax=ax, stat='density', kde=True)
    
    ax.set_xlabel(f'{dose_name} (Hourly Rate)')
    ax.set_ylabel('Density')
    ax.set_title(f'{dose_name}', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Action Distribution Comparison on Test Set: Clinical vs IQL Policy', fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(os.path.join(P5_OUT_DIR, 'action_distributions_comparison.png'), dpi=150)
plt.close()

# 2. Action Scatter Plots (RL Agent vs. Expert)
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
axes = axes.flatten()

for i, dose_name in enumerate(TPN_DOSE_COLS):
    ax = axes[i]
    ax.scatter(test_results['original_actions'][:, i],
               test_results['predicted_actions'][:, i],
               alpha=0.3, s=5, edgecolors='none')
    
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    ax.set_xlabel(f'Clinical Action')
    ax.set_ylabel(f'IQL Action')
    ax.set_title(f'{dose_name}\n(MAE: {test_results["action_mae_per_dim"][i]:.4f})', fontsize=10)
    ax.grid(True, alpha=0.3)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Action Comparison on Test Set: Clinical (Expert) vs. IQL Policy', fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(os.path.join(P5_OUT_DIR, 'action_scatter_comparison.png'), dpi=150)
plt.close()

# 3. Reward Distribution Plot
plt.figure(figsize=(10, 6))
sns.histplot(test_results['rewards'], bins=50, alpha=0.7, color='purple', edgecolor='black', kde=True)
plt.axvline(test_results['mean_reward'], color='red', linestyle='--',
            label=f'Mean: {test_results["mean_reward"]:.3f}')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Reward Distribution in Test Set')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(P5_OUT_DIR, 'reward_distribution.png'), dpi=150)
plt.close()

# 4. Learning Curves
print("\n--- Plotting learning curves ---")
try:
    log_dir = IQL_LOG_DIR
    print(f"Attempting to read log files from: {log_dir}")

    metrics_to_plot = {
        'actor_loss': 'Actor Loss',
        'critic_loss': 'Critic Loss',
        'v_loss': 'Value Function Loss',
        'validation_value_scan': 'Avg. Value Estimation (Validation)',
        'td_error': 'TD Error (Validation)'
    }

    available_metrics = {k: v for k, v in metrics_to_plot.items() if os.path.exists(os.path.join(log_dir, f"{k}.csv"))}

    if not available_metrics:
        print("Warning: No metric files were found in the log directory. Skipping plot generation.")
    else:
        num_plots = len(available_metrics)
        n_cols = 2
        n_rows = (num_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows), squeeze=False)
        axes = axes.flatten()
        plot_idx = 0

        for metric_name, metric_label in available_metrics.items():
            metric_file = os.path.join(log_dir, f'{metric_name}.csv')
            try:
                df = pd.read_csv(metric_file, header=None)
                df.columns = ['epoch', 'step', 'value']

                if df.empty:
                    print(f"Warning: Metric file '{metric_file}' is empty.")
                    continue

                ax = axes[plot_idx]
                ax.plot(df['step'], df['value'].rolling(window=10).mean())
                ax.set_xlabel('Training Steps')
                ax.set_ylabel(metric_label)
                ax.set_title(f'{metric_label} Curve')
                ax.grid(True, alpha=0.5)
                plot_idx += 1
            except Exception as e:
                print(f"Error reading or plotting {metric_name} from {metric_file}: {e}")

        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('IQL Training Metrics', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(P5_OUT_DIR, 'iql_training_curves.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Successfully saved learning curves to {save_path}")

except Exception as e:
    print(f"A general error occurred during learning curve plotting: {e}")
    traceback.print_exc()

# Save evaluation results
print("\n--- Saving evaluation results from TEST SET ---")

results_summary = {
    'test_set_metrics': {
        'num_transitions': test_results['num_transitions'],
        'action_mse': float(test_results['action_mse']),
        'action_mae': float(test_results['action_mae']),
        'mean_reward': float(test_results['mean_reward']),
        'std_reward': float(test_results['std_reward']),
        'action_mse_per_dim': test_results['action_mse_per_dim'].tolist(),
        'action_mae_per_dim': test_results['action_mae_per_dim'].tolist(),
    },
    'dose_columns': TPN_DOSE_COLS,
    'model_path': final_model_path,
    'experiment_name': EXPERIMENT_NAME,
}

with open(os.path.join(P5_OUT_DIR, 'evaluation_summary.json'), 'w') as f:
    json.dump(results_summary, f, indent=2)

# =========================================================================
# === PART 5b: ACTION DISTRIBUTION ANALYSIS ON TEST SET ===
# =========================================================================
print("\n\n" + "-"*70)
print("--- Part 5b: Action Distribution Analysis (KL & Coverage) ---")
print("-"*70 + "\n")

predicted_actions = test_results['predicted_actions']
original_actions = test_results['original_actions']
all_train_actions_unscaled = action_scaler.inverse_transform(train_acts_scaled)

# --- 1. KL-Divergence Calculation ---
print("\n--- KL-Divergence (IQL Policy || Clinical Policy) ---")
kl_divergences = []
for j, dose_name in enumerate(TPN_DOSE_COLS):
    actions_behavior = original_actions[:, j]
    actions_policy = predicted_actions[:, j]

    min_val = min(actions_behavior.min(), actions_policy.min())
    max_val = max(actions_behavior.max(), actions_policy.max())
    bins = np.linspace(min_val, max_val, 50)

    hist_behavior, _ = np.histogram(actions_behavior, bins=bins, density=True)
    hist_policy, _ = np.histogram(actions_policy, bins=bins, density=True)
    hist_behavior += 1e-8
    hist_policy += 1e-8

    kl_div = entropy(hist_policy, hist_behavior)
    kl_divergences.append(kl_div)
    print(f"  {dose_name:<45} KL Divergence: {kl_div:.4f}")

# --- 2. Behavior Support Coverage Calculation ---
print("\n--- IQL Policy Coverage of Behavior Support ---")
min_support = all_train_actions_unscaled.min(axis=0)
max_support = all_train_actions_unscaled.max(axis=0)

in_bounds = (predicted_actions >= min_support[None, :]) & (predicted_actions <= max_support[None, :])
full_coverage = np.mean(in_bounds.all(axis=1))
print(f"\nOverall Coverage: {full_coverage*100:.2f}% of IQL actions are fully within the expert's min-max range.")

print("\nPer-Dimension Coverage:")
coverage_per_dim = in_bounds.mean(axis=0)
for j, dose_name in enumerate(TPN_DOSE_COLS):
    print(f"  {dose_name:<45} Coverage: {coverage_per_dim[j]*100:.2f}%")

# --- 3. Save results to the summary file ---
results_summary['test_set_metrics']['kl_divergences'] = kl_divergences
results_summary['test_set_metrics']['overall_coverage'] = float(full_coverage)
results_summary['test_set_metrics']['coverage_per_dim'] = coverage_per_dim.tolist()

with open(os.path.join(P5_OUT_DIR, 'evaluation_summary.json'), 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\nDistribution analysis complete and results have been saved.")

print("\n--- Generating KL Divergence Plot ---")
try:
    fig, ax = plt.subplots(figsize=(12, 10))
    
    y_pos = np.arange(len(TPN_DOSE_COLS))
    ax.barh(y_pos, kl_divergences, align='center', color='indianred', edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(TPN_DOSE_COLS)
    ax.invert_yaxis()
    ax.set_xlabel('KL Divergence (IQL Policy || Clinical Policy)')
    ax.set_title('KL Divergence of Actions')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(P5_OUT_DIR, 'kl_divergence_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"KL Divergence plot saved to {save_path}")

except Exception as e:
    print(f"Could not generate KL Divergence plot. Error: {e}")
