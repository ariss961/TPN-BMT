import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader # Keep DataLoader
from torch.utils.data.dataloader import default_collate # Can be helpful for custom collate later if needed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
import traceback
import warnings
import math
from tqdm.notebook import tqdm # Use standard tqdm if not in a notebook
import random

# --- 0. Configuration ---
TPN_DATA_PATH = '/remote/home/ariss01/TPN/cohort_tpn_data_original_cleaned__decomposed_pivoted_with_Volume_lipids_corrected_measurement_v3_motor_embeddings_FIXED.csv'
ISOLATED_MEAS_PATH = '/remote/home/ariss01/TPN/Isolated_Measurements_Filtered_selected_features_pivoted.csv'
PLOT_SAVE_DIR = '/remote/home/ariss01/TPN/Final_file/Transformer/RL_ACTIVE'

# --- Column Lists (Keep as before) ---
tpn_dose_cols = [
    'Parenteral Amino Acids (g)', 'Acetate (mEq)', 'Calcium (mEq)', 'Chloride (mEq)',
    'Chromium (mcg)', 'Copper (mg)', 'Dextrose (g)', 'Famotidine (mg)',
    'Folic Acid (mg)', 'Heparin (units)', 'Insulin (units)', 'Levocarnitine (mg)',
    'Magnesium (mEq)', 'Manganese (mg)', 'Multivitamins (units)', 'Phosphate (mmol)',
    'Potassium (mEq)', 'Selenium (mcg)', 'Sodium (mEq)', 'Thiamine (mg)',
    'Pyridoxine (mg)', 'Ascorbic Acid (Vitamin C) (mg)', 'Phytonadione (mg)',
    'Zinc (mg)', 'lipids(g)'
]
embedding_cols = [f'embedding_{i}' for i in range(769)]
measurement_cols_target_placeholder = [
    "25-Hydroxyvitamin D3+25-Hydroxyvitamin D2 [Mass/volume] in Serum or Plasma", "25-hydroxyvitamin D3 [Mass/volume] in Serum or Plasma",
    "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma", "Albumin [Mass/volume] in Serum or Plasma",
    "Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma", "Amylase [Enzymatic activity/volume] in Serum or Plasma",
    "Anion gap in Serum or Plasma", "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
    "Base excess in Blood by calculation", "Bicarbonate [Moles/volume] in Specimen", "Bicarbonate measurement",
    "Bilirubin.direct [Mass/volume] in Serum or Plasma", "Bilirubin.indirect [Mass/volume] in Serum or Plasma", "Bilirubin.total [Mass/volume] in Serum or Plasma",
    "Blood potassium measurement", "Body height", "Body mass index (BMI) [Ratio]", "Body surface area", "Body temperature", "Body weight",
    "Calciferol (Vit D2) [Mass/volume] in Serum or Plasma", "Calcium [Mass/volume] in Serum or Plasma", "Calcium.ionized [Mass/volume] in Serum or Plasma",
    "Calcium.ionized [Moles/volume] in Serum or Plasma", "Carbon dioxide [Partial pressure] adjusted to patient's actual temperature in Blood",
    "Carbon dioxide [Partial pressure] in Venous blood", "Carbon dioxide, total [Moles/volume] in Serum or Plasma", "Central venous pressure (CVP)",
    "Chloride [Moles/volume] in Serum or Plasma", "Chloride measurement, blood", "Cholesterol [Mass/volume] in Serum or Plasma",
    "Cholesterol in HDL [Mass/volume] in Serum or Plasma", "Cholesterol.total/Cholesterol in HDL [Mass Ratio] in Serum or Plasma",
    "Cobalamin (Vitamin B12) [Mass/volume] in Serum or Plasma", "Creatine kinase [Enzymatic activity/volume] in Serum or Plasma",
    "Creatinine [Mass/volume] in Serum or Plasma", "D-dimer assay", "Diastolic blood pressure", "Ferritin [Mass/volume] in Serum or Plasma",
    "Fibrin D-dimer FEU [Mass/volume] in Platelet poor plasma", "Fibrinogen [Mass/volume] in Platelet poor plasma by Coagulation assay",
    "Glasgow coma scale", "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)",
    "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (MDRD)",
    "Glomerular filtration rate/1.73 sq M.predicted among blacks [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)",
    "Glucometer blood glucose", "Glucose [Mass/volume] in Blood", "Glucose [Mass/volume] in Serum or Plasma", "Glucose measurement, blood",
    "Heart rate", "INR in Platelet poor plasma by Coagulation assay", "Input/Output", "Lactate [Moles/volume] in Blood",
    "Lactate dehydrogenase [Enzymatic activity/volume] in Serum or Plasma by Lactate to pyruvate reaction", "Lactic acid measurement",
    "Left ventricular Ejection fraction", "Magnesium [Mass/volume] in Serum or Plasma", "Mean blood pressure",
    "Measurement of partial pressure of carbon dioxide in blood", "Measurement of venous partial pressure of carbon dioxide",
    "Phosphate [Mass/volume] in Serum or Plasma", "Potassium [Moles/volume] in Serum or Plasma", "Protein [Mass/volume] in Serum or Plasma",
    "Prothrombin time (PT)", "Pulse rate", "Respiratory rate", "Sodium [Moles/volume] in Blood", "Sodium [Moles/volume] in Serum or Plasma",
    "Sodium/Creatinine [Ratio] in Urine", "Systolic blood pressure", "Thrombin time", "Thyrotropin [Units/volume] in Serum or Plasma",
    "Thyroxine (T4) free [Mass/volume] in Serum or Plasma", "Triglyceride [Moles/volume] in Serum or Plasma", "Urate [Mass/volume] in Serum or Plasma",
    "Urea nitrogen [Mass/volume] in Serum or Plasma", "Venous oxygen saturation measurement", "aPTT in Platelet poor plasma by Coagulation assay",
    "aPTT.inhibitor sensitive in Platelet poor plasma by Coagulation assay",
]

# Hyperparameters (Keep as before)
SEQ_LEN = 10; PREDICTION_HORIZON = pd.Timedelta(hours=24); BATCH_SIZE = 32; D_MODEL = 128; NHEAD = 8; NUM_ENCODER_LAYERS = 3
DIM_FEEDFORWARD = 512; DROPOUT = 0.1; LEARNING_RATE = 1e-4; EPOCHS = 50; DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10; NUM_PATIENTS_TO_PLOT = 2; MEASUREMENT_INDICES_TO_PLOT = [0, 1, 3]

# --- 1. Helper Functions --- (Keep as before)
def concordance_correlation_coefficient(y_true, y_pred):
    if len(y_true) < 2 or len(y_pred) < 2: return np.nan
    cor, _ = pearsonr(y_true, y_pred); mean_true, mean_pred = np.mean(y_true), np.mean(y_pred); var_true, var_pred = np.var(y_true), np.var(y_pred)
    sd_true, sd_pred = np.std(y_true), np.std(y_pred);    if sd_true == 0 or sd_pred == 0: return np.nan
    numerator = 2 * cor * sd_true * sd_pred; denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator if denominator != 0 else np.nan

# --- 2. Data Loading and Initial Processing ---
print("\n--- [PART 1] Loading TPN & Measurement Data ---")
# ... (Keep data loading and TPN processing exactly as in the previous version) ...
tpn_df = None; meas_df = None; measurement_cols_target = []
try:
    if not os.path.exists(TPN_DATA_PATH): raise FileNotFoundError(f"TPN file not found: {TPN_DATA_PATH}")
    tpn_df = pd.read_csv(TPN_DATA_PATH, low_memory=False); tpn_df['order_inst'] = pd.to_datetime(tpn_df['order_inst'], errors='coerce')
    tpn_df['person_id'] = pd.to_numeric(tpn_df['person_id'], errors='coerce').astype('Int64'); tpn_df.dropna(subset=['order_inst', 'person_id'], inplace=True)
    if tpn_df.empty: raise ValueError("TPN DataFrame empty."); print(f"Raw TPN data loaded: {tpn_df.shape}")
    missing_dose_cols = [col for col in tpn_dose_cols if col not in tpn_df.columns]
    if missing_dose_cols: warnings.warn(f"Dose columns missing: {missing_dose_cols}. Excluding."); tpn_dose_cols = [col for col in tpn_dose_cols if col not in missing_dose_cols]
    if not tpn_dose_cols: raise ValueError("No usable dose columns.")
    missing_embed_cols = [col for col in embedding_cols if col not in tpn_df.columns]
    if missing_embed_cols: warnings.warn(f"Embedding columns missing: {missing_embed_cols}. Excluding."); embedding_cols = [col for col in embedding_cols if col not in missing_embed_cols]
    input_feature_cols = tpn_dose_cols + embedding_cols; print(f"Using {len(tpn_dose_cols)} dose and {len(embedding_cols)} embedding columns.")
    for col in input_feature_cols:
        if col not in tpn_df.columns: continue
        tpn_df[col] = pd.to_numeric(tpn_df[col], errors='coerce').fillna(0.0)
    tpn_df = tpn_df[['person_id', 'order_inst'] + input_feature_cols].sort_values(['person_id', 'order_inst']).reset_index(drop=True); print(f"Processed TPN data shape: {tpn_df.shape}")
    if not os.path.exists(ISOLATED_MEAS_PATH): raise FileNotFoundError(f"Measurement file not found: {ISOLATED_MEAS_PATH}")
    meas_df = pd.read_csv(ISOLATED_MEAS_PATH, low_memory=False); meas_df['measurement_datetime'] = pd.to_datetime(meas_df['measurement_datetime'], errors='coerce')
    meas_df['person_id'] = pd.to_numeric(meas_df['person_id'], errors='coerce').astype('Int64'); meas_df.dropna(subset=['measurement_datetime', 'person_id'], inplace=True)
    print(f"Raw Measurement data loaded (wide): {meas_df.shape}");    if meas_df.empty: raise ValueError("Measurement DataFrame empty.")
    print("Validating target measurement columns..."); valid_target_cols = []; missing_meas_cols_info = []
    for col in measurement_cols_target_placeholder:
        if col in meas_df.columns:
            meas_df[col] = pd.to_numeric(meas_df[col], errors='coerce')
            if meas_df[col].notna().any(): valid_target_cols.append(col)
            else: missing_meas_cols_info.append(f"'{col}' (All NaN)")
        else: missing_meas_cols_info.append(f"'{col}' (Missing)")
    measurement_cols_target = valid_target_cols
    if not measurement_cols_target: raise ValueError("No usable measurement columns found.")
    print(f"Using {len(measurement_cols_target)} measurement columns.");    if missing_meas_cols_info: print(f"  Excluded/missing: {missing_meas_cols_info}")
    all_meas_cols_to_keep = ['person_id', 'measurement_datetime'] + measurement_cols_target; all_meas_cols_to_keep = sorted(list(set(all_meas_cols_to_keep)), key=all_meas_cols_to_keep.index)
    meas_df = meas_df[all_meas_cols_to_keep].sort_values(['person_id', 'measurement_datetime']).reset_index(drop=True); print(f"Measurement data shape after validation: {meas_df.shape}")
    tpn_df['person_id'] = tpn_df['person_id'].astype(int); meas_df['person_id'] = meas_df['person_id'].astype(int)
except FileNotFoundError as e: print(f"ERROR: {e}"); raise
except ValueError as e: print(f"ERROR: {e}"); raise
except Exception as e: print(f"ERROR: {e}"); traceback.print_exc(); raise

# --- 3. Patient-Level Data Splitting ---
print("\n--- [PART 2] Splitting Patient IDs ---")
# ... (Keep splitting logic exactly as before) ...
all_person_ids = tpn_df['person_id'].unique(); print(f"Total unique patients: {len(all_person_ids)}")
if len(all_person_ids) < 3: raise ValueError("Not enough patients."); train_val_ids, test_ids = train_test_split(all_person_ids, test_size=0.15, random_state=42)
if len(test_ids) == 0 and len(all_person_ids) > 1: test_ids = [train_val_ids[-1]]; train_val_ids = train_val_ids[:-1]
if len(train_val_ids) < 2: raise ValueError("Not enough for train/val split."); train_ids, val_ids = train_test_split(train_val_ids, test_size=0.15 / 0.85, random_state=42)
if len(val_ids) == 0 and len(train_val_ids) > 0 : val_ids = [train_ids[-1]]; train_ids = train_ids[:-1]
if len(train_ids) == 0: raise ValueError("Not enough for train split.")
print(f"Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}, Test IDs: {len(test_ids)}")
train_ids_set = set(train_ids); val_ids_set = set(val_ids); test_ids_set = set(test_ids)

# --- 4. Calculate Measurement Means from Training Data ONLY ---
print("\n--- [PART 3] Calculating Measurement Means for Imputation (from Training Data) ---")
# ... (Keep mean calculation logic exactly as before) ...
measurement_means, imputation_values = {}, {}
meas_train_df = meas_df[meas_df['person_id'].isin(train_ids_set)]
if not meas_train_df.empty:
    print(f"Calculating means from {meas_train_df.shape[0]} measurements of {len(train_ids)} training patients.")
    measurement_means = meas_train_df[measurement_cols_target].mean().to_dict()
    for col in measurement_cols_target:
        mean_val = measurement_means.get(col)
        if pd.isna(mean_val): warnings.warn(f"Mean for '{col}' NaN. Using 0.0."); imputation_values[col] = 0.0
        else: imputation_values[col] = mean_val
else: warnings.warn("Training data empty. Using 0.0 for imputation."); imputation_values = {col: 0.0 for col in measurement_cols_target}

# --- 5. Impute Missing Measurements in the *Entire* Dataset using Training Means ---
print("\n--- [PART 4] Imputing Missing Measurements using Training Set Means ---")
# ... (Keep imputation logic exactly as before) ...
original_nan_counts = meas_df[measurement_cols_target].isna().sum()
meas_df[measurement_cols_target] = meas_df[measurement_cols_target].fillna(imputation_values)
imputed_counts = original_nan_counts - meas_df[measurement_cols_target].isna().sum(); print("Imputation complete.")
for col in measurement_cols_target:
    if original_nan_counts[col] > 0: print(f"  Imputed {imputed_counts[col]} NaN values in '{col}' using {imputation_values.get(col, 'N/A'):.4f}")

# --- 6. Sequence Creation Function ---
print("\n--- [PART 5] Defining Sequence Creation Function ---")
# --- No changes needed in function definition (TypeError fix is already included) ---
def create_sequences_with_info(tpn_data, meas_data, person_ids_set, input_cols, target_cols, seq_len, horizon):
    sequences, targets, info = [], [], []
    tpn_filtered = tpn_data[tpn_data['person_id'].isin(person_ids_set)]; meas_filtered = meas_data[meas_data['person_id'].isin(person_ids_set)]
    person_ids_in_split = tpn_filtered['person_id'].unique(); print(f"  Processing {len(person_ids_in_split)} patients...");
    for person_id in tqdm(person_ids_in_split, desc="  Creating sequences", leave=False):
        person_tpn = tpn_filtered[tpn_filtered['person_id'] == person_id].sort_values('order_inst'); person_meas = meas_filtered[meas_filtered['person_id'] == person_id].sort_values('measurement_datetime')
        if len(person_tpn) < seq_len or len(person_meas) == 0: continue
        for i in range(len(person_tpn) - seq_len + 1):
            tpn_seq_df = person_tpn.iloc[i : i + seq_len]; last_tpn_time = tpn_seq_df['order_inst'].iloc[-1]
            target_meas_options = person_meas[(person_meas['measurement_datetime'] > last_tpn_time) & (person_meas['measurement_datetime'] <= last_tpn_time + horizon)]
            if not target_meas_options.empty:
                first_target_meas_series = target_meas_options.iloc[0]; input_sequence_data = tpn_seq_df[input_cols].values
                target_data = first_target_meas_series[target_cols].values; target_time = first_target_meas_series['measurement_datetime']
                target_data = pd.to_numeric(target_data, errors='coerce') # Force numeric for isnan check
                if np.isnan(target_data).any(): warnings.warn(f"NaN found for person {person_id} time {target_time} AFTER coercion."); target_data = np.nan_to_num(target_data, nan=0.0)
                target_data = target_data.astype(np.float32) # Ensure correct type
                sequences.append(input_sequence_data); targets.append(target_data); info.append({'person_id': person_id, 'target_time': target_time})
    if not sequences:
        warnings.warn("No sequences created."); num_input_features, num_target_features = len(input_cols), len(target_cols)
        return np.empty((0, seq_len, num_input_features), dtype=np.float32), np.empty((0, num_target_features), dtype=np.float32), []
    sequences_np, targets_np = np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)
    print(f"  Created {len(sequences_np)} sequences."); return sequences_np, targets_np, info

# --- 7. Create Sequences for Each Split ---
print("\n--- [PART 6] Creating Sequences for Train/Val/Test Splits ---")
# --- No changes needed ---
X_train, y_train, info_train = create_sequences_with_info(tpn_df, meas_df, train_ids_set, input_feature_cols, measurement_cols_target, SEQ_LEN, PREDICTION_HORIZON)
X_val, y_val, info_val = create_sequences_with_info(tpn_df, meas_df, val_ids_set, input_feature_cols, measurement_cols_target, SEQ_LEN, PREDICTION_HORIZON)
X_test, y_test, info_test = create_sequences_with_info(tpn_df, meas_df, test_ids_set, input_feature_cols, measurement_cols_target, SEQ_LEN, PREDICTION_HORIZON)
if X_train.size == 0 or X_val.size == 0 or X_test.size == 0: warnings.warn("Warning: Zero sequences in one or more splits.")
print(f"Train shapes: X={X_train.shape}, y={y_train.shape}"); print(f"Val shapes: X={X_val.shape}, y={y_val.shape}"); print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

# --- 8. Data Scaling ---
print("\n--- [PART 7] Scaling Data ---")
# --- No changes needed ---
input_scaler, target_scaler = StandardScaler(), StandardScaler()
if X_train.size > 0:
    n_samples_train, seq_len_train, n_features_train = X_train.shape; X_train_reshaped = X_train.reshape(-1, n_features_train)
    input_scaler.fit(X_train_reshaped); X_train_scaled = input_scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = input_scaler.transform(X_val.reshape(-1, n_features_train)).reshape(X_val.shape) if X_val.size > 0 else X_val
    X_test_scaled = input_scaler.transform(X_test.reshape(-1, n_features_train)).reshape(X_test.shape) if X_test.size > 0 else X_test
else: warnings.warn("Train X empty, skipping scaling."); X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test
if y_train.size > 0:
    target_scaler.fit(y_train); y_train_scaled = target_scaler.transform(y_train)
    y_val_scaled = target_scaler.transform(y_val) if y_val.size > 0 else y_val
    y_test_scaled = target_scaler.transform(y_test) if y_test.size > 0 else y_test
else: warnings.warn("y_train empty, skipping target scaler."); y_train_scaled, y_val_scaled, y_test_scaled = y_train, y_val, y_test
print("Data scaling complete.")


# --- 9. PyTorch Dataset and DataLoader ---
print("\n--- [PART 8] Creating PyTorch Datasets and DataLoaders ---")

# ---> Define Custom Collate Function <---
def custom_collate_fn(batch):
    """
    Collate function to handle batching of sequences, targets, and info dicts.
    The info dicts (containing Timestamps) are kept as a list.
    """
    sequences = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    infos = [item[2] for item in batch] # Collect info dicts into a list

    # Use default_collate for sequences and targets if they are already tensors
    # Otherwise, stack them manually if needed (Dataset already returns tensors here)
    sequences_collated = torch.stack(sequences, 0)
    targets_collated = torch.stack(targets, 0)

    return sequences_collated, targets_collated, infos # Return list for infos

class TPNPredictionDatasetInfo(Dataset):
    def __init__(self, sequences, targets, info_list):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        # Store info as list of dicts, containing potentially non-collated types like Timestamp
        self.info = info_list
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        # Return the tuple as before
        return self.sequences[idx], self.targets[idx], self.info[idx]

# ---> Use custom_collate_fn in DataLoaders <---
train_dataset = TPNPredictionDatasetInfo(X_train_scaled, y_train_scaled, info_train) if X_train.size > 0 else None
val_dataset = TPNPredictionDatasetInfo(X_val_scaled, y_val_scaled, info_val) if X_val.size > 0 else None
test_dataset = TPNPredictionDatasetInfo(X_test_scaled, y_test_scaled, info_test) if X_test.size > 0 else None

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn) if train_dataset else None
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn) if val_dataset else None
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn) if test_dataset else None

print("DataLoaders created with custom collate function.")


# --- 10. Transformer Model Definition ---
print("\n--- [PART 9] Defining the Transformer Model ---")
# --- No changes needed ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000): super(PositionalEncoding, self).__init__(); self.dropout = nn.Dropout(p=dropout); pe = torch.zeros(max_len, d_model); position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1); div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)); pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term); pe = pe.unsqueeze(0).transpose(0, 1); self.register_buffer('pe', pe)
    def forward(self, x): return self.dropout(x + self.pe[:x.size(0), :])
class TPNTransformerPredictor(nn.Module):
    def __init__(self, input_dim, target_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, seq_len): super().__init__(); self.d_model, self.seq_len = d_model, seq_len; self.input_embed = nn.Linear(input_dim, d_model); encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True); self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers); self.output_layer = nn.Linear(d_model, target_dim); self.init_weights()
    def init_weights(self): initrange = 0.1; self.input_embed.weight.data.uniform_(-initrange, initrange); self.input_embed.bias.data.zero_(); self.output_layer.weight.data.uniform_(-initrange, initrange); self.output_layer.bias.data.zero_()
    def _generate_positional_encoding(self, seq_len, d_model, device): pe = torch.zeros(seq_len, d_model, device=device); position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1); div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device); pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term); return pe.unsqueeze(0)
    def forward(self, src): src_embedded = self.input_embed(src) * math.sqrt(self.d_model); pe = self._generate_positional_encoding(self.seq_len, self.d_model, src.device); src_embedded = src_embedded + pe; transformer_output = self.transformer_encoder(src_embedded); last_token_output = transformer_output[:, -1, :]; output = self.output_layer(last_token_output); return output

if X_train.size > 0 and y_train.size > 0: input_dim, target_dim = X_train_scaled.shape[2], y_train_scaled.shape[1]; model = TPNTransformerPredictor(input_dim=input_dim, target_dim=target_dim, d_model=D_MODEL, nhead=NHEAD, num_encoder_layers=NUM_ENCODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, seq_len=SEQ_LEN).to(DEVICE); print(f"Model instantiated on {DEVICE}:"); total_params = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"Total trainable parameters: {total_params:,}")
else: model = None; print("Skipping model instantiation.")

# --- 11. Training Setup ---
print("\n--- [PART 10] Setting up Training ---")
# --- No changes needed ---
if model and train_loader: criterion = nn.MSELoss(); optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE); training_possible = True
else: criterion, optimizer, training_possible = None, None, False; print("Skipping training setup.")

# --- 12. Training Loop ---
print("\n--- [PART 11] Starting Training Loop ---")
# --- Loops now correctly unpack data, target, info_batch from custom collate ---
best_val_loss = float('inf'); epochs_no_improve = 0
if training_possible and val_loader :
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True); print(f"Plots will be saved to: {PLOT_SAVE_DIR}")
    for epoch in range(EPOCHS):
        model.train(); train_loss = 0.0; train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        # ---> Unpack includes info_batch now, but it's ignored here (_) <---
        for batch_idx, (data, target, _) in enumerate(train_loop):
            data, target = data.to(DEVICE), target.to(DEVICE); optimizer.zero_grad(); output = model(data); loss = criterion(output, target)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step(); train_loss += loss.item()
            if batch_idx % 50 == 0: train_loop.set_postfix(loss=loss.item())
        avg_train_loss = train_loss / len(train_loader)
        model.eval(); val_loss = 0.0; val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validate]", leave=False)
        with torch.no_grad():
             # ---> Unpack includes info_batch now, but it's ignored here (_) <---
            for data, target, _ in val_loop:
                data, target = data.to(DEVICE), target.to(DEVICE); output = model(data); loss = criterion(output, target); val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader); print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss; epochs_no_improve = 0
            model_save_path = 'best_transformer_model_patient_split.pth' # Keep saving model locally
            torch.save(model.state_dict(), model_save_path); print(f"Saved best model to {model_save_path}")
        else: epochs_no_improve += 1; print(f"Val loss did not improve for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= PATIENCE: print(f"Early stopping after {epoch+1} epochs."); break
    print("Training finished.")
else: print("Training skipped.")

# --- 13. Evaluation on Test Set ---
print("\n--- [PART 12] Evaluating on Test Set ---")
# --- No changes needed except plot saving path ---
model_load_path = 'best_transformer_model_patient_split.pth'
if model and os.path.exists(model_load_path): print(f"Loading best model weights from {model_load_path}"); model.load_state_dict(torch.load(model_load_path))
elif model: warnings.warn("Best model file not found.")
else: print("Skipping evaluation: Model not available.")

if model and test_loader:
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True) # Ensure dir exists
    model.eval(); test_loss_scaled = 0.0; all_preds_scaled, all_targets_scaled, all_test_info = [], [], []
    with torch.no_grad():
        # ---> Unpack includes info_batch and use it <---
        for data, target, info_batch in tqdm(test_loader, desc="Testing"):
            data, target = data.to(DEVICE), target.to(DEVICE); output = model(data); loss = criterion(output, target); test_loss_scaled += loss.item()
            all_preds_scaled.append(output.cpu().numpy()); all_targets_scaled.append(target.cpu().numpy())
            all_test_info.extend(info_batch) # Collect the list of info dicts
    avg_test_loss_scaled = test_loss_scaled / len(test_loader); print(f"Test Loss (Scaled): {avg_test_loss_scaled:.6f}")
    preds_scaled_np = np.concatenate(all_preds_scaled, axis=0); targets_scaled_np = np.concatenate(all_targets_scaled, axis=0)
    if hasattr(target_scaler, 'n_features_in_'): preds_original, targets_original = target_scaler.inverse_transform(preds_scaled_np), target_scaler.inverse_transform(targets_scaled_np)
    else: warnings.warn("Target scaler not fitted."); preds_original, targets_original = preds_scaled_np, targets_scaled_np
    print("\nCalculating evaluation metrics..."); metrics = {'RMSE': [], 'MAE': [], 'R2': [], 'PearsonR': [], 'CCC': []}; per_target_metrics = {}
    num_targets = targets_original.shape[1]
    for i in range(num_targets):
        target_name = measurement_cols_target[i]; y_true, y_pred = targets_original[:, i], preds_original[:, i]
        if np.std(y_true) < 1e-6: warnings.warn(f"Zero variance in '{target_name}'."); rmse, mae, r2, pearson_r, ccc = np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred), np.nan, np.nan, np.nan
        else: rmse, mae, r2 = np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred); pearson_r, _ = pearsonr(y_true, y_pred); ccc = concordance_correlation_coefficient(y_true, y_pred)
        metrics['RMSE'].append(rmse); metrics['MAE'].append(mae); metrics['R2'].append(r2); metrics['PearsonR'].append(pearson_r); metrics['CCC'].append(ccc); per_target_metrics[target_name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'PearsonR': pearson_r, 'CCC': ccc}
    avg_metrics = {key: np.nanmean(values) for key, values in metrics.items()}; print("\n--- Overall Average Test Metrics ---"); [print(f"{key}: {value:.4f}") for key, value in avg_metrics.items()]; print("------------------------------------")
    plt.figure(figsize=(10, 6)); metrics_to_plot = {k: v for k, v in avg_metrics.items() if not np.isnan(v)}
    plt.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color=['blue', 'green', 'red', 'purple', 'orange']); plt.title('Overall Average Test Set Metrics'); plt.ylabel('Value'); plt.ylim(bottom=min(0, min(metrics_to_plot.values())-0.1), top=max(metrics_to_plot.values())*1.1); plt.grid(axis='y', linestyle='--')
    overall_metrics_save_path = os.path.join(PLOT_SAVE_DIR, 'overall_test_metrics.png'); plt.savefig(overall_metrics_save_path); plt.close(); print(f"Saved overall metrics plot to: {overall_metrics_save_path}")

    print("\n--- [PART 13] Generating Plots for Sample Test Patients ---")
    results_df = pd.DataFrame({'person_id': [info['person_id'] for info in all_test_info], 'target_time': [info['target_time'] for info in all_test_info]}) # Use collected info
    for i, col_name in enumerate(measurement_cols_target): results_df[f'target_{col_name}'] = targets_original[:, i]; results_df[f'pred_{col_name}'] = preds_original[:, i]
    results_df = results_df.sort_values(['person_id', 'target_time']); test_person_ids_with_results = results_df['person_id'].unique()
    if len(test_person_ids_with_results) > 0:
        num_available_patients = len(test_person_ids_with_results); num_to_plot = min(NUM_PATIENTS_TO_PLOT, num_available_patients)
        if num_to_plot < NUM_PATIENTS_TO_PLOT: warnings.warn(f"Plotting {num_to_plot} instead of {NUM_PATIENTS_TO_PLOT} patients.")
        if num_to_plot > 0 :
            random_patient_ids = random.sample(list(test_person_ids_with_results), k=num_to_plot); print(f"Plotting results for patients: {random_patient_ids}")
            for patient_id in random_patient_ids:
                patient_results = results_df[results_df['person_id'] == patient_id]
                if not patient_results.empty:
                    plot_cols_indices = [idx for idx in MEASUREMENT_INDICES_TO_PLOT if idx < len(measurement_cols_target)]; num_plots = len(plot_cols_indices)
                    if num_plots > 0:
                        plt.figure(figsize=(15, 5 * ((num_plots + 1) // 2))); plt.suptitle(f'Patient {patient_id} - Predicted vs. Ground Truth', fontsize=16); plot_num = 1
                        for meas_idx in plot_cols_indices:
                            meas_name = measurement_cols_target[meas_idx]; target_col, pred_col = f'target_{meas_name}', f'pred_{meas_name}'; plt.subplot((num_plots + 1) // 2, 2, plot_num)
                            plt.plot(patient_results['target_time'], patient_results[target_col], 'bo-', label='Ground Truth', markersize=4); plt.plot(patient_results['target_time'], patient_results[pred_col], 'rx--', label='Prediction', markersize=4)
                            plt.xlabel('Time'); plt.ylabel('Value'); plt.title(meas_name, fontsize=10); plt.xticks(rotation=45); plt.legend(); plt.grid(True, linestyle=':'); plot_num += 1
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); patient_plot_save_path = os.path.join(PLOT_SAVE_DIR, f'patient_{patient_id}_predictions.png'); plt.savefig(patient_plot_save_path); plt.close(); print(f"Saved plot for patient {patient_id} to: {patient_plot_save_path}")
                    else: print(f"No valid measurements to plot for patient {patient_id}.")
        else: print("Cannot plot patients.")
    else: print("Skipping patient visualization.")
else: print("Skipping Test Set evaluation.")

print("\n--- Script Finished ---")