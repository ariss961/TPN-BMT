 import numpy as np

import pandas as pd

import tensorflow as tf  # type: ignore

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential  # type: ignore

from tensorflow.keras.layers import Dense, LSTM, Dropout, Masking, Activation  # type: ignore

# --- MODIFICATION: Import AdamW ---

from tensorflow.keras.optimizers import Adam, AdamW  # type: ignore

from sklearn.preprocessing import StandardScaler, MinMaxScaler # Added MinMaxScaler

from sklearn.model_selection import train_test_split # Still needed for splitting IDs

from sklearn.metrics import r2_score

import os

from scipy.stats import pearsonr

from datetime import datetime, timedelta # Keep for potential type hints/checks

import logging # Added for better logging



# --- Basic Configuration ---

# Set up logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Specify GPU (adjust if needed)

# Example: "0" for first GPU, "1" for second, "" for CPU

os.environ["CUDA_VISIBLE_DEVICES"] = "1"



# Configure GPU memory growth

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')

        logging.info(f"Available GPUs: {len(gpus)} Physical, {len(logical_gpus)} Logical")

    except Exception as e:

        logging.error(f"Error setting GPU memory growth: {e}")

else:

    logging.warning("No GPUs detected by TensorFlow.")



# Create an output directory for saving models and plots

# Consider updating the dir name if needed

OUTPUT_DIR = "/remote/home/ariss01/TPN/Final_file/LSTM_results_active" # Updated dir name

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.info(f"Output directory set to: {OUTPUT_DIR}")



# --- Feature and Target Columns ---

# Define the feature and target columns

# --- IMPORTANT: PASTE YOUR FULL feature_cols LIST HERE ---

feature_cols =  [

        "25-Hydroxyvitamin D3+25-Hydroxyvitamin D2 [Mass/volume] in Serum or Plasma",

        "25-hydroxyvitamin D3 [Mass/volume] in Serum or Plasma",

        "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma",

        "Albumin [Mass/volume] in Serum or Plasma",

        "Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma",

        "Amylase [Enzymatic activity/volume] in Serum or Plasma",

        "Anion gap in Serum or Plasma",

        "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma",

        "Base excess in Blood by calculation",

        "Bicarbonate [Moles/volume] in Specimen",

        "Bicarbonate measurement",

        "Bilirubin.direct [Mass/volume] in Serum or Plasma",

        "Bilirubin.indirect [Mass/volume] in Serum or Plasma",

        "Bilirubin.total [Mass/volume] in Serum or Plasma",

        "Blood potassium measurement",

        "Body height",

        "Body mass index (BMI) [Ratio]",

        "Body surface area",

        "Body temperature",

        "Body weight",

        "Calciferol (Vit D2) [Mass/volume] in Serum or Plasma",

        "Calcium [Mass/volume] in Serum or Plasma",

        "Calcium.ionized [Mass/volume] in Serum or Plasma",

        "Calcium.ionized [Moles/volume] in Serum or Plasma",

        "Carbon dioxide [Partial pressure] adjusted to patient's actual temperature in Blood",

        "Carbon dioxide [Partial pressure] in Venous blood",

        "Carbon dioxide, total [Moles/volume] in Serum or Plasma",

        "Central venous pressure (CVP)",

        "Chloride [Moles/volume] in Serum or Plasma",

        "Chloride measurement, blood",

        "Cholesterol [Mass/volume] in Serum or Plasma",

        "Cholesterol in HDL [Mass/volume] in Serum or Plasma",

        "Cholesterol.total/Cholesterol in HDL [Mass Ratio] in Serum or Plasma",

        "Cobalamin (Vitamin B12) [Mass/volume] in Serum or Plasma",

        "Creatine kinase [Enzymatic activity/volume] in Serum or Plasma",

        "Creatinine [Mass/volume] in Serum or Plasma",

        "D-dimer assay",

        "Diastolic blood pressure",

        "Ferritin [Mass/volume] in Serum or Plasma",

        "Fibrin D-dimer FEU [Mass/volume] in Platelet poor plasma",

        "Fibrinogen [Mass/volume] in Platelet poor plasma by Coagulation assay",

        "Glasgow coma scale",

        "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)",

        "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)",

        "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (MDRD)",

        "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (MDRD)",

        "Glomerular filtration rate/1.73 sq M.predicted among blacks [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)",

        "Glucometer blood glucose",

        "Glucose [Mass/volume] in Blood",

        "Glucose [Mass/volume] in Serum or Plasma",

        "Glucose measurement, blood",

        "Heart rate",

        "INR in Platelet poor plasma by Coagulation assay",

        "Input/Output",

        "Lactate [Moles/volume] in Blood",

        "Lactate dehydrogenase [Enzymatic activity/volume] in Serum or Plasma by Lactate to pyruvate reaction",

        "Lactic acid measurement",

        "Left ventricular Ejection fraction",

        "Magnesium [Mass/volume] in Serum or Plasma",

        "Mean blood pressure",

        "Measurement of partial pressure of carbon dioxide in blood",

        "Measurement of venous partial pressure of carbon dioxide",

        "Phosphate [Mass/volume] in Serum or Plasma",

        "Potassium [Moles/volume] in Serum or Plasma",

        "Protein [Mass/volume] in Serum or Plasma",

        "Prothrombin time (PT)",

        "Pulse rate",

        "Respiratory rate",

        "Sodium [Moles/volume] in Blood",

        "Sodium [Moles/volume] in Serum or Plasma",

        "Sodium/Creatinine [Ratio] in Urine",

        "Systolic blood pressure",

        "Thrombin time",

        "Thyrotropin [Units/volume] in Serum or Plasma",

        "Thyroxine (T4) free [Mass/volume] in Serum or Plasma",

        "Triglyceride [Moles/volume] in Serum or Plasma",

        "Urate [Mass/volume] in Serum or Plasma",

        "Urea nitrogen [Mass/volume] in Serum or Plasma",

        "Venous oxygen saturation measurement",

        "Volume expired",

        "aPTT in Platelet poor plasma by Coagulation assay",

        "aPTT.inhibitor sensitive in Platelet poor plasma by Coagulation assay",

        ]+ [f"embedding_{i}" for i in range(769)]







target_cols = ['Parenteral Amino Acids (g)', 'Acetate (mEq)', 'Calcium (mEq)', 'Chloride (mEq)', 'Chromium (mcg)', 'Copper (mg)',

               'Dextrose (g)', 'Famotidine (mg)', 'Folic Acid (mg)', 'Heparin (units)', 'Insulin (units)', 'Levocarnitine (mg)',

               'Magnesium (mEq)', 'Manganese (mg)', 'Multivitamins (units)', 'Phosphate (mmol)', 'Potassium (mEq)',

               'Selenium (mcg)', 'Sodium (mEq)', 'Thiamine (mg)', 'Pyridoxine (mg)', 'Ascorbic Acid (Vitamin C) (mg)',

               'Phytonadione (mg)', 'Zinc (mg)',

               'lipids(g)']



# --- Data Loading and Preprocessing ---



def load_and_preprocess_data(data_path):

    """

    Load and preprocess the dataset.



    For feature columns:

        - Missing values are filled using the median value.

        - Scaled using StandardScaler.

    For target columns:

        - Missing values are replaced with 0 (CAUTION: Verify this assumption).

        - Scaled using MinMaxScaler to [0, 1].

    """

    logging.info(f"Loading data from {data_path}")

    try:

        df = pd.read_csv(data_path)

    except FileNotFoundError:

        logging.error(f"Data file not found at {data_path}")

        raise



    logging.info("Converting 'order_inst' to datetime")

    df['order_inst'] = pd.to_datetime(df['order_inst'], errors='coerce')

    if df['order_inst'].isnull().any():

        logging.warning(f"Found {df['order_inst'].isnull().sum()} null/invalid dates in 'order_inst' after conversion.")

    logging.info(f"Data type of 'order_inst' column after conversion: {df['order_inst'].dtype}")

    if not df['order_inst'].empty:

        logging.info(f"Sample 'order_inst' values: {df['order_inst'].head().to_list()}")



    # --- Imputation ---

    logging.info("Imputing missing values...")

    actual_feature_cols = [col for col in feature_cols if col in df.columns]

    duplicates = pd.Series(actual_feature_cols).value_counts()

    duplicates = duplicates[duplicates > 1]

    if not duplicates.empty:

        logging.warning(f"Duplicate columns found in feature_cols and DataFrame: {duplicates.index.tolist()}")



    missing_feature_cols = [col for col in feature_cols if col not in df.columns]

    if missing_feature_cols:

        logging.warning(f"Features defined in feature_cols but not found in DataFrame: {missing_feature_cols}")



    for col in actual_feature_cols:

        if df[col].isnull().any():

            if df[col].notnull().any():

                median_val = df[col].median()

                df[col] = df[col].fillna(median_val)

            else:

                df[col] = df[col].fillna(0)

                logging.warning(f"Feature column '{col}' was entirely null. Filled with 0.")





    actual_target_cols = [col for col in target_cols if col in df.columns]

    missing_target_cols = [col for col in target_cols if col not in df.columns]

    if missing_target_cols:

        logging.warning(f"Targets defined in target_cols but not found in DataFrame: {missing_target_cols}")



    for col in actual_target_cols:

        if df[col].isnull().any():

            missing_pct = df[col].isnull().mean() * 100

            if missing_pct > 0:

                logging.warning(f"Target column '{col}' has {missing_pct:.2f}% missing values. Filling with 0 - VERIFY THIS IS CORRECT.")

            df[col] = df[col].fillna(0)



    logging.info("Extracting raw features and targets.")

    valid_feature_cols = [col for col in feature_cols if col in df.columns]

    valid_target_cols = [col for col in target_cols if col in df.columns]



    if not valid_feature_cols:

        raise ValueError("No valid feature columns found in the DataFrame based on feature_cols list.")

    if not valid_target_cols:

        raise ValueError("No valid target columns found in the DataFrame based on target_cols list.")



    raw_features = df[valid_feature_cols].values

    raw_targets = df[valid_target_cols].values



    # --- Scaling ---

    logging.info("Scaling features (StandardScaler) and targets (MinMaxScaler).")

    feature_scaler = MinMaxScaler()

    target_scaler = MinMaxScaler()



    scaled_features = feature_scaler.fit_transform(raw_features)

    scaled_targets = target_scaler.fit_transform(raw_targets)



    scalers = {'feature_scaler': feature_scaler, 'target_scaler': target_scaler}



    # Check for NaNs/Infs *after* scaling

    if np.any(np.isnan(scaled_features)): logging.error("NaNs found in scaled_features!")

    if np.any(np.isinf(scaled_features)): logging.error("Infs found in scaled_features!")

    if np.any(np.isnan(scaled_targets)): logging.error("NaNs found in scaled_targets!")

    if np.any(np.isinf(scaled_targets)): logging.error("Infs found in scaled_targets!")



    return scaled_features, scaled_targets, df['person_id'].values, df['order_inst'].values, scalers, valid_feature_cols, valid_target_cols





# --- Sequence Creation ---



def create_sequences(features, targets, person_ids, order_insts,

                      feature_names, target_names,

                      sequence_length=500, max_gap_hours=72):

    """

    Create sequences using the NumPy timedelta calculation method.

    Uses Right-Padding (post-padding).

    """

    logging.info(f"Creating sequences with sequence_length={sequence_length}, max_gap_hours={max_gap_hours}")

    logging.info("Using NumPy timedelta division for time difference calculation.") # Note the method used

    X_seq = []

    y_seq = []

    seq_lengths = []

    seq_patient_ids = []



    unique_person_ids = np.unique(person_ids)

    logging.info(f"Total unique patients found: {len(unique_person_ids)}")

    patients_processed = 0

    patients_with_sequences = 0

    total_sequences_created = 0



    pad_value = 0.0



    n_features = len(feature_names)

    n_targets = len(target_names)

    combined_feature_dim = n_features + n_targets

    logging.info(f"Original feature dim: {n_features}, Target dim: {n_targets}, Combined sequence feature dim: {combined_feature_dim}")





    for person_id in unique_person_ids:

        patients_processed += 1

        if patients_processed % 500 == 0:

            logging.info(f"Processing patient {patients_processed}/{len(unique_person_ids)}")



        person_mask = person_ids == person_id

        person_features = features[person_mask]

        person_targets = targets[person_mask]

        person_order_insts = order_insts[person_mask]



        if len(person_features) > 1:

            # Sort data points for the current patient based on timestamp

            patient_data = sorted(list(zip(person_order_insts, person_features, person_targets)),

                                  key=lambda x: x[0] if pd.notnull(x[0]) else pd.Timestamp.max)



            sequences_for_patient = []

            # Find the first index with a valid timestamp

            first_valid_idx = next((i for i, x in enumerate(patient_data) if pd.notnull(x[0])), -1)



            if first_valid_idx != -1 and first_valid_idx < len(patient_data) - 1:

                current_seq_tuples = [patient_data[first_valid_idx]] # Start sequence with first valid data point



                # Iterate through the rest of the patient's data points

                for i in range(first_valid_idx + 1, len(patient_data)):

                    current_time = patient_data[i][0]

                    prev_time = patient_data[i-1][0]



                    # --- CALCULATE TIME DIFFERENCE ---

                    time_diff = float('inf') # Default to break sequence

                    if pd.notnull(current_time) and pd.notnull(prev_time):

                        try:

                            delta = np.datetime64(current_time) - np.datetime64(prev_time)

                            # Convert timedelta to hours

                            time_diff = delta / np.timedelta64(1, 'h')

                            if not np.isfinite(time_diff):

                                logging.warning(f"Timestamp difference resulted in non-finite value "

                                                f"for patient {person_id} (step {i}). Assuming break.")

                                time_diff = float('inf')

                        except Exception as e:

                            logging.warning(f"Timestamp difference calculation error for patient {person_id} "

                                            f"(step {i}): {e}. Assuming break.")

                            time_diff = float('inf')

                    # --- END OF TIME DIFFERENCE CALCULATION ---



                    # Check if the time gap is within the allowed maximum

                    if 0 <= time_diff <= max_gap_hours:

                        current_seq_tuples.append(patient_data[i]) # Add to current sequence

                    else:

                        # If gap is too large or invalid, finalize the previous sequence (if long enough)

                        if len(current_seq_tuples) > 1:

                            sequences_for_patient.append(current_seq_tuples)

                        # Start a new sequence if the current timestamp is valid

                        if pd.notnull(current_time):

                            current_seq_tuples = [patient_data[i]]

                        else: # If current time is also invalid, reset

                            current_seq_tuples = []



                # Add the last sequence if it's valid

                if len(current_seq_tuples) > 1:

                    sequences_for_patient.append(current_seq_tuples)



                # Process all valid sequences found for this patient

                if sequences_for_patient:

                    patients_with_sequences += 1

                    for seq_tuples in sequences_for_patient:

                        # Extract features and targets from the sequence tuples

                        sorted_features = np.array([x[1] for x in seq_tuples], dtype=np.float32)

                        sorted_targets = np.array([x[2] for x in seq_tuples], dtype=np.float32)



                        # --- Create the combined input sequence ---

                        # Shape: (timesteps, n_features + n_targets)

                        # Features at time t, Targets from time t-1

                        enhanced_sequence = np.full((len(sorted_features), combined_feature_dim), pad_value, dtype=np.float32)

                        enhanced_sequence[:, :n_features] = sorted_features      # Current features

                        enhanced_sequence[1:, n_features:] = sorted_targets[:-1] # Previous targets (shifted)

                        # The first timestep has no previous target, remains padded with pad_value (0.0)



                        # The target for this sequence is the target at the *last* step

                        current_y = sorted_targets[-1]



                        # --- Apply RIGHT-PADDING (Post-Padding) ---

                        actual_length = len(enhanced_sequence)

                        if actual_length > sequence_length:

                            # Truncate: take the last 'sequence_length' steps

                            padded_sequence = enhanced_sequence[-sequence_length:]

                            final_length = sequence_length # Effective length is the max length

                        elif actual_length < sequence_length:

                            # Pad at the END (Right-Padding / Post-Padding)

                            pad_width = sequence_length - actual_length

                            padding = np.full((pad_width, combined_feature_dim), pad_value, dtype=np.float32)

                            padded_sequence = np.vstack((enhanced_sequence, padding)) # Sequence first, then padding

                            final_length = actual_length # Store original length before padding

                        else:

                            # Exact length

                            padded_sequence = enhanced_sequence

                            final_length = sequence_length # Effective length is the max length

                        # --- End Padding Modification ---



                        X_seq.append(padded_sequence)

                        y_seq.append(current_y)

                        seq_lengths.append(final_length) # Store original length (or max length if truncated)

                        seq_patient_ids.append(person_id)

                        total_sequences_created += 1





    logging.info(f"Finished sequence creation.")

    logging.info(f"Patients with sequences processed: {patients_with_sequences}/{len(unique_person_ids)}")

    logging.info(f"Total sequences created: {total_sequences_created}")



    if not X_seq:

        logging.error("Failed to create any sequences.")

        logging.error("Possible reasons:")

        logging.error(f" - max_gap_hours ({max_gap_hours}) might genuinely be too small for the data frequency.")

        logging.error(" - Data might have < 2 valid consecutive timestamps per patient.")

        logging.error(" - Check 'order_inst' conversion and if timestamps are present and sortable.")

        logging.error(" - Underlying issues with data integrity (NaNs?) preventing sequence formation.")

        raise ValueError("No sequences could be created. Check data, max_gap_hours, and timestamp handling. See logs.")



    return np.array(X_seq), np.array(y_seq), np.array(seq_lengths), np.array(seq_patient_ids)





# --- Model Building ---



def build_lstm_model(sequence_length, n_combined_features, n_targets):

    """

    Build and compile a single-layer LSTM model with masking. (MODIFIED)



    Uses standard MSE loss with AdamW optimizer and ReLU activation for non-negativity.

    Compatible with right-padding.

    """

    # Updated log message to reflect the change

    logging.info("Building 1-Layer LSTM model with AdamW...")

    model = Sequential(name="LSTM_1_Layer_Model") # Give the model a name



    # Masking layer handles the right-padded sequences (mask_value=0.0)

    # Input shape is defined here

    model.add(Masking(mask_value=0.0,

                      input_shape=(sequence_length, n_combined_features),

                      name="Masking_Layer"))



    # --- Single LSTM Layer ---

    # Removed the previous LSTM layers (1024, 512, 128 units).

    # This is now the only LSTM layer.

    # return_sequences=False (default) because it's the last LSTM layer before Dense.

    # Units can be tuned; starting with 128.

    model.add(LSTM(2048, name="LSTM_Layer_1")) # Using 128 units as a starting point

    model.add(Dropout(0.3, name="Dropout_Layer")) # Keep dropout for regularization



    # --- Output Layer ---

    # Dense layer to map LSTM output to the desired number of target predictions

    model.add(Dense(n_targets, name="Output_Dense"))

    # ReLU activation ensures non-negative outputs

    model.add(Activation('relu', name="Output_Activation_ReLU"))



    # --- Compilation ---

    # Compile with standard MSE loss and AdamW optimizer

    optimizer = AdamW(

        learning_rate=1e-4, # Consider tuning LR

        weight_decay=1e-1   # Common default for AdamW, tune if needed

    )

    model.compile(optimizer=optimizer,

                  loss='mean_squared_error',

                  metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])



    logging.info("Model built successfully:")

    # Use print_fn=logging.info to capture the summary in logs

    model.summary(print_fn=logging.info)



    return model



# --- Model Training ---



def train_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, sequence_length, n_combined_features, n_targets):

    """

    Train the LSTM model. Includes callbacks and NaN/Inf checks.

    Uses validation_data tuple which handles empty validation set correctly in .fit().

    """

    logging.info("Building and training model...")

    model = build_lstm_model(sequence_length, n_combined_features, n_targets) # Calls the modified build function



    early_stopping = tf.keras.callbacks.EarlyStopping(

        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1

    )

    # Update checkpoint filename if desired

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(

        os.path.join(OUTPUT_DIR, 'best_lstm_model_3layer_adamw.keras'), # Updated checkpoint name

        monitor='val_loss', save_best_only=True, mode='min', verbose=1

    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

        monitor='val_loss', factor=0.7, patience=5, min_lr=1e-6, verbose=1

    )



    # --- Add NaN/Inf checks before training ---

    logging.info(f"Checking for NaNs/Infs in training/validation data before fitting...")

    data_valid = True

    # Check train data first

    if np.any(np.isnan(X_train_seq)): logging.error("FATAL: NaNs found in X_train_seq!"); data_valid = False

    if np.any(np.isinf(X_train_seq)): logging.error("FATAL: Infs found in X_train_seq!"); data_valid = False

    if np.any(np.isnan(y_train_seq)): logging.error("FATAL: NaNs found in y_train_seq!"); data_valid = False

    if np.any(np.isinf(y_train_seq)): logging.error("FATAL: Infs found in y_train_seq!"); data_valid = False

    # Check validation data only if it exists

    if X_val_seq is not None and X_val_seq.size > 0:

        if np.any(np.isnan(X_val_seq)): logging.error("FATAL: NaNs found in X_val_seq!"); data_valid = False

        if np.any(np.isinf(X_val_seq)): logging.error("FATAL: Infs found in X_val_seq!"); data_valid = False

    if y_val_seq is not None and y_val_seq.size > 0:

        if np.any(np.isnan(y_val_seq)): logging.error("FATAL: NaNs found in y_val_seq!"); data_valid = False

        if np.any(np.isinf(y_val_seq)): logging.error("FATAL: Infs found in y_val_seq!"); data_valid = False



    if not data_valid:

        raise ValueError("NaNs or Infs detected in training/validation data. Cannot proceed.")

    logging.info("Data checks passed.")

    # --- End checks ---



    # Prepare validation data argument for fit

    validation_data_arg = (X_val_seq, y_val_seq) if (X_val_seq is not None and X_val_seq.size > 0) else None



    logging.info("Starting model training...")

    history = model.fit(X_train_seq, y_train_seq,

                        epochs=300, # Adjust epochs if needed

                        batch_size=128, # Adjust batch size if needed

                        validation_data=validation_data_arg, # Pass tuple or None

                        callbacks=[early_stopping, model_checkpoint, reduce_lr],

                        verbose=1)



    logging.info("Model training finished.")

    logging.info("Loading best weights based on validation loss (if validation occurred).")

    best_model_path = os.path.join(OUTPUT_DIR, 'best_lstm_model_3layer_adamw.keras') # Use updated checkpoint name

    if os.path.exists(best_model_path):

        model.load_weights(best_model_path)

        logging.info(f"Loaded weights from {best_model_path}")

    else:

        logging.warning(f"Best model checkpoint file not found at {best_model_path}. Using final model weights.")



    return model, history





# --- Plotting and Evaluation ---



def plot_training_metrics(history):

    """ Plot training and validation metrics: Loss, MAE, and RMSE. """

    logging.info("Plotting training metrics.")

    history_dict = history.history

    required_metrics = ['loss', 'mae', 'rmse'] # Base required

    if 'val_loss' in history_dict: # Add val metrics if they exist

        required_metrics.extend(['val_loss', 'val_mae', 'val_rmse'])



    if not all(metric in history_dict for metric in ['loss', 'mae', 'rmse']):

        logging.warning("Core training metrics missing in history. Cannot plot.")

        return



    loss = history_dict.get('loss', [])

    val_loss = history_dict.get('val_loss', []) # Returns None if key missing

    mae = history_dict.get('mae', [])

    val_mae = history_dict.get('val_mae', [])

    rmse = history_dict.get('rmse', [])

    val_rmse = history_dict.get('val_rmse', [])



    if not loss: # Check if history is empty

        logging.warning("No training history epochs found to plot.")

        return



    epochs = range(1, len(loss) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)



    axes[0].plot(epochs, loss, 'b-o', label='Training Loss (MSE)')

    if val_loss: axes[0].plot(epochs, val_loss, 'r-o', label='Validation Loss (MSE)')

    axes[0].set_title('Training and Validation Loss (MSE)'); axes[0].set_ylabel('Loss (MSE)'); axes[0].legend(); axes[0].grid(True)



    axes[1].plot(epochs, mae, 'b-o', label='Training MAE')

    if val_mae: axes[1].plot(epochs, val_mae, 'r-o', label='Validation MAE')

    axes[1].set_title('Training and Validation MAE'); axes[1].set_ylabel('MAE'); axes[1].legend(); axes[1].grid(True)



    axes[2].plot(epochs, rmse, 'b-o', label='Training RMSE')

    if val_rmse: axes[2].plot(epochs, val_rmse, 'r-o', label='Validation RMSE')

    axes[2].set_title('Training and Validation RMSE'); axes[2].set_xlabel('Epochs'); axes[2].set_ylabel('RMSE'); axes[2].legend(); axes[2].grid(True)



    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, 'training_metrics.png')

    try: plt.savefig(plot_path); logging.info(f"Training metrics plot saved to {plot_path}")

    except Exception as e: logging.error(f"Failed to save training metrics plot: {e}")

    plt.close()





def calculate_rmse_components(y_true_scaled, y_pred_scaled, target_scaler, target_names):

    """ Calculate and plot the RMSE for each TPN component on the original scale. """

    logging.info("Calculating RMSE per component.")

    if y_true_scaled.shape[0] == 0 or y_pred_scaled.shape[0] == 0: logging.warning("RMSE Calc: Input arrays empty."); return {}

    y_true_scaled = np.atleast_2d(y_true_scaled); y_pred_scaled = np.atleast_2d(y_pred_scaled)

    y_true_orig = target_scaler.inverse_transform(y_true_scaled); y_pred_orig = target_scaler.inverse_transform(y_pred_scaled)

    y_pred_orig = np.maximum(y_pred_orig, 0) # Ensure predictions are non-negative

    rmse_components = {}

    for i, col in enumerate(target_names):

        actual = y_true_orig[:, i]

        pred = y_pred_orig[:, i]

        rmse_components[col] = np.sqrt(np.mean((actual - pred)**2))



    sorted_rmse = sorted(rmse_components.items(), key=lambda x: x[1], reverse=True); sorted_cols = [i[0] for i in sorted_rmse]; sorted_values = [i[1] for i in sorted_rmse]

    plt.figure(figsize=(15, 10)); plt.bar(range(len(sorted_cols)), sorted_values); plt.xticks(range(len(sorted_cols)), sorted_cols, rotation=90)

    plt.title('RMSE for Each TPN Component (Original Scale, Sorted)'); plt.ylabel('RMSE'); plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, 'rmse_components_sorted.png')

    try: plt.savefig(plot_path); logging.info(f"Component RMSE plot saved to {plot_path}")

    except Exception as e: logging.error(f"Failed to save component RMSE plot: {e}")

    plt.close()

    return rmse_components



def calculate_correlation_metrics(y_true_scaled, y_pred_scaled, target_scaler, target_names):

    """

    Calculate Pearson R, R², and CCC for each TPN component on the original scale.

    MODIFIED: Calculates average metrics only for components with non-zero variance in actual values.

    """

    logging.info("Calculating correlation metrics (R, R2, CCC) per component.")

    default_metrics = {'global_r': 0, 'global_r2': 0, 'global_ccc': 0, 'avg_r': 0, 'avg_r2': 0, 'avg_ccc': 0}

    if y_true_scaled.shape[0] == 0 or y_pred_scaled.shape[0] == 0: logging.warning("Correlation Calc: Input arrays empty."); return {}, {}, {}, default_metrics



    y_true_scaled = np.atleast_2d(y_true_scaled); y_pred_scaled = np.atleast_2d(y_pred_scaled)

    y_true_orig = target_scaler.inverse_transform(y_true_scaled); y_pred_orig = target_scaler.inverse_transform(y_pred_scaled)

    y_pred_orig = np.maximum(y_pred_orig, 0) # Ensure predictions are non-negative



    r_values, r2_values, ccc_values, warnings_list = {}, {}, {}, []

    # --- MODIFICATION: Track components with valid variance in actuals ---

    components_with_variance = []

    # --- END MODIFICATION ---



    for i, col in enumerate(target_names):

        actual, pred = y_true_orig[:, i], y_pred_orig[:, i]

        actual_var = np.var(actual)

        pred_var = np.var(pred)



        # Check for zero variance in ACTUAL values - these won't contribute to meaningful correlation

        if actual_var < 1e-9: # Threshold to consider it zero variance

            r, r2, ccc = 0, 0, 0 # Assign 0 if actual is constant

            if pred_var >= 1e-9: # If actual is 0 but prediction varies, it's technically an error

                 warnings_list.append(f"Zero variance actual, non-zero pred: {col}")

            # Don't add to components_with_variance

        # Check for zero variance in PREDICTED values (less critical for avg calc, but good to know)

        elif pred_var < 1e-9:

            r, r2, ccc = 0, 0, 0 # Assign 0 if prediction is constant (but actual wasn't)

            warnings_list.append(f"Zero variance predicted: {col}")

            # --- MODIFICATION: Add to list if actual had variance ---

            components_with_variance.append(col)

            # --- END MODIFICATION ---

        # Both actual and predicted have variance, calculate metrics

        else:

            # --- MODIFICATION: Add to list if actual had variance (redundant here, but explicit) ---

            components_with_variance.append(col)

            # --- END MODIFICATION ---

            try:

                r, _ = pearsonr(actual, pred)

                r = 0 if np.isnan(r) else r # Handle potential NaN from pearsonr

            except ValueError as e: warnings_list.append(f"R error {col}: {e}"); r = 0

            try:

                r2 = r2_score(actual, pred)

            except ValueError as e: warnings_list.append(f"R2 error {col}: {e}"); r2 = 0

            try:

                mean_true, mean_pred = np.mean(actual), np.mean(pred)

                # Use pre-calculated variances

                covariance = np.cov(actual, pred)[0, 1]; numerator = 2 * covariance

                denominator = actual_var + pred_var + (mean_true - mean_pred)**2;

                ccc = numerator / denominator if denominator != 0 else 0

                ccc = 0 if np.isnan(ccc) else ccc

            except Exception as e: warnings_list.append(f"CCC error {col}: {e}"); ccc = 0



        r_values[col], r2_values[col], ccc_values[col] = r, r2, ccc



    if warnings_list: logging.warning(f"Correlation calculation warnings: {warnings_list}")



    # --- MODIFICATION: Calculate average metrics using only components_with_variance ---

    logging.info(f"Calculating average metrics based on {len(components_with_variance)} components with non-zero actual variance.")

    filtered_r_vals = [r_values[col] for col in components_with_variance if col in r_values]

    filtered_r2_vals = [r2_values[col] for col in components_with_variance if col in r2_values]

    filtered_ccc_vals = [ccc_values[col] for col in components_with_variance if col in ccc_values]



    avg_r = np.nanmean(filtered_r_vals) if filtered_r_vals else 0; avg_r = 0 if np.isnan(avg_r) else avg_r

    avg_r2 = np.nanmean(filtered_r2_vals) if filtered_r2_vals else 0; avg_r2 = 0 if np.isnan(avg_r2) else avg_r2

    avg_ccc = np.nanmean(filtered_ccc_vals) if filtered_ccc_vals else 0; avg_ccc = 0 if np.isnan(avg_ccc) else avg_ccc

    # --- END MODIFICATION ---



    metrics_to_plot = {'R': (r_values, avg_r), 'R2': (r2_values, avg_r2), 'CCC': (ccc_values, avg_ccc)}

    titles = {'R': 'Pearson Correlation Coefficient (R)', 'R2': 'Coefficient of Determination (R²)', 'CCC': 'Concordance Correlation Coefficient (CCC)'}

    for name, (values_dict, avg_val) in metrics_to_plot.items():

        if not values_dict: continue

        sorted_items = sorted(values_dict.items(), key=lambda x: x[1], reverse=True); sorted_cols = [i[0] for i in sorted_items]; sorted_values = [i[1] for i in sorted_items]

        plt.figure(figsize=(15, 10)); plt.bar(range(len(sorted_cols)), sorted_values); plt.xticks(range(len(sorted_cols)), sorted_cols, rotation=90)

        plt.title(f'{titles[name]} for Each TPN Component (Original Scale, Sorted)'); plt.ylabel(name); plt.axhline(y=0, color='r', linestyle='-');

        # --- MODIFICATION: Add note about average calculation method to plot legend ---

        plt.axhline(y=avg_val, color='g', linestyle='--', label=f'Average {name} (Valid Components): {avg_val:.4f}');

        # --- END MODIFICATION ---

        plt.legend(); plt.tight_layout()

        plot_path = os.path.join(OUTPUT_DIR, f'{name.lower()}_values_sorted.png')

        try: plt.savefig(plot_path); logging.info(f"Component {name} plot saved to {plot_path}")

        except Exception as e: logging.error(f"Failed to save component {name} plot: {e}")

        plt.close()



    logging.info("Calculating global correlation metrics.")

    flattened_true, flattened_pred = y_true_orig.flatten(), y_pred_orig.flatten(); global_r, global_r2, global_ccc = 0, 0, 0

    if flattened_true.size > 0 and flattened_pred.size > 0 and np.var(flattened_true) > 1e-9 and np.var(flattened_pred) > 1e-9:

        try:

            global_r, _ = pearsonr(flattened_true, flattened_pred); global_r = 0 if np.isnan(global_r) else global_r

            global_r2 = r2_score(flattened_true, flattened_pred)

            mean_true_global, mean_pred_global = np.mean(flattened_true), np.mean(flattened_pred); var_true_global, var_pred_global = np.var(flattened_true), np.var(flattened_pred)

            covariance_global = np.cov(flattened_true, flattened_pred)[0, 1]; numerator_global = 2 * covariance_global

            denominator_global = var_true_global + var_pred_global + (mean_true_global - mean_pred_global)**2; global_ccc = numerator_global / denominator_global if denominator_global != 0 else 0

            global_ccc = 0 if np.isnan(global_ccc) else global_ccc

        except Exception as e: logging.error(f"Error calculating global metrics: {e}. Setting to 0.")

    else: logging.warning("Zero variance or empty arrays in flattened values. Global metrics set to 0.")



    # Return the calculated average values (based on valid components)

    global_metrics = {'global_r': global_r, 'global_r2': global_r2, 'global_ccc': global_ccc, 'avg_r': avg_r, 'avg_r2': avg_r2, 'avg_ccc': avg_ccc}

    return r_values, r2_values, ccc_values, global_metrics



def plot_pred_vs_actual(y_true_scaled, y_pred_scaled, target_scaler, target_names):

    """ Create scatter plots of predicted vs actual values for each TPN component (original scale). """

    logging.info("Creating predicted vs. actual scatter plots.")

    if y_true_scaled.shape[0] == 0 or y_pred_scaled.shape[0] == 0: logging.warning("Pred vs Actual plot: Input arrays empty."); return

    y_true_scaled = np.atleast_2d(y_true_scaled); y_pred_scaled = np.atleast_2d(y_pred_scaled)

    y_true_orig = target_scaler.inverse_transform(y_true_scaled); y_pred_orig = target_scaler.inverse_transform(y_pred_scaled)

    y_pred_orig = np.maximum(y_pred_orig, 0)

    pred_vs_actual_dir = os.path.join(OUTPUT_DIR, 'pred_vs_actual_plots'); os.makedirs(pred_vs_actual_dir, exist_ok=True)

    n_targets = len(target_names); n_cols = 5; n_rows = int(np.ceil(n_targets / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(20, n_cols * 4), n_rows * 4)); axes = axes.flatten()

    last_plot_index = -1

    for i, col in enumerate(target_names):

        ax = axes[i]; actual = y_true_orig[:, i]; pred = y_pred_orig[:, i]

        ax.scatter(actual, pred, alpha=0.3, edgecolors='k', s=20)

        min_val = min(np.min(actual) if actual.size>0 else 0, np.min(pred) if pred.size>0 else 0)

        max_val = max(np.max(actual) if actual.size>0 else 0, np.max(pred) if pred.size>0 else 0)

        # Ensure min_val and max_val are finite before plotting line

        if np.isfinite(min_val) and np.isfinite(max_val) and max_val > min_val:

            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

        ax.set_xlabel('Actual Value'); ax.set_ylabel('Predicted Value'); ax.set_title(col, fontsize=10); ax.grid(True, linestyle=':', alpha=0.6)

        last_plot_index = i

    for j in range(last_plot_index + 1, len(axes)): fig.delaxes(axes[j]) # Use last_plot_index

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]); fig.suptitle('Predicted vs Actual Values (Original Scale)', fontsize=16)

    summary_plot_path = os.path.join(OUTPUT_DIR, 'pred_vs_actual_summary.png')

    try: plt.savefig(summary_plot_path); logging.info(f"Summary predicted vs actual plot saved to {summary_plot_path}")

    except Exception as e: logging.error(f"Failed to save summary predicted vs actual plot: {e}")

    plt.close()





def extract_patient_data(X_test_seq, y_test_seq_scaled, test_patient_ids_in_seq,

                         target_scaler, target_names, model, selected_patient_ids):

    """

    Extract data for specific patients, make predictions, compare, and plot.

    Corrected Relative Error calculation.

    """

    logging.info(f"Extracting and analyzing data for selected patients: {selected_patient_ids}")

    if X_test_seq.shape[0] == 0: logging.warning("Cannot extract patient data: X_test_seq is empty."); return

    if X_test_seq.ndim != 3: raise ValueError(f"X_test_seq must be 3D, got {X_test_seq.shape}")

    if y_test_seq_scaled.ndim != 2: raise ValueError(f"y_test_seq_scaled must be 2D, got {y_test_seq_scaled.shape}")



    for patient_id in selected_patient_ids:

        patient_indices = np.where(test_patient_ids_in_seq == patient_id)[0]

        if len(patient_indices) == 0: logging.warning(f"Patient {patient_id} not found in the test sequence set."); continue

        patient_idx = patient_indices[0] # Analyze the first sequence found for the patient in the test set

        logging.info(f"Analyzing sequence index {patient_idx} for patient {patient_id}")

        patient_features_seq = X_test_seq[patient_idx:patient_idx+1]; patient_actual_scaled = y_test_seq_scaled[patient_idx]

        patient_pred_scaled = model.predict(patient_features_seq)[0]

        actual_values = target_scaler.inverse_transform(patient_actual_scaled.reshape(1, -1))[0]

        predicted_values = target_scaler.inverse_transform(patient_pred_scaled.reshape(1, -1))[0]; predicted_values = np.maximum(predicted_values, 0)



        # --- Relative Error Calculation ---

        with np.errstate(divide='ignore', invalid='ignore'):

            relative_error = np.where(

                np.abs(actual_values) > 1e-6,  # Condition: Actual value is not near zero

                np.abs(actual_values - predicted_values) / np.abs(actual_values) * 100, # Value if True

                # Value if False: check predicted

                np.where(np.abs(predicted_values) > 1e-6, np.inf, 0.0) # Assign inf if pred isn't 0, else 0

            )

            # Replace potential NaNs resulting from 0/0 with 0

            relative_error = np.nan_to_num(relative_error, nan=0.0, posinf=np.inf, neginf=-np.inf)





        results = pd.DataFrame({

            'TPN Component': target_names,

            'Actual Value': actual_values,

            'Predicted Value': predicted_values,

            'Absolute Error': np.abs(actual_values - predicted_values),

            'Relative Error (%)': relative_error # Use the calculated array

        })



        csv_path = os.path.join(OUTPUT_DIR, f'patient_{patient_id}_prediction_comparison.csv')

        try: results.to_csv(csv_path, index=False); logging.info(f"Comparison CSV saved for patient {patient_id} to {csv_path}")

        except Exception as e: logging.error(f"Failed to save comparison CSV for patient {patient_id}: {e}")



        # --- Plotting logic ---

        plt.figure(figsize=(15, 10)); x_labels = target_names; x_pos = np.arange(len(x_labels))

        plt.plot(x_pos, actual_values, 'b-o', linewidth=2, markersize=6, label='Actual'); plt.plot(x_pos, predicted_values, 'r--^', linewidth=2, markersize=6, label='Predicted')

        # Add lines connecting actual and predicted points for visual comparison

        for i in range(len(x_labels)): plt.vlines(i, actual_values[i], predicted_values[i], colors='grey', linestyles='dotted', alpha=0.7)

        plt.xlabel('TPN Components'); plt.ylabel('Values (Original Scale)'); plt.title(f'Patient {patient_id} - Actual vs Predicted TPN Components'); plt.xticks(x_pos, x_labels, rotation=90); plt.grid(True, linestyle='--', alpha=0.7, axis='y'); plt.legend(); plt.tight_layout()

        plot_path = os.path.join(OUTPUT_DIR, f'patient_{patient_id}_comparison_line.png')

        try: plt.savefig(plot_path); logging.info(f"Comparison plot saved for patient {patient_id} to {plot_path}")

        except Exception as e: logging.error(f"Failed to save comparison plot for patient {patient_id}: {e}")

        plt.close(); logging.info(f"Patient {patient_id} analysis complete.")





# --- Main Execution Logic ---



def main():

    data_path = '/remote/home/ariss01/TPN/cohort_tpn_data_original_cleaned__decomposed_pivoted_with_Volume_lipids_corrected_measurement_v3_motor_embeddings_FIXED.csv'

    # --- Tune these parameters ---

    sequence_length = 100

    max_gap_hours = 48 # Adjusted gap based on previous testing? Tune as needed.

    test_split_ratio = 0.20

    val_split_ratio = 0.10

    # ---------------------------



    scaled_features, scaled_targets, person_ids, order_insts, scalers, \

    actual_feature_cols, actual_target_cols = load_and_preprocess_data(data_path)



    X_seq, y_seq, seq_lengths, seq_patient_ids = create_sequences(

        scaled_features, scaled_targets, person_ids, order_insts,

        feature_names=actual_feature_cols, target_names=actual_target_cols,

        sequence_length=sequence_length, max_gap_hours=max_gap_hours

    )



    logging.info(f"Sequence array shape: X={X_seq.shape}, y={y_seq.shape}")

    logging.info(f"Sequence lengths stats: Mean={np.mean(seq_lengths):.2f}, Median={np.median(seq_lengths):.2f}, "

                 f"Min={np.min(seq_lengths)}, Max={np.max(seq_lengths)}")



    logging.info("Splitting data by patient ID...")

    unique_patient_ids = np.unique(seq_patient_ids)

    n_patients = len(unique_patient_ids)

    logging.info(f"Total unique patients with sequences: {n_patients}")

    if n_patients < 3: raise ValueError(f"Not enough unique patients ({n_patients}) with valid sequences for train/val/test split.")



    train_val_ids, test_ids = train_test_split(unique_patient_ids, test_size=test_split_ratio, random_state=42)

    if len(train_val_ids) == 0: raise ValueError("Test split resulted in zero patients for training/validation.")



    # Adjust validation ratio based on the remaining data after test split

    val_ratio_adjusted = val_split_ratio / (1 - test_split_ratio)

    if val_ratio_adjusted >= 1.0 or len(train_val_ids) < 2: # Cannot create a validation set

        train_ids = train_val_ids; val_ids = np.array([]); logging.warning("Could not create a distinct validation set. Using all remaining data for training.")

    else:

        train_ids, val_ids = train_test_split(train_val_ids, test_size=val_ratio_adjusted, random_state=42)



    logging.info(f"Patient Split: Train={len(train_ids)}, Validation={len(val_ids)}, Test={len(test_ids)}")

    # Create boolean masks based on patient IDs

    train_mask = np.isin(seq_patient_ids, train_ids); val_mask = np.isin(seq_patient_ids, val_ids); test_mask = np.isin(seq_patient_ids, test_ids)

    # Apply masks to get the data splits

    X_train_seq, y_train_seq = X_seq[train_mask], y_seq[train_mask]

    X_val_seq, y_val_seq = X_seq[val_mask] if len(val_ids) > 0 else None, y_seq[val_mask] if len(val_ids) > 0 else None # Handle empty val set explicitly

    X_test_seq, y_test_seq = X_seq[test_mask], y_seq[test_mask]

    test_patient_ids_in_seq = seq_patient_ids[test_mask] # Keep track for analysis



    if X_train_seq.shape[0] == 0: raise ValueError("Training set is empty after splitting by patient ID.")

    # Prepare validation data argument for model.fit

    validation_data_arg = (X_val_seq, y_val_seq) if (X_val_seq is not None and X_val_seq.shape[0] > 0) else None

    if validation_data_arg is None: logging.warning("Validation set is empty. Training will proceed without validation metrics.")

    if X_test_seq.shape[0] == 0: logging.warning("Test set is empty after splitting by patient ID. Evaluation will be skipped.")



    logging.info(f"Sequence Split Shapes: Train X={X_train_seq.shape}, Val X={X_val_seq.shape if validation_data_arg else 'N/A'}, Test X={X_test_seq.shape}")

    logging.info(f"Sequence Split Shapes: Train y={y_train_seq.shape}, Val y={y_val_seq.shape if validation_data_arg else 'N/A'}, Test y={y_test_seq.shape}")



    n_combined_features = X_train_seq.shape[2]; n_targets = y_train_seq.shape[1]



    # --- Train Model ---

    model, history = train_model(

        X_train_seq, y_train_seq, X_val_seq, y_val_seq, # Pass val data (even if None)

        sequence_length, n_combined_features, n_targets

    )



    plot_training_metrics(history)



    # --- Evaluate (Only if test set is not empty) ---

   

    if X_test_seq.shape[0] > 0:

        logging.info("Evaluating model on test data...")

        # Log the potentially inconsistent Keras-reported names, but DON'T use them for mapping

        logging.info(f"Model metric names reported by Keras (INFO ONLY): {model.metrics_names}")



        # Run evaluation - returns a list of scalar metric values

        test_evaluation = model.evaluate(X_test_seq, y_test_seq, verbose=1)



        # Define the expected metric names IN ORDER based on model.compile()

        # Your compile call is: loss='mean_squared_error', metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]

        # So the order will be: loss, mae, rmse

        expected_metric_names = ['loss', 'mae', 'rmse']



        # Ensure test_evaluation is iterable (list or tuple) in case evaluate returns a single scalar

        if not isinstance(test_evaluation, (list, tuple)):

            test_evaluation = [test_evaluation] # Wrap single value in a list



        # Check if the NUMBER of results matches expectations

        if len(test_evaluation) == len(expected_metric_names):

            # Create the dictionary using the CORRECT expected names

            test_metrics_dict = dict(zip(expected_metric_names, test_evaluation))

            logging.info(f"Test Evaluation Results Raw (Mapped Correctly): {test_metrics_dict}")

        else:

            # Log an error if the number of results is wrong - this indicates a deeper problem

            logging.error(f"CRITICAL: Mismatch between expected metrics ({len(expected_metric_names)}) "

                          f"and actual evaluation results ({len(test_evaluation)}). Results: {test_evaluation}")

            test_metrics_dict = {} # Keep it empty to prevent downstream errors trying to access keys







        # --- Safely Print Evaluation Metrics ---

        print(f"\n--- Test Set Evaluation ---")

        loss_value = test_metrics_dict.get('loss')

        if isinstance(loss_value, (int, float)): print(f"Test Loss (MSE): {loss_value:.4f}")

        else: print(f"Test Loss (MSE): {loss_value if loss_value is not None else 'N/A'}")



        mae_value = test_metrics_dict.get('mae')

        if isinstance(mae_value, (int, float)): print(f"Test MAE: {mae_value:.4f}")

        else: print(f"Test MAE: {'N/A' if mae_value is None else mae_value}")



        rmse_value = test_metrics_dict.get('rmse')

        if isinstance(rmse_value, (int, float)): print(f"Test RMSE: {rmse_value:.4f}")

        else: print(f"Test RMSE: {'N/A' if rmse_value is None else rmse_value}")



        print("\n--- All Reported Test Metrics ---")

        for key, value in test_metrics_dict.items():

            # Try to convert tensor to numpy if possible for cleaner printing

            if hasattr(value, 'numpy'): value_printable = value.numpy()

            else: value_printable = value

            # Format floats nicely

            if isinstance(value_printable, (int, float)): print(f"  {key}: {value_printable:.4f}")

            else: print(f"  {key}: {value_printable}")

        print(f"-----------------------------------\n")

        # --- End Safe Printing ---





        logging.info("Making predictions on test data...")

        y_pred_scaled = model.predict(X_test_seq)

        target_scaler = scalers['target_scaler']



        logging.info("Calculating and plotting RMSE per component...")

        rmse_components = calculate_rmse_components(y_test_seq, y_pred_scaled, target_scaler, actual_target_cols)

        if rmse_components: # Check if dict is not empty

            sorted_rmse = sorted(rmse_components.items(), key=lambda x: x[1], reverse=True)

            print("\n--- Top 5 Components by RMSE (Higher is Worse) ---")

            for col, rmse in sorted_rmse[:5]: print(f"{col}: RMSE = {rmse:.4f}")

            print("\n--- Bottom 5 Components by RMSE (Lower is Better) ---")

            for col, rmse in sorted_rmse[-5:]: print(f"{col}: RMSE = {rmse:.4f}")

            print(f"----------------------------------------------------\n")



        logging.info("Calculating and plotting correlation metrics (R, R², CCC)...")

        r_values, r2_values, ccc_values, global_metrics = calculate_correlation_metrics(

            y_test_seq, y_pred_scaled, target_scaler, actual_target_cols

        )

        if r_values: # Check if dict is not empty

            sorted_r_components = sorted(r_values.items(), key=lambda x: x[1], reverse=True)

            print("\n--- Top 5 Components by Pearson R (Closer to 1 is Better) ---")

            for col, r in sorted_r_components[:5]: print(f"{col}: R = {r:.4f}, R² = {r2_values.get(col, float('nan')):.4f}, CCC = {ccc_values.get(col, float('nan')):.4f}")

            print("\n--- Bottom 5 Components by Pearson R ---")

            for col, r in sorted_r_components[-5:]: print(f"{col}: R = {r:.4f}, R² = {r2_values.get(col, float('nan')):.4f}, CCC = {ccc_values.get(col, float('nan')):.4f}")

            print(f"-----------------------------------------------------------\n")



        print("\n--- Overall Correlation Metrics ---")

        print(f"Global Pearson's R (Flattened): {global_metrics['global_r']:.4f}")

        print(f"Global R² (Flattened): {global_metrics['global_r2']:.4f}")

        print(f"Global CCC (Flattened): {global_metrics['global_ccc']:.4f}")

        print("---")

        # --- MODIFICATION: Add clarification for average metrics ---

        print(f"Average Pearson's R (Mean of Components with Actual Variance): {global_metrics['avg_r']:.4f}")

        print(f"Average R² (Mean of Components with Actual Variance): {global_metrics['avg_r2']:.4f}")

        print(f"Average CCC (Mean of Components with Actual Variance): {global_metrics['avg_ccc']:.4f}")

        # --- END MODIFICATION ---

        print(f"-----------------------------------\n")



        logging.info("Saving evaluation metrics to CSV files.")

        avg_metrics_df = pd.DataFrame({'Metric': ['Pearson R', 'R²', 'CCC'], 'Global': [global_metrics['global_r'], global_metrics['global_r2'], global_metrics['global_ccc']], 'Average (Valid Components)': [global_metrics['avg_r'], global_metrics['avg_r2'], global_metrics['avg_ccc']]}) # Updated column name

        try: avg_metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'correlation_metrics_summary.csv'), index=False)

        except Exception as e: logging.error(f"Failed to save summary metrics CSV: {e}")



        if rmse_components and r_values and r2_values and ccc_values: # Ensure dicts are populated

            components_df = pd.DataFrame({

                'Component': actual_target_cols,

                'RMSE': [rmse_components.get(col, float('nan')) for col in actual_target_cols],

                'R': [r_values.get(col, float('nan')) for col in actual_target_cols],

                'R²': [r2_values.get(col, float('nan')) for col in actual_target_cols],

                'CCC': [ccc_values.get(col, float('nan')) for col in actual_target_cols]

            })

            components_df = components_df.sort_values('R', ascending=False) # Sort by R for reporting

            try: components_df.to_csv(os.path.join(OUTPUT_DIR, 'component_metrics.csv'), index=False)

            except Exception as e: logging.error(f"Failed to save component metrics CSV: {e}")



        plot_pred_vs_actual(y_test_seq, y_pred_scaled, target_scaler, actual_target_cols)



        logging.info("Selecting random patients from test set for detailed analysis.")

        unique_test_patient_ids = np.unique(test_patient_ids_in_seq)

        if len(unique_test_patient_ids) >= 1:

            # Select up to 2 random patients for detailed plotting/CSV generation

            selected_patients = np.random.choice(unique_test_patient_ids, size=min(2, len(unique_test_patient_ids)), replace=False)

            logging.info(f"Selected patients for detailed analysis: {selected_patients}")

            extract_patient_data(X_test_seq, y_test_seq, test_patient_ids_in_seq, target_scaler, actual_target_cols, model, selected_patients)

        else:

            logging.warning("No unique patients found in the test set sequences to analyze.")

    else:

        logging.warning("Test set was empty. Skipping test set evaluation and detailed analysis.")



    # Use the correct path based on training outcome

    final_model_path = os.path.join(OUTPUT_DIR, 'best_lstm_model_3layer_adamw.keras')

    logging.info(f"Model saved (best weights during training based on val_loss, if available) as '{os.path.basename(final_model_path)}' in {OUTPUT_DIR}")

    logging.info("\n--- Script finished ---")

    logging.info(f"See generated plots, logs, and analysis files in: {OUTPUT_DIR}")





if __name__ == "__main__":

    # Basic check for feature_cols placeholder

    if not feature_cols or feature_cols[0] == "placeholder_feature_1":

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        print("!!! ERROR: `feature_cols` list is missing or invalid. !!!")

        print("!!! Please paste your complete list of feature names.  !!!")

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    else:

        main()