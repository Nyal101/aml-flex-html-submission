import pandas as pd
import numpy as np
import tsfel

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# --------------------------------------------------------------------
# STEP 1: LOAD AND PARSE CSV FILES (same as before)
# --------------------------------------------------------------------
def load_file_and_mark_reps(csv_file, exercise_label):
    """
    Reads a CSV file that contains numeric IMU data rows plus lines with 'NEW_REP'.
    Returns:
      df_data: DataFrame of numeric rows with columns
               [Timestamp, Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z, 'exercise']
      rep_count: integer count of how many 'NEW_REP' lines were found
    """
    numeric_rows = []
    rep_count = 0

    with open(csv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check if this line is the 'NEW_REP' marker
            if "NEW_REP" in line:
                rep_count += 1
                continue

            parts = line.split(',')
            if len(parts) < 7:
                continue

            try:
                timestamp = float(parts[0])
                ax = float(parts[1])
                ay = float(parts[2])
                az = float(parts[3])
                gx = float(parts[4])
                gy = float(parts[5])
                gz = float(parts[6])
                numeric_rows.append([timestamp, ax, ay, az, gx, gy, gz])
            except ValueError:
                # Some lines might not parse nicely
                pass

    df_data = pd.DataFrame(
        numeric_rows,
        columns=["Timestamp","Accel_X","Accel_Y","Accel_Z","Gyro_X","Gyro_Y","Gyro_Z"]
    )
    df_data["exercise"] = exercise_label
    return df_data, rep_count

# --------------------------------------------------------------------
# Load each file
# --------------------------------------------------------------------
df_curl_1, reps_curl_1 = load_file_and_mark_reps("Bicep_Curl_1.csv", "Bicep_Curl")
df_curl_2, reps_curl_2 = load_file_and_mark_reps("Bicep_Curl_2.csv", "Bicep_Curl")
df_lat_1,  reps_lat_1  = load_file_and_mark_reps("Lat_raise_1.csv",  "Lat_Raise")
df_lat_2,  reps_lat_2  = load_file_and_mark_reps("Lat_raise_2.csv",  "Lat_Raise")

# Combine all into one DataFrame
df_all = pd.concat([df_curl_1, df_curl_2, df_lat_1, df_lat_2], ignore_index=True)
df_all.sort_values(by="Timestamp", inplace=True)
df_all.reset_index(drop=True, inplace=True)

print("Combined shape:", df_all.shape)
print("Bicep Curl Reps:", reps_curl_1 + reps_curl_2)
print("Lateral Raise Reps:", reps_lat_1 + reps_lat_2)
print("Total Reps:", reps_curl_1 + reps_curl_2 + reps_lat_1 + reps_lat_2)

# --------------------------------------------------------------------
# STEP 2: SEGMENT DATA INTO WINDOWS & EXTRACT FEATURES
# --------------------------------------------------------------------
WINDOW_SIZE = 100
STEP_SIZE   = 100  # no overlap for simplicity

segments = []
labels   = []
num_rows = len(df_all)
start = 0

while (start + WINDOW_SIZE) <= num_rows:
    window = df_all.iloc[start:start+WINDOW_SIZE]
    # Get the majority exercise label in this window
    exercise_mode = window["exercise"].value_counts().idxmax()
    segments.append(window)
    labels.append(exercise_mode)
    start += STEP_SIZE

print(f"Created {len(segments)} windows of size {WINDOW_SIZE}.")

# TSFEL config (all features)
cfg = tsfel.get_features_by_domain()

X_features = []
for seg in segments:
    # Extract TSFEL features per axis.
    df_x = seg[["Accel_X"]].reset_index(drop=True)
    df_y = seg[["Accel_Y"]].reset_index(drop=True)
    df_z = seg[["Accel_Z"]].reset_index(drop=True)

    feat_x = tsfel.time_series_features_extractor(cfg, df_x, window_size=len(df_x), overlap=0, fs=None, verbose=0)
    feat_y = tsfel.time_series_features_extractor(cfg, df_y, window_size=len(df_y), overlap=0, fs=None, verbose=0)
    feat_z = tsfel.time_series_features_extractor(cfg, df_z, window_size=len(df_z), overlap=0, fs=None, verbose=0)

    # Concatenate side-by-side
    feat_combined = pd.concat([feat_x, feat_y, feat_z], axis=1)
    # Remove duplicate columns if TSFEL reuses feature names for each axis
    feat_combined = feat_combined.loc[:, ~feat_combined.columns.duplicated()].copy()

    X_features.append(feat_combined.values[0])

X_features = np.array(X_features)
y_labels   = np.array(labels)

print("Feature matrix shape:", X_features.shape)
print("Labels shape:", y_labels.shape)

# Encode string labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)  # e.g. 0 or 1
num_classes = len(label_encoder.classes_)

# --------------------------------------------------------------------
# STEP 3: TRAIN/TEST SPLIT
# --------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_encoded,
    test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --------------------------------------------------------------------
# Utility to build different neural networks
# --------------------------------------------------------------------
def build_mlp(input_dim, num_classes):
    """Simple MLP model."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def build_cnn(input_dim, num_classes):
    """1D CNN model. We reshape (batch_size, input_dim) -> (batch_size, input_dim, 1)."""
    model = keras.Sequential([
        layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def build_lstm(input_dim, num_classes):
    """
    LSTM model. We'll treat each 'feature' as a 'time step'.
    So input shape = (batch_size, time_steps=input_dim, features=1).
    This is somewhat contrived for feature vectors,
    but demonstrates an LSTM pipeline.
    """
    model = keras.Sequential([
        layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

# --------------------------------------------------------------------
# STEP 4: Train each model & compare
# --------------------------------------------------------------------
batch_size = 16
epochs = 10  # increase if needed

models = {
    'MLP': build_mlp(X_train.shape[1], num_classes),
    'CNN': build_cnn(X_train.shape[1], num_classes),
    'LSTM': build_lstm(X_train.shape[1], num_classes)
}

results = {}

for model_name, model in models.items():
    print(f"\n=== Training {model_name} ===")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{model_name} accuracy on test set = {acc:.3f}")
    results[model_name] = (model, acc)

# --------------------------------------------------------------------
# Pick the best model
# --------------------------------------------------------------------
best_model_name = max(results, key=lambda m: results[m][1])  # model with highest acc
best_model, best_acc = results[best_model_name]
print(f"\nBest model is {best_model_name} with accuracy = {best_acc:.3f}")

# --------------------------------------------------------------------
# STEP 5: Simple Permutation Feature Importance for best model
# --------------------------------------------------------------------
def permutation_importance(model, X, y, baseline_acc, n_repeats=1):
    """
    For each feature column in X, shuffle it and measure drop in accuracy.
    This is a naive but straightforward measure of feature importance.
    """
    import copy
    import random

    importance_scores = np.zeros(X.shape[1])

    for col in range(X.shape[1]):
        acc_drops = []
        X_permuted = np.array(X, copy=True)

        for _ in range(n_repeats):
            # Shuffle only this column
            shuffled_col = np.random.permutation(X_permuted[:, col])
            X_permuted[:, col] = shuffled_col
            # Evaluate new accuracy
            loss, new_acc = model.evaluate(X_permuted, y, verbose=0)
            acc_drops.append(baseline_acc - new_acc)

        # Restore column in X_permuted for next repetition
        X_permuted[:, col] = X[:, col]
        importance_scores[col] = np.mean(acc_drops)

    return importance_scores

print("\nComputing permutation feature importance on the best model. This may take a while if you have many features...")
# First, get baseline accuracy
_, baseline_acc = best_model.evaluate(X_test, y_test, verbose=0)

feat_importances = permutation_importance(best_model, X_test, y_test, baseline_acc, n_repeats=1)

# Sort features by importance
sorted_idx = np.argsort(feat_importances)[::-1]  # descending

print("Top 10 most important features (indices & importance score):")
for i in range(10):
    idx = sorted_idx[i]
    print(f"Feature {idx} -> Importance Score = {feat_importances[idx]:.4f}")

# --------------------------------------------------------------------
# STEP 6: Rep Counting
# --------------------------------------------------------------------
# You already have "NEW_REP" lines in each file that count actual reps.
# We'll just display them again. There's no "model-based" rep counting
# included here, unless you define a separate approach for raw time-series
# peak detection or multi-output neural net.

print("\n====== REP COUNTS FROM FILES (marker-based) ======")
print(f"Bicep_Curl_1: {reps_curl_1} reps")
print(f"Bicep_Curl_2: {reps_curl_2} reps")
print(f"Lat_raise_1 : {reps_lat_1} reps")
print(f"Lat_raise_2 : {reps_lat_2} reps")
print("Total Reps:", reps_curl_1 + reps_curl_2 + reps_lat_1 + reps_lat_2)
