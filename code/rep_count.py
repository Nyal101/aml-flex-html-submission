import numpy as np
import pandas as pd
import tsfel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, recall_score)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib

###############################################################################
# 1) Data Loading and Label Construction
###############################################################################
def load_and_label_data(csv_path):
    """
    Loads an IMU CSV file that may contain lines of real sensor data
    and lines of 'NEW_REP'. Returns a DataFrame with columns:
    [Timestamp, Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z, Label],
    where Label=1 is assigned to the row immediately before 'NEW_REP'.
    """
    rows = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    last_valid_idx = None
    for line in lines:
        line = line.strip()
        if line.startswith("NEW_REP"):
            if last_valid_idx is not None and last_valid_idx < len(rows):
                rows[last_valid_idx][-1] = 1
        else:
            parts = line.split(",")
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
            except:
                continue
            rows.append([timestamp, ax, ay, az, gx, gy, gz, 0])
            last_valid_idx = len(rows) - 1
    columns = ["Timestamp", "Accel_X", "Accel_Y", "Accel_Z",
               "Gyro_X", "Gyro_Y", "Gyro_Z", "Label"]
    return pd.DataFrame(rows, columns=columns)

###############################################################################
# 2) Load Training and Test Data
###############################################################################
train_df = pd.concat([
    load_and_label_data("/content/NewBicep_1.csv"),
    load_and_label_data("/content/NewShoulder_1.csv"),
    load_and_label_data("/content/NewPress_1.csv"),
    load_and_label_data("/content/NewBicep_2.csv"),
    load_and_label_data("/content/NewShoulder_2.csv"),
    load_and_label_data("/content/NewPress_2.csv"),
    load_and_label_data("/content/NewBicep_3.csv"),
    load_and_label_data("/content/NewShoulder_3.csv"),
    load_and_label_data("/content/NewPress_3.csv"),
    load_and_label_data("/content/NewBicep_4.csv"),
    load_and_label_data("/content/NewShoulder_4.csv"),
    load_and_label_data("/content/NewPress_4.csv"),
    load_and_label_data("/content/NewShoulder_5.csv"),
    load_and_label_data("/content/NewPress_5.csv"),
    load_and_label_data("/content/NewBicep_6.csv"),
    load_and_label_data("/content/NewShoulder_6.csv"),
    load_and_label_data("/content/NewPress_6.csv"),
    load_and_label_data("/content/NewBicep_7.csv"),
    load_and_label_data("/content/NewShoulder_7.csv"),
    load_and_label_data("/content/NewPress_7.csv")
], axis=0).reset_index(drop=True)

test_df = load_and_label_data("/content/NewBicep_5.csv") # this file was exectued by Andrea, whose data are not used for training but just for testing


###############################################################################
# 3) Create Non-Overlapping Windows (Label Window=1 if ANY sample == 1)
###############################################################################
def create_non_overlapping_windows_label_if_any_end(df, window_size):
    sensor_cols = ["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]
    data_array = df[sensor_cols].values
    labels = df["Label"].values
    timestamps = df["Timestamp"].values

    X_list, y_list, ts_list = [], [], []
    start_idx = 0
    while start_idx + window_size <= len(df):
        end_idx = start_idx + window_size
        window_data = data_array[start_idx:end_idx]
        window_label = 1 if np.any(labels[start_idx:end_idx] == 1) else 0
        X_list.append(window_data)
        y_list.append(window_label)
        ts_list.append(timestamps[end_idx - 1])
        start_idx = end_idx

    return np.array(X_list), np.array(y_list), np.array(ts_list)

WINDOW_SIZE = 30
X_train, y_train, train_ts = create_non_overlapping_windows_label_if_any_end(train_df, WINDOW_SIZE)
X_test, y_test, test_ts = create_non_overlapping_windows_label_if_any_end(test_df, WINDOW_SIZE)

print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

###############################################################################
# 4) Extract Temporal Features with TSFEL
###############################################################################
def extract_tsfel_temporal_features_from_windows(X, fs=50):
    """
    For each window in X (shape: [num_windows, window_size, num_channels]),
    extracts only temporal features using TSFEL.
    Returns a feature matrix of shape [num_windows, num_features].
    """
    cfg = tsfel.get_features_by_domain("temporal")
    feature_list = []
    for window in X:
        df_window = pd.DataFrame(window, columns=["Accel_X", "Accel_Y", "Accel_Z",
                                                  "Gyro_X", "Gyro_Y", "Gyro_Z"])
        features_df = tsfel.time_series_features_extractor(cfg, df_window, fs=fs, verbose=0)
        features = np.nan_to_num(features_df.values[0], nan=0.0)
        feature_list.append(features)
    return np.array(feature_list)

FS = 130
X_train_feat = extract_tsfel_temporal_features_from_windows(X_train, fs=FS)
X_test_feat = extract_tsfel_temporal_features_from_windows(X_test, fs=FS)

print("X_train_feat shape:", X_train_feat.shape)
print("X_test_feat shape:", X_test_feat.shape)

###############################################################################
# 5) Feature Scaling
###############################################################################
scaler = StandardScaler()
X_train_feat_scaled = scaler.fit_transform(X_train_feat)
X_test_feat_scaled = scaler.transform(X_test_feat)

###############################################################################
# 6) Train Final Classifier with Fixed Hyperparameters
###############################################################################
# Fixed hyperparameters 
FIXED_CLASS_WEIGHT = {0: 1, 1: 3}

final_clf = RandomForestClassifier(n_estimators=500, class_weight=FIXED_CLASS_WEIGHT, random_state=42)
final_clf.fit(X_train_feat_scaled, y_train)

# Get predicted probabilities on the test set
final_probs = final_clf.predict_proba(X_test_feat_scaled)[:, 1]

###############################################################################
# 7) Grid Search Over Thresholds with Composite Metric
###############################################################################
true_count = np.sum(y_test)

# Search thresholds from 0 to 1 in increments of 0.01
thresholds = np.linspace(0, 1, 101)
alpha = 0.7  # Weight for count error penalty

best_score = -np.inf
best_threshold = 0.5
best_recall = 0
best_count_error = None

recall_scores = []
count_errors = []
composite_scores = []
f1_scores = []

for t in thresholds:
    y_pred_temp = (final_probs >= t).astype(int)
    recall_val = recall_score(y_test, y_pred_temp, zero_division=0)
    pred_count = np.sum(y_pred_temp)

    # Normalized count error
    if true_count > 0:
        count_error = abs(pred_count - true_count) / true_count
    else:
        count_error = 0.0

    composite = recall_val - alpha * count_error

    recall_scores.append(recall_val)
    count_errors.append(count_error)
    composite_scores.append(composite)
    f1_scores.append(f1_score(y_test, y_pred_temp, zero_division=0))

    if composite > best_score:
        best_score = composite
        best_threshold = t
        best_recall = recall_val
        best_count_error = count_error

print(f"Best Composite Score: {best_score:.4f}")
print("Optimal Threshold (Composite Metric):", best_threshold)
print(f"Recall at Optimal Threshold: {best_recall:.4f}")
print(f"Count Error at Optimal Threshold: {best_count_error:.4f}")

predicted_count = np.sum((final_probs >= best_threshold).astype(int))
print("Total true rep windows (test):", true_count)
print("Total predicted rep windows (test):", predicted_count)

# Visualize metric trade-offs over the thresholds
plt.figure(figsize=(10, 6))
plt.plot(thresholds, recall_scores, label='Recall')
plt.plot(thresholds, count_errors, label='Count Error')
plt.plot(thresholds, composite_scores, label='Composite Score')
plt.plot(thresholds, f1_scores, label='F1 Score', linestyle='--')
plt.axvline(best_threshold, color='r', linestyle='--', label='Optimal Threshold')
plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.title('Composite Metric Grid Search (RandomForest)')
plt.legend()
plt.grid(True)
plt.show()

###############################################################################
# 8) Evaluate the Final Model with the Optimal Threshold
###############################################################################
y_pred_final_grid = (final_probs >= best_threshold).astype(int)

print("\nClassification Report (Optimal Composite Metric):")
print(classification_report(y_test, y_pred_final_grid))

cm = confusion_matrix(y_test, y_pred_final_grid)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Rep", "Rep"],
            yticklabels=["No Rep", "Rep"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (RandomForest with Composite Grid Search)")
plt.show()

###############################################################################
# 9) (Optional) Visualize a Snippet of the Test Set with Accel & Gyro Magnitudes
###############################################################################
snippet_len = 150  # Adjust to visualize more or fewer windows
snippet_ts = test_ts[:snippet_len]
snippet_y_true = y_test[:snippet_len]
snippet_y_pred = y_pred_final_grid[:snippet_len]
snippet_X = X_test[:snippet_len]

# Compute mid-sample values from each window (using the middle index)
mid_idx = WINDOW_SIZE // 2

# Accelerometer components (channels 0, 1, 2) and Gyroscope components (channels 3, 4, 5)
accel_x = snippet_X[:, mid_idx, 0]
accel_y = snippet_X[:, mid_idx, 1]
accel_z = snippet_X[:, mid_idx, 2]
gyro_x  = snippet_X[:, mid_idx, 3]
gyro_y  = snippet_X[:, mid_idx, 4]
gyro_z  = snippet_X[:, mid_idx, 5]

# Compute magnitudes
accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
gyro_magnitude  = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

# Identify indices where the actual and predicted labels are 1 (rep-end windows)
actual_ones = np.where(snippet_y_true == 1)[0]
pred_ones   = np.where(snippet_y_pred == 1)[0]

# Plot Accelerometer Magnitude and Gyroscope Magnitude in two subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axs[0].plot(snippet_ts, accel_magnitude, label='Accel Magnitude', color='blue')
axs[0].scatter(snippet_ts[actual_ones], accel_magnitude[actual_ones],
               marker='x', color='red', label='Actual Rep-End')
axs[0].scatter(snippet_ts[pred_ones], accel_magnitude[pred_ones],
               marker='o', facecolors='none', edgecolors='orange', label='Predicted Rep-End')
axs[0].set_ylabel('Accel Magnitude')
axs[0].legend()

axs[1].plot(snippet_ts, gyro_magnitude, label='Gyro Magnitude', color='blue')
axs[1].scatter(snippet_ts[actual_ones], gyro_magnitude[actual_ones],
               marker='x', color='red', label='Actual Rep-End')
axs[1].scatter(snippet_ts[pred_ones], gyro_magnitude[pred_ones],
               marker='o', facecolors='none', edgecolors='orange', label='Predicted Rep-End')
axs[1].set_ylabel('Gyro Magnitude')
axs[1].set_xlabel('Timestamp')
axs[1].legend()

plt.suptitle('Test Data: Actual vs. Predicted Rep-End Windows (Accel & Gyro Magnitudes)')
plt.show()
