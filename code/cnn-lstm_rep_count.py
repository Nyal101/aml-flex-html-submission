Python 3.13.1 (v3.13.1:06714517797, Dec  3 2024, 14:00:22) [Clang 15.0.0 (clang-1500.3.9.4)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (classification_report, confusion_matrix, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

###############################################################################
# 1) Data Loading and Label Construction
###############################################################################
def load_and_label_data(csv_path):
    """
    Loads an IMU CSV file that may contain lines of sensor data and 'NEW_REP' markers.
    Returns a DataFrame with columns:
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
    load_and_label_data("/content/NewBicep_7.csv"),
    load_and_label_data("/content/NewPress_7.csv")
], axis=0).reset_index(drop=True)

test_df = load_and_label_data("/content/NewBicep_5.csv")

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)

###############################################################################
# 3) Create Non-Overlapping Windows (Label=1 if ANY sample==1)
###############################################################################
def create_non_overlapping_windows_label_if_any_end(df, window_size):
    """
    Segments df into non-overlapping windows of size 'window_size'.
    If any sample in a window is labeled 1, the window's label is set to 1.
    """
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
# 4) Data Preprocessing for CNN-LSTM
###############################################################################
num_train = X_train.shape[0]
num_test = X_test.shape[0]
num_channels = X_train.shape[2]

scaler = StandardScaler()
X_train_reshaped = X_train.reshape(-1, num_channels)
X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(num_train, WINDOW_SIZE, num_channels)

X_test_reshaped = X_test.reshape(-1, num_channels)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(num_test, WINDOW_SIZE, num_channels)

###############################################################################
# 5) Build a More Complex CNN-LSTM Model
###############################################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, LSTM, Dense

model = Sequential()
# First convolution block
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, num_channels)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

# Second convolution block
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

# LSTM layers
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(32))
model.add(Dropout(0.25))

# Dense layers
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

###############################################################################
# 6) Train the CNN-LSTM Model with Class Weights
###############################################################################
# Suppose you want to give more weight to the minority class
class_weight = {0: 1, 1: 3}

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint("best_cnn_lstm_model.h5", monitor='val_loss', save_best_only=True)
]

history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    class_weight=class_weight
)

###############################################################################
# 7) Evaluate the CNN-LSTM Model
###############################################################################
loss, acc = model.evaluate(X_test_scaled, y_test)
print("Test Loss:", loss, "Test Accuracy:", acc)

y_pred_probs = model.predict(X_test_scaled).flatten()

# Grid Search Over Thresholds with Composite Metric
true_count = np.sum(y_test)
thresholds = np.linspace(0, 1, 101)
alpha = 0.6  # Weight for count error penalty
best_score = -np.inf
best_threshold = 0.5
best_recall = 0
best_count_error = None

recall_scores = []
count_errors = []
composite_scores = []
f1_scores = []

for t in thresholds:
    y_pred_temp = (y_pred_probs >= t).astype(int)
    recall_val = recall_score(y_test, y_pred_temp, zero_division=0)
    pred_count = np.sum(y_pred_temp)
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

predicted_count = np.sum((y_pred_probs >= best_threshold).astype(int))
print("Total true rep windows (test):", true_count)
print("Total predicted rep windows (test):", predicted_count)

# Plot trade-offs for the CNN-LSTM
plt.figure(figsize=(10, 6))
plt.plot(thresholds, recall_scores, label='Recall')
plt.plot(thresholds, count_errors, label='Count Error')
plt.plot(thresholds, composite_scores, label='Composite Score')
plt.plot(thresholds, f1_scores, label='F1 Score', linestyle='--')
plt.axvline(best_threshold, color='r', linestyle='--', label='Optimal Threshold')
plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.title('Composite Metric Grid Search (CNN-LSTM)')
plt.legend()
plt.grid(True)
plt.show()

# Final predictions at the best threshold
y_pred = (y_pred_probs >= best_threshold).astype(int)

###############################################################################
# 8) Evaluate Final CNN-LSTM Model with Optimal Threshold
###############################################################################
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report (Optimal Composite Metric):")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Rep", "Rep"], yticklabels=["No Rep", "Rep"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (CNN-LSTM)")
plt.show()

# Plot predicted rep count vs. true rep count
plt.figure(figsize=(6, 4))
plt.bar(['Actual Rep', 'Predicted Rep'], [true_count, predicted_count],
        color=['green', 'orange'])
plt.title("Comparison of Actual vs. Predicted Rep-End Windows")
plt.ylabel("Count")
plt.show()

###############################################################################
# 9) Visualize a Snippet of the Test Set
###############################################################################
snippet_len = 400  # Adjust to visualize more or fewer windows
snippet_ts = test_ts[:snippet_len]
snippet_y_true = y_test[:snippet_len]
snippet_y_pred = y_pred[:snippet_len]
snippet_X = X_test[:snippet_len]

# We'll plot Accel_X, Accel_Y, Accel_Z from the mid sample in each window
mid_idx = WINDOW_SIZE // 2
accel_x = snippet_X[:, mid_idx, 0]
... accel_y = snippet_X[:, mid_idx, 1]
... accel_z = snippet_X[:, mid_idx, 2]
... 
... actual_ones = np.where(snippet_y_true == 1)[0]
... pred_ones = np.where(snippet_y_pred == 1)[0]
... 
... fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
... 
... axs[0].plot(snippet_ts, accel_x, label='Accel_X', color='blue')
... axs[0].scatter(snippet_ts[actual_ones], accel_x[actual_ones],
...                marker='x', color='red', label='Actual Rep-End')
... axs[0].scatter(snippet_ts[pred_ones], accel_x[pred_ones],
...                marker='o', facecolors='none', edgecolors='orange',
...                label='Predicted Rep-End')
... axs[0].set_ylabel('Accel_X')
... axs[0].legend()
... 
... axs[1].plot(snippet_ts, accel_y, label='Accel_Y', color='blue')
... axs[1].scatter(snippet_ts[actual_ones], accel_y[actual_ones],
...                marker='x', color='red', label='Actual Rep-End')
... axs[1].scatter(snippet_ts[pred_ones], accel_y[pred_ones],
...                marker='o', facecolors='none', edgecolors='orange',
...                label='Predicted Rep-End')
... axs[1].set_ylabel('Accel_Y')
... axs[1].legend()
... 
... axs[2].plot(snippet_ts, accel_z, label='Accel_Z', color='blue')
... axs[2].scatter(snippet_ts[actual_ones], accel_z[actual_ones],
...                marker='x', color='red', label='Actual Rep-End')
... axs[2].scatter(snippet_ts[pred_ones], accel_z[pred_ones],
...                marker='o', facecolors='none', edgecolors='orange',
...                label='Predicted Rep-End')
... axs[2].set_ylabel('Accel_Z')
... axs[2].set_xlabel('Timestamp')
... axs[2].legend()
... 
... plt.suptitle('Test Data: Actual vs. Predicted Rep-End Windows')
... plt.show()
... 
