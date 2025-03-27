import pandas as pd
import numpy as np
import tsfel
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score
import os

# -------------------------------
# PART 1: LOAD DATA & EXTRACT FEATURES
# -------------------------------
def load_file_segment_reps(csv_file, exercise_label):
    data = []
    segments = []
    labels = []
    rep_start_idx = None
    with open(csv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "NEW_REP" in line:
                if rep_start_idx is not None and len(data) > rep_start_idx:
                    seg = data[rep_start_idx:len(data)]
                    segments.append(seg)
                    labels.append(exercise_label)
                rep_start_idx = len(data)
                continue
            parts = line.split(',')
            if len(parts) < 7:
                continue
            try:
                row = list(map(float, parts[:7]))
                data.append(row)
            except:
                continue
    if rep_start_idx is not None and len(data) > rep_start_idx:
        segments.append(data[rep_start_idx:])
        labels.append(exercise_label)
    return segments, labels

# 🔁 Define your new files and labels for 3 exercises: Bicep_Curl, Shoulder_Raise and Press.
all_segments = []
all_labels = []

# Add Bicep Curl files
for i in range(1, 8):
    fname = f"/content/NewBicep_{i}.csv"
    segs, labels = load_file_segment_reps(fname, "Bicep_Curl")
    all_segments.extend(segs)
    all_labels.extend(labels)

# Add Shoulder Raise files
for i in range(1, 8):
    fname = f"/content/NewShoulder_{i}.csv"
    segs, labels = load_file_segment_reps(fname, "Shoulder_Raise")
    all_segments.extend(segs)
    all_labels.extend(labels)

# Add Press files
for i in range(1, 8):
    fname = f"/content/NewPress_{i}.csv"
    segs, labels = load_file_segment_reps(fname, "Press")
    all_segments.extend(segs)
    all_labels.extend(labels)

print(f"📦 Loaded {len(all_segments)} segments.")

# Use only time-domain features from TSFEL
cfg = tsfel.get_features_by_domain("temporal")
X_features = []
valid_labels = []
expected_length = None

for seg, label in zip(all_segments, all_labels):
    df_seg = pd.DataFrame(seg, columns=["Timestamp", "Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"])
    # We extract features only from the acceleration channels here
    df_x = df_seg[["Accel_X"]].reset_index(drop=True)
    df_y = df_seg[["Accel_Y"]].reset_index(drop=True)
    df_z = df_seg[["Accel_Z"]].reset_index(drop=True)

    try:
        feat_x = tsfel.time_series_features_extractor(cfg, df_x, window_size=len(df_x), verbose=0)
        feat_y = tsfel.time_series_features_extractor(cfg, df_y, window_size=len(df_y), verbose=0)
        feat_z = tsfel.time_series_features_extractor(cfg, df_z, window_size=len(df_z), verbose=0)

        # Combine features from X, Y and Z
        feat_combined = pd.concat([feat_x, feat_y, feat_z], axis=1)
        # Remove any duplicate columns if present
        feat_combined = feat_combined.loc[:, ~feat_combined.columns.duplicated()]
        if feat_combined.isnull().values.any():
            continue
        feature_vector = feat_combined.values[0]

        if expected_length is None:
            expected_length = len(feature_vector)
            print(f"✅ Feature vector length: {expected_length}")
        if len(feature_vector) != expected_length:
            continue

        X_features.append(feature_vector)
        valid_labels.append(label)
    except Exception as e:
        continue

X_features = np.array(X_features)
y_labels = np.array(valid_labels)
print("✅ Final feature matrix shape:", X_features.shape)

# -------------------------------
# PART 2: TRAIN MLP WITH ALL TIME FEATURES
# -------------------------------
# Encode labels (now there should be 3 classes)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)
num_classes = len(label_encoder.classes_)
print("✅ Classes:", label_encoder.classes_)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)

# Define MLP architecture
def build_mlp(input_dim, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

mlp_model = build_mlp(X_train.shape[1], num_classes)
mlp_model.summary()

# Train the model, storing the history (loss & accuracy during training and validation)
history = mlp_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=100, batch_size=16, verbose=1)

# Evaluate on test set
loss, acc = mlp_model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Final Test Accuracy: {acc:.4f}")

# -------------------------------
# PART 3: SAVE THE MODEL
# -------------------------------
os.makedirs("saved_models", exist_ok=True)
model_path = "saved_models/MLP_all_time_features.h5"
mlp_model.save(model_path)
print(f"✅ MLP model saved as '{model_path}'")

# If running in Google Colab, you can enable download:
try:
    from google.colab import files
    files.download(model_path)
except ImportError:
    pass

# -------------------------------
# PART 4: EVALUATION & PLOTTING
# -------------------------------
# Generate predictions on the test set
y_pred_prob = mlp_model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Compute the confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred)
print("✅ Confusion Matrix:")
print(cm)

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("✅ Classification Report:")
print(report)

weighted_f1 = f1_score(y_test, y_pred, average='weighted')
weighted_recall = recall_score(y_test, y_pred, average='weighted')
print(f"Weighted F1 Score: {weighted_f1:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
plt.yticks(tick_marks, label_encoder.classes_)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

# Plot training and validation loss and accuracy curves
epochs = range(1, len(history.history['loss']) + 1)

plt.figure(figsize=(12, 5))
# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['loss'], 'bo-', label='Training Loss')
plt.plot(epochs, history.history['val_loss'], 'ro-', label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['accuracy'], 'bo-', label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'ro-', label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

#Classes (Bicep_Curl, Shoulder_Raise, and Press) in this format. Probability.
