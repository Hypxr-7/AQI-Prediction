import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config.utils import load_and_preprocess_data, save_and_register_model

print("="*80)
print("AQI PREDICTION - SVM Training (72-hour forecast)")
print("="*80)

# Load and preprocess data
X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols = load_and_preprocess_data()

# Train SVM model
print("\n[6/7] Training SVM model...")
print("   Note: SVM training may take longer for large datasets")
model = SVC(
    kernel='rbf',
    C=10.0,
    gamma='scale',
    random_state=42,
    verbose=True
)

model.fit(X_train_scaled, y_train)
print("   ✓ Training completed")

# Evaluate
print("\n[7/7] Evaluating model...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n   Accuracy: {accuracy:.4f}")
print("\n   Classification Report:")

# Dynamic target names based on actual classes present
aqi_labels = {1: 'Good', 2: 'Fair', 3: 'Moderate', 4: 'Poor', 5: 'Very Poor'}
unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
target_names = [aqi_labels[c] for c in unique_classes]

print(classification_report(y_test, y_pred, labels=unique_classes, target_names=target_names))
print("\n   Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# SVM statistics
print("\n   SVM Statistics:")
print(f"   - Number of support vectors: {model.n_support_}")
print(f"   - Support vectors per class: {model.n_support_}")

# Store model directly in MongoDB (no local files)
model_id = save_and_register_model(
    model=model,
    scaler=scaler,
    model_type='svm',
    model_name='Support Vector Machine (72h forecast)',
    accuracy=accuracy,
    feature_cols=feature_cols,
    train_size=len(y_train),
    test_size=len(y_test),
    hyperparameters={
        "kernel": "rbf",
        "C": 10.0,
        "gamma": "scale"
    },
    classification_report_dict=classification_report(y_test, y_pred, output_dict=True),
    extra_metadata={
        "n_support_vectors": int(sum(model.n_support_))
    },
    forecast_hours=72
)

print("\n" + "="*80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print(f"Model ID: {model_id}")
print("="*80)
