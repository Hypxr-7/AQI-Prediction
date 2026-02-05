import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from config.utils import load_and_preprocess_data, save_and_register_model

print("="*80)
print("AQI PREDICTION - XGBoost Training (72-hour forecast)")
print("="*80)

# Load and preprocess data
X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols = load_and_preprocess_data()

# Map AQI classes to 0-indexed for XGBoost
print("\n[6/9] Mapping AQI classes to 0-indexed labels...")
unique_classes = sorted(np.unique(np.concatenate([y_train, y_test])))
class_mapping = {original: idx for idx, original in enumerate(unique_classes)}
reverse_mapping = {idx: original for original, idx in class_mapping.items()}

y_train_mapped = np.array([class_mapping[label] for label in y_train])
y_test_mapped = np.array([class_mapping[label] for label in y_test])
print(f"   ✓ Mapped classes {unique_classes} to {list(range(len(unique_classes)))}")

# Train XGBoost model
print("\n[7/9] Training XGBoost model...")
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(unique_classes),
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(
    X_train_scaled, y_train_mapped,
    eval_set=[(X_test_scaled, y_test_mapped)],
    verbose=False
)
print("   ✓ Training completed")

# Evaluate
print("\n[8/9] Evaluating model...")
y_pred_mapped = model.predict(X_test_scaled)
# Map predictions back to original AQI classes
y_pred = np.array([reverse_mapping[pred] for pred in y_pred_mapped])
accuracy = accuracy_score(y_test, y_pred)

print(f"\n   Accuracy: {accuracy:.4f}")
print("\n   Classification Report:")

# Dynamic target names based on actual classes present
aqi_labels = {1: 'Good', 2: 'Fair', 3: 'Moderate', 4: 'Poor', 5: 'Very Poor'}
target_names = [aqi_labels[c] for c in unique_classes]

print(classification_report(y_test, y_pred, labels=unique_classes, target_names=target_names))
print("\n   Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n   Top 5 Feature Importances:")
print(feature_importance.head())

# Store model directly in MongoDB (no local files)
print("\n[9/9] Storing model in MongoDB...")
model_id = save_and_register_model(
    model=model,
    scaler=scaler,
    model_type='xgboost',
    model_name='XGBoost Classifier (72h forecast)',
    accuracy=accuracy,
    feature_cols=feature_cols,
    train_size=len(y_train),
    test_size=len(y_test),
    hyperparameters={
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },
    classification_report_dict=classification_report(y_test, y_pred, output_dict=True),
    extra_metadata={
        "feature_importance": feature_importance.to_dict('records')
    },
    forecast_hours=72
)

print("\n" + "="*80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print(f"Model ID: {model_id}")
print("="*80)
