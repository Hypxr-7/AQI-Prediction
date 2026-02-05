import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config.utils import load_and_preprocess_data, save_and_register_model

print("="*80)
print("AQI PREDICTION - Random Forest Training (72-hour forecast)")
print("="*80)

# Load and preprocess data
X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols = load_and_preprocess_data()

# Train Random Forest model
print("\n[6/7] Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=1
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

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n   Top 5 Feature Importances:")
print(feature_importance.head())

# Store model directly in MongoDB (no local files)
model_id = save_and_register_model(
    model=model,
    scaler=scaler,
    model_type='random_forest',
    model_name='Random Forest Classifier (72h forecast)',
    accuracy=accuracy,
    feature_cols=feature_cols,
    train_size=len(y_train),
    test_size=len(y_test),
    hyperparameters={
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt"
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
