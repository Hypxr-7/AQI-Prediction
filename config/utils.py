import os
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

load_dotenv()

username = os.getenv("MONGODB_USERNAME")
password = os.getenv("MONGODB_PASSWORD")
cluster_url = os.getenv("MONGODB_CLUSTER")
CONNECTION_STRING = f"mongodb+srv://{username}:{password}@{cluster_url}/"

client = MongoClient(CONNECTION_STRING)

def get_dataset():
    """Fetch dataset from MongoDB feature store"""
    db = client["aqi_prediction"]
    fs = db["feature_store"]

    cursor = fs.find({}, {"_id": 0})
    df = pd.DataFrame(list(cursor))

    return df

def get_model_registry():
    """Get MongoDB model registry collection"""
    db = client["aqi_prediction"]
    return db["model_registry"]

def load_and_preprocess_data(forecast_hours=72):
    """
    Load and preprocess data for training
    
    Args:
        forecast_hours: Number of hours ahead to predict (default: 72)
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_cols
    """
    print(f"\n[1/5] Loading data (predicting {forecast_hours} hours ahead)...")
    df = get_dataset()
    print(f"   Loaded {len(df)} records")
    
    # Sort by timestamp to ensure proper ordering
    if 'timestamp' not in df.columns:
        raise ValueError("Dataset must contain 'timestamp' column for time-series forecasting")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"   Data sorted by timestamp")
    
    print(f"\n[2/5] Creating {forecast_hours}-hour ahead target...")
    # Shift target variable by -forecast_hours to create future prediction target
    # This means we're using current features to predict AQI 72 hours in the future
    df['target_aqi'] = df['epa_aqi'].shift(-forecast_hours)
    
    # Remove rows where target is NaN (last forecast_hours rows)
    df_clean = df.dropna(subset=['target_aqi'])
    print(f"   Created target for {len(df_clean)} records ({forecast_hours} records removed from end)")
    
    print("\n[3/5] Feature engineering...")
    feature_cols = ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3',
                    'temp_c', 'humidity_pct', 'pressure_hpa', 
                    'wind_speed_kmh', 'wind_dir_deg', 'rain_mm', 'solar_rad_wm2',
                    'hour', 'month']
    
    # Drop rows with missing feature values
    df_clean = df_clean[feature_cols + ['target_aqi']].dropna()
    print(f"   Records after removing NaN: {len(df_clean)}")
    
    X = df_clean[feature_cols]
    y = df_clean['target_aqi'].astype(int)
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target distribution:\n{y.value_counts().sort_index()}")
    
    print("\n[4/5] Splitting data (time-aware split)...")
    # Use time-aware split: train on earlier data, test on later data
    # This simulates real-world forecasting scenario
    split_idx = int(len(df_clean) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    print("\n[5/5] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler, feature_cols

def save_and_register_model(model, scaler, model_type, model_name, accuracy, 
                            feature_cols, train_size, test_size, 
                            hyperparameters, classification_report_dict,
                            extra_metadata=None, forecast_hours=72):
    """
    Serialize model and scaler, store directly in MongoDB model registry
    
    Args:
        model: Trained model object
        scaler: Fitted scaler object
        model_type: Type of model (e.g., 'xgboost', 'random_forest', 'svm', 'lstm')
        model_name: Display name of the model
        accuracy: Model accuracy score
        feature_cols: List of feature column names
        train_size: Number of training samples
        test_size: Number of test samples
        hyperparameters: Dict of model hyperparameters
        classification_report_dict: Classification report as dictionary
        extra_metadata: Optional dict with additional metadata (e.g., feature_importance)
        forecast_hours: Number of hours ahead the model predicts (default: 72)
    
    Returns:
        mongodb_id
    """
    import pickle
    import io
    from bson import Binary
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\nSerializing model and scaler...")
    
    # Serialize model to binary
    if model_type == 'lstm':
        # For LSTM/Keras models, save to BytesIO
        model_buffer = io.BytesIO()
        model.save(model_buffer, save_format='h5')
        model_binary = model_buffer.getvalue()
    else:
        # For sklearn/xgboost models, use pickle
        model_binary = pickle.dumps(model)
    
    # Serialize scaler to binary
    scaler_binary = pickle.dumps(scaler)
    
    print(f"   ✓ Model serialized ({len(model_binary) / 1024:.2f} KB)")
    print(f"   ✓ Scaler serialized ({len(scaler_binary) / 1024:.2f} KB)")
    
    # Store in MongoDB
    print("\nStoring model in MongoDB model registry...")
    model_registry = get_model_registry()
    
    model_metadata = {
        "model_name": model_name,
        "model_type": model_type,
        "version": timestamp,
        "accuracy": float(accuracy),
        "training_date": datetime.utcnow(),
        "model_binary": Binary(model_binary),
        "scaler_binary": Binary(scaler_binary),
        "model_size_kb": len(model_binary) / 1024,
        "scaler_size_kb": len(scaler_binary) / 1024,
        "features": feature_cols,
        "target": "epa_aqi",
        "forecast_hours": forecast_hours,
        "train_size": int(train_size),
        "test_size": int(test_size),
        "hyperparameters": hyperparameters,
        "classification_report": classification_report_dict,
        "status": "active"
    }
    
    # Add extra metadata if provided
    if extra_metadata:
        model_metadata.update(extra_metadata)
    
    result = model_registry.insert_one(model_metadata)
    print(f"   ✓ Model stored in MongoDB with ID: {result.inserted_id}")
    
    return result.inserted_id

def load_model_from_registry(model_id=None, model_type=None, latest=True):
    """
    Load a model and scaler from MongoDB model registry
    
    Args:
        model_id: Specific MongoDB ObjectId to load
        model_type: Type of model to load (if model_id not provided)
        latest: If True, load the latest model (highest accuracy)
    
    Returns:
        model, scaler, metadata
    """
    import pickle
    import io
    from tensorflow import keras
    
    model_registry = get_model_registry()
    
    if model_id:
        # Load specific model by ID
        doc = model_registry.find_one({"_id": model_id})
    elif model_type:
        # Load latest model of specific type
        if latest:
            doc = model_registry.find_one(
                {"model_type": model_type, "status": "active"},
                sort=[("accuracy", -1), ("training_date", -1)]
            )
        else:
            doc = model_registry.find_one(
                {"model_type": model_type, "status": "active"},
                sort=[("training_date", -1)]
            )
    else:
        # Load best model overall
        doc = model_registry.find_one(
            {"status": "active"},
            sort=[("accuracy", -1), ("training_date", -1)]
        )
    
    if not doc:
        raise ValueError("No model found matching criteria")
    
    print(f"Loading model: {doc['model_name']} (v{doc['version']})")
    print(f"Accuracy: {doc['accuracy']:.4f}")
    
    # Deserialize model
    if doc['model_type'] == 'lstm':
        model_buffer = io.BytesIO(doc['model_binary'])
        model = keras.models.load_model(model_buffer)
    else:
        model = pickle.loads(doc['model_binary'])
    
    # Deserialize scaler
    scaler = pickle.loads(doc['scaler_binary'])
    
    # Return metadata without binary data
    metadata = {k: v for k, v in doc.items() if k not in ['model_binary', 'scaler_binary', '_id']}
    metadata['_id'] = str(doc['_id'])
    
    return model, scaler, metadata
