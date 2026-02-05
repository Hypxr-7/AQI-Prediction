import pandas as pd
import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# MongoDB connection
MONGODB_USERNAME = os.getenv("MONGODB_USERNAME")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")
MONGODB_CLUSTER = os.getenv("MONGODB_CLUSTER")
CONNECTION_STRING = f"mongodb+srv://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_CLUSTER}/"

client = MongoClient(CONNECTION_STRING)
db = client["aqi_prediction"]
feature_store = db["feature_store"]

print("=" * 80)
print("PREPARING TRAINING DATA FROM FEATURE STORE")
print("=" * 80)

try:
    # Fetch all data from feature store
    print("\n[1/2] Fetching data from MongoDB feature store...")
    cursor = feature_store.find({}, {"_id": 0, "inserted_at": 0})
    df = pd.DataFrame(list(cursor))
    
    if len(df) == 0:
        print("   ✗ No data found in feature store!")
        exit(1)
    
    print(f"   ✓ Fetched {len(df)} records")
    print(f"   Columns: {list(df.columns)}")
    
    # Sort by timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    # Save to CSV for training
    print("\n[2/2] Saving to CSV...")
    output_file = 'hourly_aqi_weather_data.csv'
    df.to_csv(output_file, index=False)
    print(f"   ✓ Data saved to {output_file}")
    
    # Display summary
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Total records: {len(df)}")
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    if 'epa_aqi' in df.columns:
        print(f"\nAQI distribution:")
        print(df['epa_aqi'].value_counts().sort_index())
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETED!")
    print("=" * 80)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
