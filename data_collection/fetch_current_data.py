import requests
import os
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from datetime import datetime

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = 24.8607  # Karachi
LON = 67.0011
MONGODB_USERNAME = os.getenv("MONGODB_USERNAME")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")
MONGODB_CLUSTER = os.getenv("MONGODB_CLUSTER")

# MongoDB connection
CONNECTION_STRING = f"mongodb+srv://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_CLUSTER}/"
client = MongoClient(CONNECTION_STRING)
db = client["aqi_prediction"]
feature_store = db["feature_store"]

print("=" * 80)
print("FETCHING CURRENT AQI AND WEATHER DATA")
print("=" * 80)

try:
    # 1. Fetch current air pollution data
    print("\n[1/3] Fetching current air pollution data...")
    pollution_url = "http://api.openweathermap.org/data/2.5/air_pollution"
    p_params = {'lat': LAT, 'lon': LON, 'appid': API_KEY}
    p_res = requests.get(pollution_url, params=p_params).json()
    
    if 'list' not in p_res or len(p_res['list']) == 0:
        print("   ✗ Failed to fetch pollution data")
        exit(1)
    
    pollution_data = p_res['list'][0]
    print("   ✓ Air pollution data fetched")
    
    # 2. Fetch current weather data from Open-Meteo
    print("\n[2/3] Fetching current weather data...")
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        'latitude': LAT,
        'longitude': LON,
        'current': 'temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m,rain,direct_radiation',
        'timezone': 'auto'
    }
    
    w_res = requests.get(weather_url, params=weather_params).json()
    
    if 'current' not in w_res:
        print("   ✗ Failed to fetch weather data")
        exit(1)
    
    weather_data = w_res['current']
    print("   ✓ Weather data fetched")
    
    # 3. Combine and prepare data
    print("\n[3/3] Preparing data for storage...")
    ts = pollution_data['dt']
    dt_obj = datetime.fromtimestamp(ts)
    
    record = {
        "timestamp": ts,
        "year": dt_obj.year,
        "month": dt_obj.month,
        "day": dt_obj.day,
        "hour": dt_obj.hour + 5,
        "epa_aqi": pollution_data['main']['aqi'],
        "pm2_5": pollution_data['components']['pm2_5'],
        "pm10": pollution_data['components']['pm10'],
        "co": pollution_data['components']['co'],
        "no2": pollution_data['components']['no2'],
        "so2": pollution_data['components']['so2'],
        "o3": pollution_data['components']['o3'],
        "temp_c": weather_data.get('temperature_2m'),
        "humidity_pct": weather_data.get('relative_humidity_2m'),
        "pressure_hpa": weather_data.get('pressure_msl'),
        "wind_speed_kmh": weather_data.get('wind_speed_10m'),
        "wind_dir_deg": weather_data.get('wind_direction_10m'),
        "rain_mm": weather_data.get('rain', 0),
        "solar_rad_wm2": weather_data.get('direct_radiation'),
        "inserted_at": datetime.utcnow()
    }
    
    # 4. Store in MongoDB (upsert to avoid duplicates)
    print("\n[4/4] Storing data in MongoDB...")
    result = feature_store.update_one(
        {'timestamp': ts},
        {'$set': record},
        upsert=True
    )
    
    if result.upserted_id:
        print(f"   ✓ New record inserted")
    else:
        print(f"   ✓ Existing record updated")
    
    print(f"\n   Timestamp: {dt_obj}")
    print(f"   EPA AQI: {record['epa_aqi']}")
    print(f"   PM2.5: {record['pm2_5']} µg/m³")
    print(f"   Temperature: {record['temp_c']}°C")
    print(f"   Humidity: {record['humidity_pct']}%")
    
    # 5. Verify total records
    total_records = feature_store.count_documents({})
    print(f"\n   Total records in feature store: {total_records}")
    
    print("\n" + "=" * 80)
    print("DATA FETCHING COMPLETED SUCCESSFULLY!")
    print("=" * 80)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
