import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv
import ssl
from pymongo import MongoClient
import joblib
import pickle
from io import BytesIO

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AQI Predictor - Karachi",
    page_icon="🌫️",
    layout="wide"
)

# Constants
LAT = 24.8607  # Karachi
LON = 67.0011
MONGODB_USERNAME = os.getenv("MONGODB_USERNAME")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")
MONGODB_CLUSTER = os.getenv("MONGODB_CLUSTER")

# AQI Labels and Colors
AQI_LABELS = {
    1: {'name': 'Good', 'color': '#00e400'},
    2: {'name': 'Fair', 'color': '#ffff00'},
    3: {'name': 'Moderate', 'color': '#ff7e00'},
    4: {'name': 'Poor', 'color': '#ff0000'},
    5: {'name': 'Very Poor', 'color': '#8f3f97'}
}

@st.cache_resource
def get_mongodb_client():
    """Get MongoDB client connection"""
    CONNECTION_STRING = f"mongodb+srv://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_CLUSTER}"
    
    client = MongoClient(
        CONNECTION_STRING
    )
    return client

@st.cache_data(ttl=3600)
def load_models_from_registry():
    """Load the latest models from MongoDB model registry"""
    try:
        client = get_mongodb_client()
        db = client["aqi_prediction"]
        registry = db["model_registry"]
        
        models = {}
        model_types = ['xgboost', 'random_forest', 'svm']
        
        for model_type in model_types:
            # Get latest model of this type
            doc = registry.find_one(
                {'model_type': model_type},
                sort=[('registered_at', -1)]
            )
            
            if doc:
                # Deserialize model and scaler
                model_bytes = BytesIO(doc['model_binary'])
                scaler_bytes = BytesIO(doc['scaler_binary'])
                
                model = pickle.load(model_bytes)
                scaler = joblib.load(scaler_bytes)
                
                models[model_type] = {
                    'model': model,
                    'scaler': scaler,
                    'feature_cols': doc.get('feature_cols', []),
                    'accuracy': doc.get('accuracy', doc.get('metrics', {}).get('accuracy', 0.0)),
                    'registered_at': doc.get('registered_at'),
                    'class_mapping': doc.get('additional_info', {}).get('class_mapping'),
                    'reverse_mapping': doc.get('additional_info', {}).get('reverse_mapping')
                }
                
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}

@st.cache_data(ttl=1800)
def fetch_forecast_data():
    """Fetch weather forecast for next 3 days"""
    try:
        # Open-Meteo forecast API
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': LAT,
            'longitude': LON,
            'hourly': 'temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m,rain,direct_radiation',
            'forecast_days': 3,
            'timezone': 'Asia/Karachi'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        hourly = data['hourly']
        df = pd.DataFrame({
            'time': pd.to_datetime(hourly['time']),
            'temp_c': hourly['temperature_2m'],
            'humidity_pct': hourly['relative_humidity_2m'],
            'pressure_hpa': hourly['pressure_msl'],
            'wind_speed_kmh': hourly['wind_speed_10m'],
            'wind_dir_deg': hourly['wind_direction_10m'],
            'rain_mm': hourly['rain'],
            'solar_rad_wm2': hourly['direct_radiation']
        })
        
        return df
    except Exception as e:
        st.error(f"Error fetching forecast data: {e}")
        return None

@st.cache_data(ttl=1800)
def fetch_current_pollution():
    """Fetch current air pollution data"""
    try:
        API_KEY = os.getenv("OPENWEATHER_API_KEY")
        if not API_KEY:
            st.warning("OpenWeather API key not found. Using default pollution values.")
            return {
                'pm2_5': 35.0,
                'pm10': 50.0,
                'co': 400.0,
                'no2': 30.0,
                'so2': 10.0,
                'o3': 60.0
            }
        
        url = "http://api.openweathermap.org/data/2.5/air_pollution"
        params = {'lat': LAT, 'lon': LON, 'appid': API_KEY}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        components = data['list'][0]['components']
        return {
            'pm2_5': components['pm2_5'],
            'pm10': components['pm10'],
            'co': components['co'],
            'no2': components['no2'],
            'so2': components['so2'],
            'o3': components['o3']
        }
    except Exception as e:
        st.warning(f"Error fetching pollution data: {e}. Using defaults.")
        return {
            'pm2_5': 35.0,
            'pm10': 50.0,
            'co': 400.0,
            'no2': 30.0,
            'so2': 10.0,
            'o3': 60.0
        }

def prepare_features(weather_df, pollution_data):
    """Prepare features for prediction"""
    # Add pollution data (assume constant for forecast)
    for key, value in pollution_data.items():
        weather_df[key] = value
    
    # Add temporal features
    weather_df['hour'] = weather_df['time'].dt.hour
    weather_df['month'] = weather_df['time'].dt.month
    
    # Feature order (must match training)
    feature_cols = ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3',
                    'temp_c', 'humidity_pct', 'pressure_hpa', 
                    'wind_speed_kmh', 'wind_dir_deg', 'rain_mm', 'solar_rad_wm2',
                    'hour', 'month']
    
    return weather_df[feature_cols]

def predict_aqi(models, features_df):
    """Make predictions using all models"""
    predictions = {}
    
    for model_name, model_data in models.items():
        try:
            # Scale features
            features_scaled = model_data['scaler'].transform(features_df)
            
            # Make predictions
            y_pred = model_data['model'].predict(features_scaled)
            
            # Map back to original classes if mapping exists
            if model_name == 'xgboost' and model_data['reverse_mapping']:
                reverse_map = {int(k): int(v) for k, v in model_data['reverse_mapping'].items()}
                y_pred = np.array([reverse_map.get(pred, pred) for pred in y_pred])
            
            predictions[model_name] = y_pred
        except Exception as e:
            st.error(f"Error predicting with {model_name}: {e}")
            predictions[model_name] = None
    
    return predictions

def plot_predictions(weather_df, predictions):
    """Plot AQI predictions for all models"""
    fig = go.Figure()
    
    # Color mapping for models
    model_colors = {
        'xgboost': '#1f77b4',
        'randomforest': '#2ca02c',
        'svm': '#ff7f0e'
    }
    
    for model_name, preds in predictions.items():
        if preds is not None:
            fig.add_trace(go.Scatter(
                x=weather_df['time'],
                y=preds,
                mode='lines+markers',
                name=model_name.upper(),
                line=dict(width=2, color=model_colors.get(model_name, '#000000')),
                marker=dict(size=6),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Time: %{x}<br>' +
                              'AQI Level: %{y}<br>' +
                              '<extra></extra>'
            ))
    
    # Add AQI level background colors
    fig.add_hrect(y0=0.5, y1=1.5, fillcolor=AQI_LABELS[1]['color'], opacity=0.1, line_width=0)
    fig.add_hrect(y0=1.5, y1=2.5, fillcolor=AQI_LABELS[2]['color'], opacity=0.1, line_width=0)
    fig.add_hrect(y0=2.5, y1=3.5, fillcolor=AQI_LABELS[3]['color'], opacity=0.1, line_width=0)
    fig.add_hrect(y0=3.5, y1=4.5, fillcolor=AQI_LABELS[4]['color'], opacity=0.1, line_width=0)
    fig.add_hrect(y0=4.5, y1=5.5, fillcolor=AQI_LABELS[5]['color'], opacity=0.1, line_width=0)
    
    fig.update_layout(
        title='AQI Predictions - Next 72 Hours',
        xaxis_title='Time',
        yaxis_title='AQI Level',
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3, 4, 5],
            ticktext=['Good', 'Fair', 'Moderate', 'Poor', 'Very Poor']
        ),
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_weather_conditions(weather_df):
    """Plot weather conditions"""
    fig = go.Figure()
    
    # Temperature
    fig.add_trace(go.Scatter(
        x=weather_df['time'],
        y=weather_df['temp_c'],
        mode='lines',
        name='Temperature (°C)',
        yaxis='y',
        line=dict(color='#ff6b6b', width=2)
    ))
    
    # Humidity
    fig.add_trace(go.Scatter(
        x=weather_df['time'],
        y=weather_df['humidity_pct'],
        mode='lines',
        name='Humidity (%)',
        yaxis='y2',
        line=dict(color='#4ecdc4', width=2)
    ))
    
    fig.update_layout(
        title='Weather Conditions - Next 72 Hours',
        xaxis_title='Time',
        yaxis=dict(
            title=dict(text='Temperature (°C)', font=dict(color='#ff6b6b')),
            tickfont=dict(color='#ff6b6b')
        ),
        yaxis2=dict(
            title=dict(text='Humidity (%)', font=dict(color='#4ecdc4')),
            tickfont=dict(color='#4ecdc4'),
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Main App
def main():
    st.title("🌫️ AQI Predictor - Karachi, Pakistan")
    st.markdown("Real-time Air Quality Index predictions for the next 3 days")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application predicts Air Quality Index (AQI) for Karachi using:
        - **XGBoost Classifier**
        - **Random Forest Classifier**
        - **Support Vector Machine (SVM)**
        
        ### AQI Categories
        - 🟢 **Good (1)**: Air quality is satisfactory
        - 🟡 **Fair (2)**: Acceptable air quality
        - 🟠 **Moderate (3)**: Sensitive groups may experience effects
        - 🔴 **Poor (4)**: Health effects for everyone
        - 🟣 **Very Poor (5)**: Health alert, everyone may experience serious effects
        
        ### Data Sources
        - Weather: Open-Meteo API
        - Pollution: OpenWeatherMap API
        - Models: MongoDB Model Registry
        """)
        
        if st.button("🔄 Refresh Predictions"):
            st.cache_data.clear()
            st.rerun()
    
    # Load models
    with st.spinner("Loading models from registry..."):
        models = load_models_from_registry()
    
    if not models:
        st.error("❌ Failed to load models from registry. Please check MongoDB connection.")
        return
    
    # Display model info
    st.success(f"✅ Successfully loaded {len(models)} models")
    
    cols = st.columns(len(models))
    for idx, (model_name, model_data) in enumerate(models.items()):
        with cols[idx]:
            st.metric(
                label=f"{model_name.upper()}",
                value=f"{model_data['accuracy']:.2%}",
                delta="Accuracy"
            )
    
    st.divider()
    
    # Plot predictions
    st.subheader("🔮 AQI Predictions")
    fig_aqi = plot_predictions(weather_df, predictions)
    st.plotly_chart(fig_aqi, use_container_width=True)
    
    # Plot weather conditions
    st.subheader("🌤️ Weather Forecast")
    fig_weather = plot_weather_conditions(weather_df)
    st.plotly_chart(fig_weather, use_container_width=True)
    
    # Predictions table
    st.subheader("📋 Detailed Predictions")
    
    # Create comparison dataframe
    results_df = weather_df[['time']].copy()
    for model_name, preds in predictions.items():
        if preds is not None:
            results_df[f'{model_name}_aqi'] = preds
            results_df[f'{model_name}_label'] = [AQI_LABELS.get(int(p), {}).get('name', 'Unknown') for p in preds]
    
    # Show next 24 hours
    st.dataframe(
        results_df.head(24),
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Predictions (CSV)",
        data=csv,
        file_name=f"aqi_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Data updated every 1 hour | Models retrained daily
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
