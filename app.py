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
def load_best_model_from_registry():
    """Load the model with highest accuracy from MongoDB model registry"""
    try:
        client = get_mongodb_client()
        db = client["aqi_prediction"]
        registry = db["model_registry"]
        
        # Get model with highest accuracy
        doc = registry.find_one(
            {'status': 'active'},
            sort=[('accuracy', -1), ('training_date', -1)]
        )
        
        if not doc:
            return None
        
        # Deserialize model and scaler
        model_bytes = BytesIO(doc['model_binary'])
        scaler_bytes = BytesIO(doc['scaler_binary'])
        
        model = pickle.load(model_bytes)
        scaler = pickle.load(scaler_bytes)
        
        model_info = {
            'model': model,
            'scaler': scaler,
            'model_name': doc.get('model_name', 'Unknown'),
            'model_type': doc.get('model_type', 'unknown'),
            'feature_cols': doc.get('features', []),
            'accuracy': doc.get('accuracy', 0.0),
            'training_date': doc.get('training_date'),
            'forecast_hours': doc.get('forecast_hours', 72),
            'version': doc.get('version', 'N/A')
        }
        
        return model_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data(ttl=1800)
def fetch_latest_data_from_mongodb():
    """Fetch the latest 72 entries from MongoDB feature store"""
    try:
        client = get_mongodb_client()
        db = client["aqi_prediction"]
        fs = db["feature_store"]
        
        # Get latest 72 records sorted by timestamp
        cursor = fs.find({}, {"_id": 0}).sort("timestamp", -1).limit(72)
        df = pd.DataFrame(list(cursor))
        
        if df.empty:
            st.error("No data found in feature store")
            return None
        
        # Reverse to get chronological order (oldest to newest)
        df = df.iloc[::-1].reset_index(drop=True)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except Exception as e:
        st.error(f"Error fetching data from MongoDB: {e}")
        return None

def predict_aqi(model_info, features_df):
    """Make predictions using the best model"""
    try:
        # Extract features in correct order
        feature_cols = model_info['feature_cols']
        X = features_df[feature_cols]
        
        # Scale features
        X_scaled = model_info['scaler'].transform(X)
        
        # Make predictions
        predictions = model_info['model'].predict(X_scaled)
        
        return predictions.astype(int)
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        return None

def plot_predictions(predictions_df, model_name):
    """Plot AQI predictions"""
    fig = go.Figure()
    
    # Main prediction line
    fig.add_trace(go.Scatter(
        x=predictions_df['predicted_time'],
        y=predictions_df['predicted_aqi'],
        mode='lines+markers',
        name=f'{model_name} Predictions',
        line=dict(width=3, color='#1f77b4'),
        marker=dict(size=6),
        hovertemplate='<b>Predicted AQI</b><br>' +
                      'Time: %{x}<br>' +
                      'AQI Level: %{y}<br>' +
                      'Category: %{customdata}<br>' +
                      '<extra></extra>',
        customdata=predictions_df['predicted_label']
    ))
    
    # Add AQI level background colors
    fig.add_hrect(y0=0.5, y1=1.5, fillcolor=AQI_LABELS[1]['color'], opacity=0.1, line_width=0)
    fig.add_hrect(y0=1.5, y1=2.5, fillcolor=AQI_LABELS[2]['color'], opacity=0.1, line_width=0)
    fig.add_hrect(y0=2.5, y1=3.5, fillcolor=AQI_LABELS[3]['color'], opacity=0.1, line_width=0)
    fig.add_hrect(y0=3.5, y1=4.5, fillcolor=AQI_LABELS[4]['color'], opacity=0.1, line_width=0)
    fig.add_hrect(y0=4.5, y1=5.5, fillcolor=AQI_LABELS[5]['color'], opacity=0.1, line_width=0)
    
    fig.update_layout(
        title=f'AQI Predictions - Next 72 Hours (Using {model_name})',
        xaxis_title='Predicted Time',
        yaxis_title='AQI Level',
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3, 4, 5],
            ticktext=['Good', 'Fair', 'Moderate', 'Poor', 'Very Poor'],
            range=[0.5, 5.5]
        ),
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_input_conditions(input_df):
    """Plot input environmental conditions used for predictions"""
    fig = go.Figure()
    
    # Temperature
    fig.add_trace(go.Scatter(
        x=input_df['timestamp'],
        y=input_df['temp_c'],
        mode='lines',
        name='Temperature (°C)',
        yaxis='y',
        line=dict(color='#ff6b6b', width=2)
    ))
    
    # Humidity
    fig.add_trace(go.Scatter(
        x=input_df['timestamp'],
        y=input_df['humidity_pct'],
        mode='lines',
        name='Humidity (%)',
        yaxis='y2',
        line=dict(color='#4ecdc4', width=2)
    ))
    
    fig.update_layout(
        title='Input Environmental Conditions (Latest 72 Hours)',
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
    st.markdown("72-hour Air Quality Index predictions using historical data and ML models")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application predicts Air Quality Index (AQI) for the next 72 hours using:
        - **Best performing ML model** from model registry
        - **Historical data** from MongoDB feature store
        
        ### AQI Categories
        - 🟢 **Good (1)**: Air quality is satisfactory
        - 🟡 **Fair (2)**: Acceptable air quality
        - 🟠 **Moderate (3)**: Sensitive groups may experience effects
        - 🔴 **Poor (4)**: Health effects for everyone
        - 🟣 **Very Poor (5)**: Health alert, everyone may experience serious effects
        
        ### How It Works
        The model uses the latest 72 hours of environmental data to predict AQI levels 72 hours into the future.
        
        ### Data Source
        - Historical Data: MongoDB Feature Store
        - Models: MongoDB Model Registry
        """)
        
        if st.button("🔄 Refresh Predictions"):
            st.cache_data.clear()
            st.rerun()
    
    # Load best model
    with st.spinner("Loading best model from registry..."):
        model_info = load_best_model_from_registry()
    
    if not model_info:
        st.error("❌ Failed to load model from registry. Please check MongoDB connection.")
        return
    
    # Display model info
    st.success(f"✅ Successfully loaded best model: {model_info['model_name']}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Type", model_info['model_type'].upper())
    with col2:
        st.metric("Accuracy", f"{model_info['accuracy']:.2%}")
    with col3:
        st.metric("Forecast Horizon", f"{model_info['forecast_hours']} hrs")
    with col4:
        st.metric("Version", model_info['version'][:8])
    
    st.divider()
    
    # Fetch latest data from MongoDB
    with st.spinner("Fetching latest data from MongoDB..."):
        data_df = fetch_latest_data_from_mongodb()
    
    if data_df is None or data_df.empty:
        st.error("❌ Failed to fetch data from MongoDB feature store.")
        return
    
    st.info(f"📊 Using latest {len(data_df)} data points from feature store")
    
    # Make predictions
    with st.spinner("Generating 72-hour predictions..."):
        predictions = predict_aqi(model_info, data_df)
    
    if predictions is None:
        st.error("❌ Failed to generate predictions.")
        return
    
    # Create predictions dataframe with future timestamps
    # Predictions are for 72 hours ahead from the last timestamp
    last_timestamp = data_df['timestamp'].iloc[-1]
    predicted_times = [last_timestamp + timedelta(hours=i) for i in range(1, 73)]
    
    predictions_df = pd.DataFrame({
        'predicted_time': predicted_times,
        'predicted_aqi': predictions,
        'predicted_label': [AQI_LABELS.get(int(p), {}).get('name', 'Unknown') for p in predictions]
    })
    
    # Plot predictions
    st.subheader("🔮 AQI Predictions (Next 72 Hours)")
    fig_aqi = plot_predictions(predictions_df, model_info['model_name'])
    st.plotly_chart(fig_aqi, use_container_width=True)
    
    # Plot input conditions
    st.subheader("📈 Input Environmental Conditions")
    fig_input = plot_input_conditions(data_df)
    st.plotly_chart(fig_input, use_container_width=True)
    
    # Predictions table
    st.subheader("📋 Detailed Predictions")
    
    # Show first 24 hours
    st.dataframe(
        predictions_df.head(24),
        use_container_width=True,
        hide_index=True
    )
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_aqi = predictions.mean()
        st.metric("Average Predicted AQI", f"{avg_aqi:.2f}")
    with col2:
        most_common = pd.Series(predictions).mode()[0]
        st.metric("Most Common Category", AQI_LABELS.get(most_common, {}).get('name', 'Unknown'))
    with col3:
        worst_aqi = predictions.max()
        st.metric("Worst Predicted Level", AQI_LABELS.get(worst_aqi, {}).get('name', 'Unknown'))
    
    # Download button
    csv = predictions_df.to_csv(index=False)
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
    Predictions based on latest feature store data | Model automatically selected by highest accuracy
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
