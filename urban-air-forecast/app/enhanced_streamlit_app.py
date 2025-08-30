import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import asyncio
import sqlite3
from pathlib import Path
import sys

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from database_setup import DatabaseManager
from realtime_data_integration import RealTimeDataIntegrator

# Page configuration
st.set_page_config(
    page_title="Real-Time Air Quality Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .realtime-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #48bb78;
        border-radius: 50%;
        margin-right: 5px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .data-source-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background-color: #e2e8f0;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_database_data():
    """Load data from database with caching"""
    db_manager = DatabaseManager()
    
    stations = db_manager.get_stations()
    sensor_data = db_manager.get_latest_sensor_data(hours_back=48)
    weather_data = db_manager.get_latest_weather_data(hours_back=48)
    
    return stations, sensor_data, weather_data

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_realtime_data():
    """Get real-time data from APIs"""
    try:
        integrator = RealTimeDataIntegrator()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(integrator.collect_realtime_data())
        loop.close()
        return data
    except Exception as e:
        st.error(f"Error fetching real-time data: {e}")
        return []

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Real-Time Air Quality Forecasting System</h1>
        <p><span class="realtime-indicator"></span>Live data integration with advanced forecasting</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Controls")
        
        # Data source selection
        st.subheader("📡 Data Sources")
        use_realtime = st.checkbox("Enable Real-time Data", value=True)
        use_database = st.checkbox("Use Database", value=True)
        
        # Refresh controls
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Time range selection
        st.subheader("⏰ Time Range")
        hours_back = st.slider("Hours of historical data", 6, 168, 24)
        
        # Station selection
        st.subheader("📍 Station Filter")
        if use_database:
            stations, _, _ = load_database_data()
            if not stations.empty:
                selected_stations = st.multiselect(
                    "Select stations",
                    options=stations['id'].tolist(),
                    default=stations['id'].tolist()[:3]
                )
            else:
                selected_stations = ["ST001", "ST002", "ST003"]
        else:
            selected_stations = st.multiselect(
                "Select stations",
                options=["ST001", "ST002", "ST003"],
                default=["ST001", "ST002", "ST003"]
            )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Real-Time Dashboard", "🔮 Forecasting", "📈 Analytics", "⚙️ System Status"])
    
    with tab1:
        st.header("📊 Real-Time Air Quality Dashboard")
        
        # Real-time data section
        if use_realtime:
            st.subheader("🔴 Live Data Feed")
            
            realtime_data = get_realtime_data()
            
            if realtime_data:
                # Display real-time metrics
                cols = st.columns(len(realtime_data))
                
                for i, data in enumerate(realtime_data):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{data['station_id']}</h4>
                            <div class="realtime-indicator"></div>
                            <span>Live</span>
                            <br><br>
                            <strong>AQI: {data.get('aqi', 'N/A')}</strong><br>
                            <small>PM2.5: {data.get('pm25', 'N/A')} μg/m³</small><br>
                            <small>Source: {data['source']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Real-time data table
                st.subheader("📋 Latest Readings")
                realtime_df = pd.DataFrame(realtime_data)
                st.dataframe(realtime_df, use_container_width=True)
            else:
                st.warning("⚠️ No real-time data available. Check API configuration.")
        
        # Database data section
        if use_database:
            st.subheader("💾 Database Data")
            
            stations, sensor_data, weather_data = load_database_data()
            
            if not sensor_data.empty:
                # Filter by selected stations
                if selected_stations:
                    sensor_data = sensor_data[sensor_data['station_id'].isin(selected_stations)]
                    weather_data = weather_data[weather_data['station_id'].isin(selected_stations)]
                
                # Latest readings
                st.subheader("📊 Current Conditions")
                
                latest_readings = sensor_data.groupby('station_id').last().reset_index()
                
                cols = st.columns(len(latest_readings))
                for i, (_, row) in enumerate(latest_readings.iterrows()):
                    with cols[i]:
                        st.metric(
                            label=f"🏭 {row['station_id']}",
                            value=f"{row['pm25']:.1f} μg/m³" if pd.notna(row['pm25']) else "N/A",
                            delta=f"AQI: {row['aqi']}" if pd.notna(row['aqi']) else None
                        )
                
                # Time series chart
                st.subheader("📈 PM2.5 Trends")
                
                if not sensor_data.empty:
                    sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp'])
                    
                    fig = px.line(
                        sensor_data,
                        x='timestamp',
                        y='pm25',
                        color='station_id',
                        title='PM2.5 Concentration Over Time',
                        labels={'pm25': 'PM2.5 (μg/m³)', 'timestamp': 'Time'}
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Weather correlation
                if not weather_data.empty:
                    st.subheader("🌤️ Weather Conditions")
                    
                    weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
                    latest_weather = weather_data.groupby('station_id').last().reset_index()
                    
                    weather_cols = st.columns(4)
                    with weather_cols[0]:
                        avg_temp = latest_weather['temp_c'].mean()
                        st.metric("🌡️ Temperature", f"{avg_temp:.1f}°C")
                    
                    with weather_cols[1]:
                        avg_humidity = latest_weather['humidity'].mean()
                        st.metric("💧 Humidity", f"{avg_humidity:.0f}%")
                    
                    with weather_cols[2]:
                        avg_wind = latest_weather['wind_speed'].mean()
                        st.metric("💨 Wind Speed", f"{avg_wind:.1f} m/s")
                    
                    with weather_cols[3]:
                        avg_pressure = latest_weather['pressure'].mean()
                        st.metric("📊 Pressure", f"{avg_pressure:.0f} hPa")
            else:
                st.info("📝 No database data available. Run data migration first.")
    
    with tab2:
        st.header("🔮 Air Quality Forecasting")
        
        # Forecast generation interface
        st.subheader("⚙️ Generate New Forecast")
        
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_station = st.selectbox("Select Station", selected_stations)
            forecast_hours = st.slider("Forecast Horizon (hours)", 1, 72, 24)
        
        with col2:
            model_version = st.selectbox("Model Version", ["Ultimate_v5.0", "Enhanced_v4.0", "Standard_v3.0"])
            include_uncertainty = st.checkbox("Include Uncertainty Bands", value=True)
        
        if st.button("🎯 Generate Forecast"):
            with st.spinner("Generating forecast..."):
                # Simulate forecast generation
                forecast_times = pd.date_range(
                    start=datetime.now(),
                    periods=forecast_hours,
                    freq='H'
                )
                
                # Generate realistic forecast data
                base_pm25 = 45 + np.random.normal(0, 10)
                trend = np.random.normal(0, 0.5, forecast_hours).cumsum()
                diurnal = 10 * np.sin(2 * np.pi * np.arange(forecast_hours) / 24)
                noise = np.random.normal(0, 5, forecast_hours)
                
                forecast_values = base_pm25 + trend + diurnal + noise
                forecast_values = np.clip(forecast_values, 5, 300)
                
                uncertainty = 5 + 0.2 * np.arange(forecast_hours)
                
                forecast_df = pd.DataFrame({
                    'timestamp': forecast_times,
                    'pm25_forecast': forecast_values,
                    'uncertainty': uncertainty,
                    'lower_ci': forecast_values - 1.96 * uncertainty,
                    'upper_ci': forecast_values + 1.96 * uncertainty
                })
                
                # Plot forecast
                fig = go.Figure()
                
                # Add forecast line
                fig.add_trace(go.Scatter(
                    x=forecast_df['timestamp'],
                    y=forecast_df['pm25_forecast'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='blue', width=2)
                ))
                
                # Add uncertainty bands
                if include_uncertainty:
                    fig.add_trace(go.Scatter(
                        x=forecast_df['timestamp'],
                        y=forecast_df['upper_ci'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['timestamp'],
                        y=forecast_df['lower_ci'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        name='95% Confidence Interval',
                        fillcolor='rgba(0,100,80,0.2)'
                    ))
                
                fig.update_layout(
                    title=f'PM2.5 Forecast for {forecast_station}',
                    xaxis_title='Time',
                    yaxis_title='PM2.5 (μg/m³)',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast summary
                st.subheader("📋 Forecast Summary")
                
                summary_cols = st.columns(4)
                with summary_cols[0]:
                    st.metric("📊 Mean Forecast", f"{forecast_values.mean():.1f} μg/m³")
                
                with summary_cols[1]:
                    st.metric("📈 Max Forecast", f"{forecast_values.max():.1f} μg/m³")
                
                with summary_cols[2]:
                    st.metric("📉 Min Forecast", f"{forecast_values.min():.1f} μg/m³")
                
                with summary_cols[3]:
                    st.metric("🎯 Avg Uncertainty", f"±{uncertainty.mean():.1f} μg/m³")
    
    with tab3:
        st.header("📈 Advanced Analytics")
        
        if use_database:
            stations, sensor_data, weather_data = load_database_data()
            
            if not sensor_data.empty and not weather_data.empty:
                # Correlation analysis
                st.subheader("🔗 Weather-Pollution Correlation")
                
                # Merge sensor and weather data
                sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp'])
                weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
                
                merged_data = pd.merge(
                    sensor_data[['timestamp', 'station_id', 'pm25']],
                    weather_data[['timestamp', 'station_id', 'temp_c', 'humidity', 'wind_speed']],
                    on=['timestamp', 'station_id'],
                    how='inner'
                )
                
                if not merged_data.empty:
                    # Correlation matrix
                    corr_cols = ['pm25', 'temp_c', 'humidity', 'wind_speed']
                    corr_data = merged_data[corr_cols].corr()
                    
                    fig = px.imshow(
                        corr_data,
                        text_auto=True,
                        aspect="auto",
                        title="Weather-Pollution Correlation Matrix"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Scatter plots
                    st.subheader("🎯 Relationship Analysis")
                    
                    scatter_cols = st.columns(2)
                    
                    with scatter_cols[0]:
                        fig = px.scatter(
                            merged_data,
                            x='temp_c',
                            y='pm25',
                            color='station_id',
                            title='Temperature vs PM2.5',
                            trendline='ols'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with scatter_cols[1]:
                        fig = px.scatter(
                            merged_data,
                            x='wind_speed',
                            y='pm25',
                            color='station_id',
                            title='Wind Speed vs PM2.5',
                            trendline='ols'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Enable database to view analytics")
    
    with tab4:
        st.header("⚙️ System Status")
        
        # Database status
        st.subheader("💾 Database Status")
        
        if use_database:
            db_manager = DatabaseManager()
            stats = db_manager.get_database_stats()
            
            status_cols = st.columns(3)
            
            with status_cols[0]:
                st.metric("🏭 Stations", stats['stations'])
                st.metric("📊 Sensor Records", stats['sensor_data'])
            
            with status_cols[1]:
                st.metric("🌤️ Weather Records", stats['weather_data'])
                st.metric("🔮 Forecasts", stats['forecasts'])
            
            with status_cols[2]:
                st.metric("📋 Policy Records", stats['policy_recommendations'])
        
        # Real-time data status
        st.subheader("📡 Real-time Data Sources")
        
        integrator = RealTimeDataIntegrator()
        config = integrator.config
        
        for source_name, source_config in config['data_sources'].items():
            if isinstance(source_config, dict) and 'enabled' in source_config:
                status = "🟢 Active" if source_config['enabled'] else "🔴 Disabled"
                api_configured = "🔑 Configured" if source_config.get('api_key', '').startswith('YOUR_') == False else "⚠️ Not Configured"
                
                st.markdown(f"""
                <div class="data-source-badge">
                    <strong>{source_name.upper()}</strong><br>
                    Status: {status}<br>
                    API Key: {api_configured}
                </div>
                """, unsafe_allow_html=True)
        
        # System configuration
        st.subheader("⚙️ Configuration")
        
        with st.expander("📋 View Configuration"):
            st.json(config)
        
        # Data refresh controls
        st.subheader("🔄 Data Management")
        
        mgmt_cols = st.columns(3)
        
        with mgmt_cols[0]:
            if st.button("🔄 Refresh Real-time Data"):
                st.cache_data.clear()
                st.success("✅ Cache cleared")
        
        with mgmt_cols[1]:
            if st.button("📥 Migrate CSV Data"):
                db_manager = DatabaseManager()
                db_manager.migrate_existing_data()
                st.success("✅ Data migrated")
        
        with mgmt_cols[2]:
            if st.button("🧹 Clean Old Data"):
                st.info("🔧 Feature coming soon")

if __name__ == "__main__":
    main()