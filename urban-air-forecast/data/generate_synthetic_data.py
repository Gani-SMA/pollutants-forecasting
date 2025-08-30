import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

np.random.seed(42)

def generate_synthetic_data():
    # Generate 30 days of hourly data for 3 stations
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    stations = ['ST001', 'ST002', 'ST003']
    
    # Create hourly timestamps
    timestamps = []
    current = start_date
    while current < end_date:
        timestamps.append(current)
        current += timedelta(hours=1)
    
    all_data = []
    
    for station in stations:
        station_base = {'ST001': 50, 'ST002': 65, 'ST003': 45}[station]
        
        for i, ts in enumerate(timestamps):
            # Diurnal pattern
            hour_factor = 1 + 0.3 * np.sin(2 * np.pi * ts.hour / 24)
            
            # Weekly pattern
            day_factor = 1 + 0.1 * np.sin(2 * np.pi * ts.weekday() / 7)
            
            # Random noise
            noise = np.random.normal(0, 5)
            
            # Base PM2.5 with patterns
            base_pm25 = station_base * hour_factor * day_factor + noise
            
            # Weather correlations
            temp_c = 15 + 10 * np.sin(2 * np.pi * ts.hour / 24) + np.random.normal(0, 2)
            wind_speed = 3 + 2 * np.random.exponential(1)
            wind_dir = np.random.uniform(0, 360)
            humidity = 60 + 20 * np.sin(2 * np.pi * (ts.hour + 6) / 24) + np.random.normal(0, 5)
            precip_mm = max(0, np.random.exponential(0.1) if np.random.random() < 0.1 else 0)
            
            # Pollution correlations
            no2 = base_pm25 * 0.7 + np.random.normal(0, 3)
            so2 = base_pm25 * 0.3 + np.random.normal(0, 2)
            co = base_pm25 * 0.02 + np.random.normal(0, 0.2)
            o3 = max(0, 100 - base_pm25 * 0.5 + np.random.normal(0, 10))
            
            # Traffic and industrial patterns
            traffic_idx = 0.8 if 7 <= ts.hour <= 9 or 17 <= ts.hour <= 19 else 0.3
            traffic_idx += np.random.normal(0, 0.1)
            
            industrial_idx = 0.6 + 0.2 * np.sin(2 * np.pi * ts.hour / 24) + np.random.normal(0, 0.1)
            dust_idx = 0.2 + 0.1 * (wind_speed / 10) + np.random.normal(0, 0.05)
            dispersion_pm25 = base_pm25 * (1 + 0.1 * np.random.normal())
            
            all_data.append({
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'station_id': station,
                'pm25': max(0, base_pm25),
                'no2': max(0, no2),
                'so2': max(0, so2),
                'co': max(0, co),
                'o3': max(0, o3),
                'temp_c': temp_c,
                'wind_speed': max(0, wind_speed),
                'wind_dir': wind_dir,
                'humidity': max(0, min(100, humidity)),
                'precip_mm': precip_mm,
                'traffic_idx': max(0, min(1, traffic_idx)),
                'industrial_idx': max(0, min(1, industrial_idx)),
                'dust_idx': max(0, min(1, dust_idx)),
                'dispersion_pm25': max(0, dispersion_pm25)
            })
    
    return pd.DataFrame(all_data)

# Generate and split data
df = generate_synthetic_data()

# Split into separate files
sensors_df = df[['timestamp', 'station_id', 'pm25', 'no2', 'so2', 'co', 'o3']]
weather_df = df[['timestamp', 'station_id', 'temp_c', 'wind_speed', 'wind_dir', 'humidity', 'precip_mm']]
simulation_df = df[['timestamp', 'station_id', 'traffic_idx', 'industrial_idx', 'dust_idx', 'dispersion_pm25']]

sensors_df.to_csv('urban-air-forecast/data/sensors.csv', index=False)
weather_df.to_csv('urban-air-forecast/data/weather.csv', index=False)
simulation_df.to_csv('urban-air-forecast/data/simulation.csv', index=False)

print(f"Generated {len(df)} rows of synthetic data")
print(f"Sensors: {sensors_df.shape}")
print(f"Weather: {weather_df.shape}")
print(f"Simulation: {simulation_df.shape}")