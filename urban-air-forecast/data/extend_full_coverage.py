import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

np.random.seed(42)

def extend_full_coverage():
    # Load current data
    features = pd.read_parquet('urban-air-forecast/data/feature_table.parquet')
    weather = pd.read_csv('urban-air-forecast/data/weather.csv')
    simulation = pd.read_csv('urban-air-forecast/data/simulation.csv')
    
    # Convert timestamps
    tz = pytz.timezone('Asia/Kolkata')
    features['timestamp'] = pd.to_datetime(features['timestamp']).dt.tz_convert(tz)
    weather['timestamp'] = pd.to_datetime(weather['timestamp']).dt.tz_localize(tz)
    simulation['timestamp'] = pd.to_datetime(simulation['timestamp']).dt.tz_localize(tz)
    
    # Find the latest feature time
    latest_feature_time = features['timestamp'].max()
    print(f"Latest feature time: {latest_feature_time}")
    
    # Extend weather and simulation to cover 72h beyond latest features
    target_end_time = latest_feature_time + timedelta(hours=72)
    print(f"Target coverage until: {target_end_time}")
    
    # Extend weather data
    extended_weather = []
    for station in weather['station_id'].unique():
        station_weather = weather[weather['station_id'] == station].copy()
        last_time = station_weather['timestamp'].max()
        
        # Get last 24 hours as pattern
        pattern_data = station_weather.tail(24).copy()
        
        current_time = last_time
        while current_time < target_end_time:
            current_time += timedelta(hours=1)
            
            # Use cyclical pattern with variation
            pattern_idx = (current_time.hour) % 24
            base_row = pattern_data.iloc[pattern_idx].copy()
            
            # Add realistic variation
            new_row = base_row.copy()
            new_row['timestamp'] = current_time
            new_row['temp_c'] += np.random.normal(0, 2)
            new_row['wind_speed'] = max(0, new_row['wind_speed'] + np.random.normal(0, 1))
            new_row['humidity'] = max(0, min(100, new_row['humidity'] + np.random.normal(0, 5)))
            new_row['precip_mm'] = max(0, np.random.exponential(0.1) if np.random.random() < 0.15 else 0)
            
            extended_weather.append(new_row)
    
    # Extend simulation data
    extended_simulation = []
    for station in simulation['station_id'].unique():
        station_simulation = simulation[simulation['station_id'] == station].copy()
        last_time = station_simulation['timestamp'].max()
        
        # Get last 24 hours as pattern
        pattern_data = station_simulation.tail(24).copy()
        
        current_time = last_time
        while current_time < target_end_time:
            current_time += timedelta(hours=1)
            
            # Use cyclical pattern with variation
            pattern_idx = (current_time.hour) % 24
            base_row = pattern_data.iloc[pattern_idx].copy()
            
            # Add realistic variation
            new_row = base_row.copy()
            new_row['timestamp'] = current_time
            new_row['traffic_idx'] = max(0, min(1, new_row['traffic_idx'] + np.random.normal(0, 0.1)))
            new_row['industrial_idx'] = max(0, min(1, new_row['industrial_idx'] + np.random.normal(0, 0.08)))
            new_row['dust_idx'] = max(0, min(1, new_row['dust_idx'] + np.random.normal(0, 0.05)))
            new_row['dispersion_pm25'] = max(0, new_row['dispersion_pm25'] + np.random.normal(0, 3))
            
            extended_simulation.append(new_row)
    
    # Combine and save
    if extended_weather:
        extended_weather_df = pd.concat([weather, pd.DataFrame(extended_weather)], ignore_index=True)
        extended_weather_df.to_csv('urban-air-forecast/data/weather.csv', index=False)
        print(f"Extended weather to: {extended_weather_df['timestamp'].max()}")
    
    if extended_simulation:
        extended_simulation_df = pd.concat([simulation, pd.DataFrame(extended_simulation)], ignore_index=True)
        extended_simulation_df.to_csv('urban-air-forecast/data/simulation.csv', index=False)
        print(f"Extended simulation to: {extended_simulation_df['timestamp'].max()}")
    
    print(f"Added {len(extended_weather)} weather records and {len(extended_simulation)} simulation records")

if __name__ == "__main__":
    extend_full_coverage()