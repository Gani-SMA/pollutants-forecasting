import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def extend_data():
    weather = pd.read_csv('urban-air-forecast/data/weather.csv')
    simulation = pd.read_csv('urban-air-forecast/data/simulation.csv')
    
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])
    simulation['timestamp'] = pd.to_datetime(simulation['timestamp'])
    
    last_weather_time = weather['timestamp'].max()
    last_simulation_time = simulation['timestamp'].max()
    
    # Extend for 72 hours ahead
    extended_weather = []
    extended_simulation = []
    
    for station in weather['station_id'].unique():
        station_weather = weather[weather['station_id'] == station].iloc[-24:].copy()
        station_simulation = simulation[simulation['station_id'] == station].iloc[-24:].copy()
        
        for h in range(1, 73):
            future_time = last_weather_time + timedelta(hours=h)
            
            # Use cyclical pattern from last 24 hours
            base_idx = (h - 1) % 24
            base_weather = station_weather.iloc[base_idx].copy()
            base_simulation = station_simulation.iloc[base_idx].copy()
            
            # Add some variation
            weather_row = base_weather.copy()
            weather_row['timestamp'] = future_time
            weather_row['temp_c'] += np.random.normal(0, 1)
            weather_row['wind_speed'] = max(0, weather_row['wind_speed'] + np.random.normal(0, 0.5))
            weather_row['humidity'] = max(0, min(100, weather_row['humidity'] + np.random.normal(0, 3)))
            weather_row['precip_mm'] = max(0, weather_row['precip_mm'] + np.random.exponential(0.05) if np.random.random() < 0.1 else 0)
            
            simulation_row = base_simulation.copy()
            simulation_row['timestamp'] = future_time
            simulation_row['traffic_idx'] = max(0, min(1, simulation_row['traffic_idx'] + np.random.normal(0, 0.05)))
            simulation_row['industrial_idx'] = max(0, min(1, simulation_row['industrial_idx'] + np.random.normal(0, 0.05)))
            simulation_row['dust_idx'] = max(0, min(1, simulation_row['dust_idx'] + np.random.normal(0, 0.02)))
            simulation_row['dispersion_pm25'] = max(0, simulation_row['dispersion_pm25'] + np.random.normal(0, 2))
            
            extended_weather.append(weather_row)
            extended_simulation.append(simulation_row)
    
    # Combine original and extended data
    extended_weather_df = pd.concat([weather, pd.DataFrame(extended_weather)], ignore_index=True)
    extended_simulation_df = pd.concat([simulation, pd.DataFrame(extended_simulation)], ignore_index=True)
    
    # Save extended data
    extended_weather_df.to_csv('urban-air-forecast/data/weather.csv', index=False)
    extended_simulation_df.to_csv('urban-air-forecast/data/simulation.csv', index=False)
    
    print(f"Extended weather data to: {extended_weather_df['timestamp'].max()}")
    print(f"Extended simulation data to: {extended_simulation_df['timestamp'].max()}")
    print(f"Added {len(extended_weather)} weather records and {len(extended_simulation)} simulation records")

if __name__ == "__main__":
    extend_data()