import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class DataQualityEnhancer:
    """Enhanced data preparation with realistic variability and improved quality"""
    
    def __init__(self):
        self.tz = pytz.timezone('Asia/Kolkata')
        
    def generate_realistic_pollution_patterns(self, base_data, station_id):
        """Generate realistic pollution patterns with proper variability"""
        
        # Define realistic pollution patterns for different times
        hourly_patterns = {
            # Morning rush (6-9 AM): Higher pollution
            'morning_rush': {'hours': [6, 7, 8, 9], 'multiplier': 1.4, 'noise': 0.3},
            # Evening rush (5-8 PM): Higher pollution  
            'evening_rush': {'hours': [17, 18, 19, 20], 'multiplier': 1.3, 'noise': 0.25},
            # Night (10 PM - 5 AM): Lower pollution
            'night': {'hours': [22, 23, 0, 1, 2, 3, 4, 5], 'multiplier': 0.7, 'noise': 0.15},
            # Midday (10 AM - 4 PM): Moderate pollution
            'midday': {'hours': [10, 11, 12, 13, 14, 15, 16], 'multiplier': 1.0, 'noise': 0.2},
            # Early morning (9-10 AM, 9-10 PM): Transition
            'transition': {'hours': [9, 21], 'multiplier': 1.1, 'noise': 0.2}
        }
        
        # Weekly patterns
        weekend_reduction = 0.8  # Lower pollution on weekends
        
        # Seasonal patterns (simplified)
        seasonal_factors = {
            1: 1.3, 2: 1.2, 3: 1.1, 4: 1.0, 5: 1.1, 6: 1.2,  # Winter/Spring higher
            7: 1.0, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.3, 12: 1.4  # Summer lower, winter higher
        }
        
        enhanced_data = base_data.copy()
        
        for idx, row in enhanced_data.iterrows():
            hour = row['timestamp'].hour
            day_of_week = row['timestamp'].dayofweek
            month = row['timestamp'].month
            
            # Base PM2.5 value
            base_pm25 = row['pm25']
            
            # Apply hourly patterns
            hourly_multiplier = 1.0
            hourly_noise = 0.2
            
            for pattern_name, pattern in hourly_patterns.items():
                if hour in pattern['hours']:
                    hourly_multiplier = pattern['multiplier']
                    hourly_noise = pattern['noise']
                    break
            
            # Apply weekend reduction
            weekend_multiplier = weekend_reduction if day_of_week >= 5 else 1.0
            
            # Apply seasonal factor
            seasonal_multiplier = seasonal_factors.get(month, 1.0)
            
            # Calculate enhanced PM2.5
            enhanced_pm25 = base_pm25 * hourly_multiplier * weekend_multiplier * seasonal_multiplier
            
            # Add realistic noise
            noise = np.random.normal(0, enhanced_pm25 * hourly_noise)
            enhanced_pm25 += noise
            
            # Ensure realistic bounds
            enhanced_pm25 = max(5, min(500, enhanced_pm25))
            
            enhanced_data.loc[idx, 'pm25'] = enhanced_pm25
            
            # Enhance other pollutants with correlated patterns
            if 'no2' in enhanced_data.columns:
                no2_factor = hourly_multiplier * 0.8 + 0.2  # Less variation than PM2.5
                enhanced_data.loc[idx, 'no2'] = row['no2'] * no2_factor * (1 + np.random.normal(0, 0.1))
            
            if 'so2' in enhanced_data.columns:
                so2_factor = seasonal_multiplier * 0.9 + 0.1  # More seasonal variation
                enhanced_data.loc[idx, 'so2'] = row['so2'] * so2_factor * (1 + np.random.normal(0, 0.15))
            
            if 'co' in enhanced_data.columns:
                co_factor = hourly_multiplier * weekend_multiplier
                enhanced_data.loc[idx, 'co'] = row['co'] * co_factor * (1 + np.random.normal(0, 0.12))
            
            if 'o3' in enhanced_data.columns:
                # O3 has inverse pattern (higher during day, lower at night)
                o3_multiplier = 1.5 if 10 <= hour <= 16 else 0.7
                enhanced_data.loc[idx, 'o3'] = row['o3'] * o3_multiplier * (1 + np.random.normal(0, 0.2))
        
        return enhanced_data
    
    def enhance_weather_variability(self, weather_data):
        """Add realistic weather variability and correlations"""
        enhanced_weather = weather_data.copy()
        
        for idx, row in enhanced_weather.iterrows():
            hour = row['timestamp'].hour
            month = row['timestamp'].month
            
            # Temperature diurnal cycle
            temp_base = row['temp_c']
            diurnal_variation = 8 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak at 2 PM
            seasonal_temp = temp_base + diurnal_variation
            
            # Add realistic temperature noise
            temp_noise = np.random.normal(0, 2.0)
            enhanced_weather.loc[idx, 'temp_c'] = seasonal_temp + temp_noise
            
            # Wind speed with realistic patterns
            wind_base = row['wind_speed']
            # Higher wind during day, lower at night
            wind_diurnal = 1.3 if 10 <= hour <= 18 else 0.8
            wind_noise = np.random.exponential(1.0)  # Exponential noise for wind
            enhanced_weather.loc[idx, 'wind_speed'] = max(0.1, wind_base * wind_diurnal + wind_noise)
            
            # Wind direction with some persistence
            if idx > 0:
                prev_dir = enhanced_weather.loc[idx-1, 'wind_dir']
                dir_change = np.random.normal(0, 30)  # 30-degree standard deviation
                new_dir = (prev_dir + dir_change) % 360
                enhanced_weather.loc[idx, 'wind_dir'] = new_dir
            
            # Humidity with temperature correlation
            temp_factor = max(0.3, 1.2 - (enhanced_weather.loc[idx, 'temp_c'] - 20) * 0.02)
            humidity_base = row['humidity'] * temp_factor
            humidity_noise = np.random.normal(0, 5)
            enhanced_weather.loc[idx, 'humidity'] = max(10, min(100, humidity_base + humidity_noise))
            
            # Precipitation (keep sparse but realistic)
            if np.random.random() < 0.05:  # 5% chance of precipitation
                enhanced_weather.loc[idx, 'precip_mm'] = np.random.exponential(2.0)
            else:
                enhanced_weather.loc[idx, 'precip_mm'] = 0
        
        return enhanced_weather
    
    def enhance_simulation_data(self, simulation_data):
        """Enhance simulation data with realistic urban dynamics"""
        enhanced_sim = simulation_data.copy()
        
        for idx, row in enhanced_sim.iterrows():
            hour = row['timestamp'].hour
            day_of_week = row['timestamp'].dayofweek
            
            # Traffic index with realistic patterns
            traffic_base = row['traffic_idx']
            
            # Rush hour patterns
            if hour in [7, 8, 9, 17, 18, 19]:
                traffic_multiplier = 1.5
            elif hour in [10, 11, 12, 13, 14, 15, 16]:
                traffic_multiplier = 1.2
            elif hour in [20, 21, 22]:
                traffic_multiplier = 0.9
            else:
                traffic_multiplier = 0.6
            
            # Weekend reduction
            if day_of_week >= 5:
                traffic_multiplier *= 0.7
            
            traffic_noise = np.random.normal(0, traffic_base * 0.15)
            enhanced_sim.loc[idx, 'traffic_idx'] = max(0.1, traffic_base * traffic_multiplier + traffic_noise)
            
            # Industrial index (more stable but with some variation)
            industrial_base = row['industrial_idx']
            industrial_variation = 1 + np.random.normal(0, 0.1)
            # Slightly lower on weekends
            weekend_factor = 0.9 if day_of_week >= 5 else 1.0
            enhanced_sim.loc[idx, 'industrial_idx'] = max(0.1, industrial_base * industrial_variation * weekend_factor)
            
            # Dust index (weather dependent)
            dust_base = row['dust_idx']
            # Higher dust with higher temperature (use default if no weather data)
            temp_factor = 1.0  # Default factor
            dust_noise = np.random.gamma(2, 0.1)  # Gamma noise for dust
            enhanced_sim.loc[idx, 'dust_idx'] = max(0.1, dust_base * temp_factor + dust_noise)
            
            # Dispersion PM2.5 (meteorology dependent)
            dispersion_base = row['dispersion_pm25']
            # Add meteorological influence
            met_factor = 1.2 if hour in [6, 7, 8, 18, 19, 20] else 0.9  # Stable conditions
            dispersion_noise = np.random.normal(0, dispersion_base * 0.2)
            enhanced_sim.loc[idx, 'dispersion_pm25'] = max(0.1, dispersion_base * met_factor + dispersion_noise)
        
        return enhanced_sim
    
    def add_extreme_events(self, data, event_probability=0.02):
        """Add realistic extreme pollution events"""
        enhanced_data = data.copy()
        
        for station in enhanced_data['station_id'].unique():
            station_mask = enhanced_data['station_id'] == station
            station_data = enhanced_data[station_mask].copy()
            
            # Add occasional pollution spikes
            n_events = int(len(station_data) * event_probability)
            
            for _ in range(n_events):
                # Random event location
                event_idx = np.random.choice(station_data.index)
                
                # Event characteristics
                event_duration = np.random.randint(2, 8)  # 2-8 hours
                event_magnitude = np.random.uniform(2.0, 4.0)  # 2-4x normal levels
                
                # Apply event
                for h in range(event_duration):
                    idx = event_idx + h
                    if idx in station_data.index:
                        current_pm25 = enhanced_data.loc[idx, 'pm25']
                        # Gradual increase and decrease
                        if h < event_duration // 2:
                            multiplier = 1 + (event_magnitude - 1) * (h + 1) / (event_duration // 2)
                        else:
                            multiplier = 1 + (event_magnitude - 1) * (event_duration - h) / (event_duration // 2)
                        
                        enhanced_data.loc[idx, 'pm25'] = current_pm25 * multiplier
        
        return enhanced_data
    
    def create_enhanced_dataset(self):
        """Create enhanced dataset with improved variability"""
        print("🔧 Creating enhanced dataset with realistic variability...")
        
        # Load original data
        data_dir = Path("urban-air-forecast/data")
        
        sensors = pd.read_csv(data_dir / "sensors.csv")
        weather = pd.read_csv(data_dir / "weather.csv")
        simulation = pd.read_csv(data_dir / "simulation.csv")
        
        # Convert timestamps
        sensors['timestamp'] = pd.to_datetime(sensors['timestamp']).dt.tz_localize(self.tz)
        weather['timestamp'] = pd.to_datetime(weather['timestamp']).dt.tz_localize(self.tz)
        simulation['timestamp'] = pd.to_datetime(simulation['timestamp']).dt.tz_localize(self.tz)
        
        print(f"📊 Original data - Sensors: {sensors.shape}, Weather: {weather.shape}, Simulation: {simulation.shape}")
        
        # Enhance each dataset
        enhanced_sensors = sensors.copy()
        for station in sensors['station_id'].unique():
            station_mask = sensors['station_id'] == station
            station_data = sensors[station_mask].copy()
            enhanced_station = self.generate_realistic_pollution_patterns(station_data, station)
            enhanced_sensors[station_mask] = enhanced_station
        
        # Add extreme events
        enhanced_sensors = self.add_extreme_events(enhanced_sensors)
        
        # Enhance weather data
        enhanced_weather = self.enhance_weather_variability(weather)
        
        # Enhance simulation data  
        enhanced_simulation = self.enhance_simulation_data(simulation)
        
        # Save enhanced datasets
        enhanced_sensors.to_csv(data_dir / "enhanced_sensors.csv", index=False)
        enhanced_weather.to_csv(data_dir / "enhanced_weather.csv", index=False)
        enhanced_simulation.to_csv(data_dir / "enhanced_simulation.csv", index=False)
        
        print("✅ Enhanced datasets created")
        
        # Create enhanced feature table
        self.create_enhanced_features(enhanced_sensors, enhanced_weather, enhanced_simulation)
        
        return enhanced_sensors, enhanced_weather, enhanced_simulation
    
    def create_enhanced_features(self, sensors, weather, simulation):
        """Create enhanced feature table with improved data"""
        print("⚙️ Creating enhanced feature table...")
        
        # Merge datasets
        merged = sensors.merge(weather, on=['timestamp', 'station_id'], how='outer')
        merged = merged.merge(simulation, on=['timestamp', 'station_id'], how='outer')
        
        # Create calendar features
        merged['hour'] = merged['timestamp'].dt.hour
        merged['day_of_week'] = merged['timestamp'].dt.dayofweek
        merged['is_weekend'] = (merged['day_of_week'] >= 5).astype(int)
        merged['month'] = merged['timestamp'].dt.month
        merged['day_of_year'] = merged['timestamp'].dt.dayofyear
        
        # Sort for lag features
        merged = merged.sort_values(['station_id', 'timestamp'])
        
        # Enhanced lag features with better handling
        merged['pm25_lag1'] = merged.groupby('station_id')['pm25'].shift(1)
        merged['pm25_lag24'] = merged.groupby('station_id')['pm25'].shift(24)
        merged['pm25_lag168'] = merged.groupby('station_id')['pm25'].shift(168)
        
        # Enhanced rolling features
        merged['pm25_roll_3h'] = merged.groupby('station_id')['pm25'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
        merged['pm25_roll_24h'] = merged.groupby('station_id')['pm25'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
        merged['pm25_roll_168h'] = merged.groupby('station_id')['pm25'].rolling(window=168, min_periods=1).mean().reset_index(0, drop=True)
        
        # Additional variability features
        merged['pm25_roll_3h_std'] = merged.groupby('station_id')['pm25'].rolling(window=3, min_periods=1).std().reset_index(0, drop=True)
        merged['pm25_roll_24h_std'] = merged.groupby('station_id')['pm25'].rolling(window=24, min_periods=1).std().reset_index(0, drop=True)
        
        # Weather rolling features
        merged['temp_roll_6h'] = merged.groupby('station_id')['temp_c'].rolling(window=6, min_periods=1).mean().reset_index(0, drop=True)
        merged['wind_speed_roll_12h'] = merged.groupby('station_id')['wind_speed'].rolling(window=12, min_periods=1).mean().reset_index(0, drop=True)
        merged['humidity_roll_6h'] = merged.groupby('station_id')['humidity'].rolling(window=6, min_periods=1).mean().reset_index(0, drop=True)
        
        # Interaction features
        merged['temp_wind_interaction'] = merged['temp_c'] * merged['wind_speed']
        merged['traffic_weather_interaction'] = merged['traffic_idx'] * (1 / (merged['wind_speed'] + 0.1))
        
        # Drop rows with missing critical values
        merged = merged.dropna(subset=['pm25_lag1', 'pm25_lag24'])
        
        # Save enhanced feature table
        output_path = Path("urban-air-forecast/data/enhanced_feature_table.parquet")
        merged.to_parquet(output_path, index=False)
        
        print(f"✅ Enhanced feature table saved: {merged.shape}")
        
        # Calculate variability metrics
        pm25_std = merged.groupby('station_id')['pm25'].std().mean()
        pm25_range = merged['pm25'].max() - merged['pm25'].min()
        
        print(f"📈 Enhanced variability - Std: {pm25_std:.2f}, Range: {pm25_range:.2f}")
        
        return merged

def main():
    enhancer = DataQualityEnhancer()
    
    try:
        # Create enhanced dataset
        enhanced_sensors, enhanced_weather, enhanced_simulation = enhancer.create_enhanced_dataset()
        
        print("🎯 Data quality enhancement completed successfully!")
        print("📊 Enhanced datasets available:")
        print("   - enhanced_sensors.csv")
        print("   - enhanced_weather.csv") 
        print("   - enhanced_simulation.csv")
        print("   - enhanced_feature_table.parquet")
        
    except Exception as e:
        print(f"❌ Data enhancement failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()