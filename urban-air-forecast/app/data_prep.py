import pandas as pd
import numpy as np
from pathlib import Path
import pytz
from scipy import stats

def detect_outliers(df, columns, method='iqr', threshold=3):
    """Detect outliers using IQR or Z-score method"""
    outlier_mask = pd.Series([False] * len(df), index=df.index)
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask |= (df[col] < lower_bound) | (df[col] > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_mask |= z_scores > threshold
    
    return outlier_mask

def load_and_prepare_data():
    data_dir = Path("urban-air-forecast/data")
    
    sensors = pd.read_csv(data_dir / "sensors.csv")
    weather = pd.read_csv(data_dir / "weather.csv")
    simulation = pd.read_csv(data_dir / "simulation.csv")
    
    print(f"📊 Raw data loaded - Sensors: {sensors.shape}, Weather: {weather.shape}, Simulation: {simulation.shape}")
    
    tz = pytz.timezone('Asia/Kolkata')
    sensors['timestamp'] = pd.to_datetime(sensors['timestamp'], format='mixed').dt.tz_localize(tz)
    weather['timestamp'] = pd.to_datetime(weather['timestamp'], format='mixed').dt.tz_localize(tz)
    simulation['timestamp'] = pd.to_datetime(simulation['timestamp'], format='mixed').dt.tz_localize(tz)
    
    # Handle duplicates and outliers
    sensors = sensors.drop_duplicates(subset=['timestamp', 'station_id'])
    weather = weather.drop_duplicates(subset=['timestamp', 'station_id'])
    simulation = simulation.drop_duplicates(subset=['timestamp', 'station_id'])
    
    # Detect and handle outliers
    sensor_cols = ['pm25', 'no2', 'so2', 'co', 'o3']
    weather_cols = ['temp_c', 'wind_speed', 'humidity']
    
    for station in sensors['station_id'].unique():
        station_mask = sensors['station_id'] == station
        outliers = detect_outliers(sensors[station_mask], sensor_cols)
        if outliers.sum() > 0:
            print(f"🔍 Found {outliers.sum()} outliers in sensors for station {station}")
            sensors.loc[sensors['station_id'] == station, sensor_cols] = sensors.loc[sensors['station_id'] == station, sensor_cols].mask(outliers)
    
    def resample_hourly(df):
        df = df.set_index('timestamp').groupby('station_id').resample('1H').mean()
        df = df.reset_index()
        return df
    
    sensors = resample_hourly(sensors)
    weather = resample_hourly(weather)
    simulation = resample_hourly(simulation)
    
    # Fill gaps with forward/backward fill
    sensors = sensors.groupby('station_id').apply(lambda x: x.fillna(method='ffill').fillna(method='bfill')).reset_index(drop=True)
    weather = weather.groupby('station_id').apply(lambda x: x.fillna(method='ffill').fillna(method='bfill')).reset_index(drop=True)
    simulation = simulation.groupby('station_id').apply(lambda x: x.fillna(method='ffill').fillna(method='bfill')).reset_index(drop=True)
    
    print("🔧 Data cleaned and resampled to hourly frequency")
    
    # Merge all dataframes
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
    
    # Create lag features
    merged['pm25_lag1'] = merged.groupby('station_id')['pm25'].shift(1)
    merged['pm25_lag24'] = merged.groupby('station_id')['pm25'].shift(24)
    merged['pm25_lag168'] = merged.groupby('station_id')['pm25'].shift(168)  # 1 week
    
    # Create rolling features
    merged['pm25_roll_3h'] = merged.groupby('station_id')['pm25'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    merged['pm25_roll_24h'] = merged.groupby('station_id')['pm25'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
    merged['temp_roll_6h'] = merged.groupby('station_id')['temp_c'].rolling(window=6, min_periods=1).mean().reset_index(0, drop=True)
    merged['wind_speed_roll_12h'] = merged.groupby('station_id')['wind_speed'].rolling(window=12, min_periods=1).mean().reset_index(0, drop=True)
    
    print("⚙️ Calendar and lag/rolling features engineered")
    
    # Drop rows with missing critical lag values
    merged = merged.dropna(subset=['pm25_lag1', 'pm25_lag24'])
    
    return merged

def main():
    Path("urban-air-forecast/data").mkdir(parents=True, exist_ok=True)
    
    feature_table = load_and_prepare_data()
    
    output_path = Path("urban-air-forecast/data/feature_table.parquet")
    feature_table.to_parquet(output_path, index=False)
    
    print(f"✅ Features saved: {feature_table.shape}")

if __name__ == "__main__":
    main()