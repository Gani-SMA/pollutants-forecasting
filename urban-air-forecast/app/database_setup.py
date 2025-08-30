import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging

class DatabaseManager:
    """Database management for air quality forecasting system"""
    
    def __init__(self, db_path="urban-air-forecast/data/air_quality.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.setup_database()
        
    def setup_database(self):
        """Create database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Stations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    city TEXT,
                    state TEXT,
                    country TEXT,
                    latitude REAL,
                    longitude REAL,
                    elevation REAL,
                    station_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Sensor data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sensor_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_id TEXT,
                    timestamp TIMESTAMP,
                    pm25 REAL,
                    pm10 REAL,
                    no2 REAL,
                    so2 REAL,
                    co REAL,
                    o3 REAL,
                    aqi INTEGER,
                    data_source TEXT,
                    quality_flag TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (station_id) REFERENCES stations (id)
                )
            """)
            
            # Weather data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS weather_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_id TEXT,
                    timestamp TIMESTAMP,
                    temp_c REAL,
                    humidity REAL,
                    pressure REAL,
                    wind_speed REAL,
                    wind_dir REAL,
                    precip_mm REAL,
                    visibility REAL,
                    weather_condition TEXT,
                    data_source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (station_id) REFERENCES stations (id)
                )
            """)
            
            # Forecasts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_id TEXT,
                    issue_time TIMESTAMP,
                    target_time TIMESTAMP,
                    horizon_hours INTEGER,
                    pm25_forecast REAL,
                    pm25_lower_ci REAL,
                    pm25_upper_ci REAL,
                    uncertainty REAL,
                    quality_flag TEXT,
                    model_version TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (station_id) REFERENCES stations (id)
                )
            """)
            
            # Model metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    model_version TEXT,
                    model_hash TEXT,
                    training_data_start TIMESTAMP,
                    training_data_end TIMESTAMP,
                    feature_count INTEGER,
                    performance_metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Policy recommendations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS policy_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_id TEXT,
                    timestamp TIMESTAMP,
                    aqi_value INTEGER,
                    aqi_category TEXT,
                    urgency_level TEXT,
                    recommendations TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (station_id) REFERENCES stations (id)
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sensor_data_station_time ON sensor_data(station_id, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_weather_data_station_time ON weather_data(station_id, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_forecasts_station_time ON forecasts(station_id, issue_time)")
            
            conn.commit()
            print("✅ Database tables created successfully")
    
    def insert_station(self, station_data):
        """Insert or update station information"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO stations 
                (id, name, city, state, country, latitude, longitude, elevation, station_type, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                station_data['id'],
                station_data['name'],
                station_data.get('city'),
                station_data.get('state'),
                station_data.get('country'),
                station_data.get('latitude'),
                station_data.get('longitude'),
                station_data.get('elevation'),
                station_data.get('station_type', 'monitoring'),
                datetime.now()
            ))
            conn.commit()
    
    def insert_sensor_data(self, sensor_data_list):
        """Insert sensor data in batch"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT INTO sensor_data 
                (station_id, timestamp, pm25, pm10, no2, so2, co, o3, aqi, data_source, quality_flag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    data['station_id'],
                    data['timestamp'],
                    data.get('pm25'),
                    data.get('pm10'),
                    data.get('no2'),
                    data.get('so2'),
                    data.get('co'),
                    data.get('o3'),
                    data.get('aqi'),
                    data.get('data_source', 'unknown'),
                    data.get('quality_flag', 'ok')
                ) for data in sensor_data_list
            ])
            conn.commit()
    
    def insert_weather_data(self, weather_data_list):
        """Insert weather data in batch"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT INTO weather_data 
                (station_id, timestamp, temp_c, humidity, pressure, wind_speed, wind_dir, 
                 precip_mm, visibility, weather_condition, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    data['station_id'],
                    data['timestamp'],
                    data.get('temp_c'),
                    data.get('humidity'),
                    data.get('pressure'),
                    data.get('wind_speed'),
                    data.get('wind_dir'),
                    data.get('precip_mm'),
                    data.get('visibility'),
                    data.get('weather_condition'),
                    data.get('data_source', 'unknown')
                ) for data in weather_data_list
            ])
            conn.commit()
    
    def insert_forecasts(self, forecast_data_list):
        """Insert forecast data in batch"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT INTO forecasts 
                (station_id, issue_time, target_time, horizon_hours, pm25_forecast, 
                 pm25_lower_ci, pm25_upper_ci, uncertainty, quality_flag, model_version, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    data['station_id'],
                    data['issue_time'],
                    data['target_time'],
                    data['horizon_hours'],
                    data['pm25_forecast'],
                    data.get('pm25_lower_ci'),
                    data.get('pm25_upper_ci'),
                    data.get('uncertainty'),
                    data.get('quality_flag', 'ok'),
                    data.get('model_version', 'unknown'),
                    data.get('confidence')
                ) for data in forecast_data_list
            ])
            conn.commit()
    
    def get_latest_sensor_data(self, station_id=None, hours_back=24):
        """Get latest sensor data"""
        with sqlite3.connect(self.db_path) as conn:
            where_clause = f"AND station_id = '{station_id}'" if station_id else ""
            
            query = f"""
                SELECT * FROM sensor_data 
                WHERE timestamp >= datetime('now', '-{hours_back} hours')
                {where_clause}
                ORDER BY timestamp DESC
            """
            
            return pd.read_sql_query(query, conn)
    
    def get_latest_weather_data(self, station_id=None, hours_back=24):
        """Get latest weather data"""
        with sqlite3.connect(self.db_path) as conn:
            where_clause = f"AND station_id = '{station_id}'" if station_id else ""
            
            query = f"""
                SELECT * FROM weather_data 
                WHERE timestamp >= datetime('now', '-{hours_back} hours')
                {where_clause}
                ORDER BY timestamp DESC
            """
            
            return pd.read_sql_query(query, conn)
    
    def get_stations(self):
        """Get all stations"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM stations", conn)
    
    def migrate_existing_data(self):
        """Migrate existing CSV data to database"""
        print("🔄 Migrating existing data to database...")
        
        data_dir = Path("urban-air-forecast/data")
        
        # Migrate stations
        stations_data = [
            {"id": "ST001", "name": "Delhi Central", "city": "Delhi", "state": "Delhi", 
             "country": "India", "latitude": 28.6139, "longitude": 77.2090},
            {"id": "ST002", "name": "Mumbai Central", "city": "Mumbai", "state": "Maharashtra", 
             "country": "India", "latitude": 19.0760, "longitude": 72.8777},
            {"id": "ST003", "name": "Bangalore Central", "city": "Bangalore", "state": "Karnataka", 
             "country": "India", "latitude": 12.9716, "longitude": 77.5946}
        ]
        
        for station in stations_data:
            self.insert_station(station)
        
        # Migrate sensor data if exists
        if (data_dir / "sensors.csv").exists():
            sensors_df = pd.read_csv(data_dir / "sensors.csv")
            sensors_df['timestamp'] = pd.to_datetime(sensors_df['timestamp'])
            
            sensor_data_list = []
            for _, row in sensors_df.iterrows():
                sensor_data_list.append({
                    'station_id': row['station_id'],
                    'timestamp': row['timestamp'],
                    'pm25': row.get('pm25'),
                    'no2': row.get('no2'),
                    'so2': row.get('so2'),
                    'co': row.get('co'),
                    'o3': row.get('o3'),
                    'data_source': 'csv_migration'
                })
            
            self.insert_sensor_data(sensor_data_list)
            print(f"✅ Migrated {len(sensor_data_list)} sensor records")
        
        # Migrate weather data if exists
        if (data_dir / "weather.csv").exists():
            weather_df = pd.read_csv(data_dir / "weather.csv")
            weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
            
            weather_data_list = []
            for _, row in weather_df.iterrows():
                weather_data_list.append({
                    'station_id': row['station_id'],
                    'timestamp': row['timestamp'],
                    'temp_c': row.get('temp_c'),
                    'humidity': row.get('humidity'),
                    'wind_speed': row.get('wind_speed'),
                    'wind_dir': row.get('wind_dir'),
                    'precip_mm': row.get('precip_mm'),
                    'data_source': 'csv_migration'
                })
            
            self.insert_weather_data(weather_data_list)
            print(f"✅ Migrated {len(weather_data_list)} weather records")
        
        print("✅ Data migration completed")
    
    def get_database_stats(self):
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Count records in each table
            tables = ['stations', 'sensor_data', 'weather_data', 'forecasts', 'policy_recommendations']
            
            for table in tables:
                result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                stats[table] = result[0]
            
            # Get date ranges
            sensor_range = conn.execute("""
                SELECT MIN(timestamp), MAX(timestamp) FROM sensor_data
            """).fetchone()
            
            weather_range = conn.execute("""
                SELECT MIN(timestamp), MAX(timestamp) FROM weather_data
            """).fetchone()
            
            stats['data_ranges'] = {
                'sensor_data': sensor_range,
                'weather_data': weather_range
            }
            
            return stats

def main():
    """Setup and test database"""
    print("🗄️ Setting up Air Quality Database")
    print("=" * 40)
    
    # Initialize database
    db_manager = DatabaseManager()
    
    # Migrate existing data
    db_manager.migrate_existing_data()
    
    # Show statistics
    stats = db_manager.get_database_stats()
    
    print("\n📊 Database Statistics:")
    for table, count in stats.items():
        if table != 'data_ranges':
            print(f"   {table}: {count} records")
    
    if stats['data_ranges']['sensor_data'][0]:
        print(f"\n📅 Data Coverage:")
        print(f"   Sensor data: {stats['data_ranges']['sensor_data'][0]} to {stats['data_ranges']['sensor_data'][1]}")
        print(f"   Weather data: {stats['data_ranges']['weather_data'][0]} to {stats['data_ranges']['weather_data'][1]}")
    
    print(f"\n💾 Database location: {db_manager.db_path}")
    print("✅ Database setup completed!")

if __name__ == "__main__":
    main()