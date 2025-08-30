import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import sqlite3
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import requests
import time

class RealTimeDataIntegrator:
    """Real-time data integration for urban air quality forecasting"""
    
    def __init__(self, config_path="urban-air-forecast/config/realtime_config.json"):
        self.config = self.load_config(config_path)
        self.logger = self.setup_logging()
        self.influx_client = None
        self.setup_databases()
        
    def load_config(self, config_path):
        """Load real-time data source configurations"""
        default_config = {
            "data_sources": {
                "openweather": {
                    "api_key": "YOUR_OPENWEATHER_API_KEY",
                    "base_url": "https://api.openweathermap.org/data/2.5",
                    "enabled": True
                },
                "aqicn": {
                    "api_key": "YOUR_AQICN_API_KEY", 
                    "base_url": "https://api.waqi.info",
                    "enabled": True
                },
                "government_apis": {
                    "cpcb_india": "https://api.cpcb.nic.in/aqi",
                    "enabled": False
                }
            },
            "databases": {
                "influxdb": {
                    "url": "http://localhost:8086",
                    "token": "YOUR_INFLUX_TOKEN",
                    "org": "air-quality",
                    "bucket": "sensor-data"
                },
                "sqlite": {
                    "path": "urban-air-forecast/data/realtime.db"
                }
            },
            "stations": [
                {"id": "ST001", "name": "Delhi Central", "lat": 28.6139, "lon": 77.2090},
                {"id": "ST002", "name": "Mumbai Central", "lat": 19.0760, "lon": 72.8777},
                {"id": "ST003", "name": "Bangalore Central", "lat": 12.9716, "lon": 77.5946}
            ],
            "update_intervals": {
                "air_quality": 3600,  # 1 hour
                "weather": 1800,      # 30 minutes
                "forecasts": 21600    # 6 hours
            }
        }
        
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def setup_logging(self):
        """Setup logging for real-time data integration"""
        log_dir = Path("urban-air-forecast/logs")
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger('realtime_data')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_dir / "realtime_data.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def setup_databases(self):
        """Setup database connections"""
        # Setup InfluxDB
        if self.config["databases"]["influxdb"]["token"] != "YOUR_INFLUX_TOKEN":
            try:
                self.influx_client = InfluxDBClient(
                    url=self.config["databases"]["influxdb"]["url"],
                    token=self.config["databases"]["influxdb"]["token"],
                    org=self.config["databases"]["influxdb"]["org"]
                )
                self.logger.info("InfluxDB connection established")
            except Exception as e:
                self.logger.warning(f"InfluxDB connection failed: {e}")
        
        # Setup SQLite for metadata
        sqlite_path = Path(self.config["databases"]["sqlite"]["path"])
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(sqlite_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stations (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    latitude REAL,
                    longitude REAL,
                    last_updated TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_sources (
                    source_name TEXT,
                    station_id TEXT,
                    last_fetch TIMESTAMP,
                    status TEXT,
                    error_count INTEGER DEFAULT 0
                )
            """)
        
        self.logger.info("SQLite database initialized")
    
    async def fetch_openweather_data(self, station):
        """Fetch real-time weather data from OpenWeatherMap"""
        if not self.config["data_sources"]["openweather"]["enabled"]:
            return None
            
        api_key = self.config["data_sources"]["openweather"]["api_key"]
        if api_key == "YOUR_OPENWEATHER_API_KEY":
            self.logger.warning("OpenWeatherMap API key not configured")
            return None
        
        url = f"{self.config['data_sources']['openweather']['base_url']}/weather"
        params = {
            "lat": station["lat"],
            "lon": station["lon"],
            "appid": api_key,
            "units": "metric"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self.parse_openweather_data(data, station)
                    else:
                        self.logger.error(f"OpenWeather API error: {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Error fetching OpenWeather data: {e}")
            return None
    
    def parse_openweather_data(self, data, station):
        """Parse OpenWeatherMap API response"""
        return {
            "station_id": station["id"],
            "timestamp": datetime.now(),
            "source": "openweather",
            "temp_c": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"].get("speed", 0),
            "wind_dir": data["wind"].get("deg", 0),
            "visibility": data.get("visibility", 10000) / 1000,  # Convert to km
            "weather_condition": data["weather"][0]["main"].lower()
        }
    
    async def fetch_aqicn_data(self, station):
        """Fetch real-time air quality data from AQICN"""
        if not self.config["data_sources"]["aqicn"]["enabled"]:
            return None
            
        api_key = self.config["data_sources"]["aqicn"]["api_key"]
        if api_key == "YOUR_AQICN_API_KEY":
            self.logger.warning("AQICN API key not configured")
            return None
        
        # Use coordinates to find nearest station
        url = f"{self.config['data_sources']['aqicn']['base_url']}/feed/geo:{station['lat']};{station['lon']}/"
        params = {"token": api_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data["status"] == "ok":
                            return self.parse_aqicn_data(data["data"], station)
                    else:
                        self.logger.error(f"AQICN API error: {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Error fetching AQICN data: {e}")
            return None
    
    def parse_aqicn_data(self, data, station):
        """Parse AQICN API response"""
        iaqi = data.get("iaqi", {})
        
        return {
            "station_id": station["id"],
            "timestamp": datetime.now(),
            "source": "aqicn",
            "aqi": data.get("aqi", None),
            "pm25": iaqi.get("pm25", {}).get("v", None),
            "pm10": iaqi.get("pm10", {}).get("v", None),
            "no2": iaqi.get("no2", {}).get("v", None),
            "so2": iaqi.get("so2", {}).get("v", None),
            "co": iaqi.get("co", {}).get("v", None),
            "o3": iaqi.get("o3", {}).get("v", None),
            "station_name": data.get("city", {}).get("name", "Unknown")
        }
    
    def store_data_influxdb(self, data_points):
        """Store data in InfluxDB"""
        if not self.influx_client:
            return False
        
        try:
            write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            points = []
            
            for data in data_points:
                point = Point("air_quality_data") \
                    .tag("station_id", data["station_id"]) \
                    .tag("source", data["source"]) \
                    .time(data["timestamp"])
                
                # Add fields dynamically
                for key, value in data.items():
                    if key not in ["station_id", "timestamp", "source"] and value is not None:
                        if isinstance(value, (int, float)):
                            point = point.field(key, float(value))
                        else:
                            point = point.field(key, str(value))
                
                points.append(point)
            
            write_api.write(
                bucket=self.config["databases"]["influxdb"]["bucket"],
                record=points
            )
            
            self.logger.info(f"Stored {len(points)} data points in InfluxDB")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing data in InfluxDB: {e}")
            return False
    
    def store_data_sqlite(self, data_points):
        """Store data in SQLite as backup"""
        sqlite_path = Path(self.config["databases"]["sqlite"]["path"])
        
        try:
            with sqlite3.connect(sqlite_path) as conn:
                # Create table if not exists
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS realtime_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        station_id TEXT,
                        timestamp TIMESTAMP,
                        source TEXT,
                        data_json TEXT
                    )
                """)
                
                # Insert data
                for data in data_points:
                    conn.execute("""
                        INSERT INTO realtime_data (station_id, timestamp, source, data_json)
                        VALUES (?, ?, ?, ?)
                    """, (
                        data["station_id"],
                        data["timestamp"],
                        data["source"],
                        json.dumps(data)
                    ))
                
                conn.commit()
                self.logger.info(f"Stored {len(data_points)} data points in SQLite")
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing data in SQLite: {e}")
            return False
    
    async def collect_realtime_data(self):
        """Collect real-time data from all sources"""
        all_data = []
        
        for station in self.config["stations"]:
            self.logger.info(f"Collecting data for station {station['id']}")
            
            # Collect weather data
            weather_data = await self.fetch_openweather_data(station)
            if weather_data:
                all_data.append(weather_data)
            
            # Collect air quality data
            aqi_data = await self.fetch_aqicn_data(station)
            if aqi_data:
                all_data.append(aqi_data)
            
            # Small delay between stations
            await asyncio.sleep(1)
        
        return all_data
    
    def get_latest_data(self, station_id=None, hours_back=24):
        """Retrieve latest data from database"""
        if self.influx_client:
            return self.get_data_from_influxdb(station_id, hours_back)
        else:
            return self.get_data_from_sqlite(station_id, hours_back)
    
    def get_data_from_influxdb(self, station_id=None, hours_back=24):
        """Get data from InfluxDB"""
        try:
            query_api = self.influx_client.query_api()
            
            station_filter = f'|> filter(fn: (r) => r.station_id == "{station_id}")' if station_id else ""
            
            query = f'''
                from(bucket: "{self.config["databases"]["influxdb"]["bucket"]}")
                |> range(start: -{hours_back}h)
                |> filter(fn: (r) => r._measurement == "air_quality_data")
                {station_filter}
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            result = query_api.query_data_frame(query)
            return result
            
        except Exception as e:
            self.logger.error(f"Error querying InfluxDB: {e}")
            return pd.DataFrame()
    
    def get_data_from_sqlite(self, station_id=None, hours_back=24):
        """Get data from SQLite"""
        sqlite_path = Path(self.config["databases"]["sqlite"]["path"])
        
        try:
            with sqlite3.connect(sqlite_path) as conn:
                where_clause = f"AND station_id = '{station_id}'" if station_id else ""
                
                query = f"""
                    SELECT * FROM realtime_data 
                    WHERE timestamp >= datetime('now', '-{hours_back} hours')
                    {where_clause}
                    ORDER BY timestamp DESC
                """
                
                df = pd.read_sql_query(query, conn)
                
                # Parse JSON data
                if not df.empty:
                    parsed_data = []
                    for _, row in df.iterrows():
                        data = json.loads(row['data_json'])
                        parsed_data.append(data)
                    
                    return pd.DataFrame(parsed_data)
                
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error querying SQLite: {e}")
            return pd.DataFrame()
    
    async def run_continuous_collection(self):
        """Run continuous data collection"""
        self.logger.info("Starting continuous data collection")
        
        while True:
            try:
                # Collect data
                data_points = await self.collect_realtime_data()
                
                if data_points:
                    # Store in primary database
                    if self.influx_client:
                        self.store_data_influxdb(data_points)
                    
                    # Store in backup database
                    self.store_data_sqlite(data_points)
                    
                    self.logger.info(f"Collected and stored {len(data_points)} data points")
                else:
                    self.logger.warning("No data collected in this cycle")
                
                # Wait for next collection cycle
                await asyncio.sleep(self.config["update_intervals"]["air_quality"])
                
            except Exception as e:
                self.logger.error(f"Error in continuous collection: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

def main():
    """Main function to run real-time data integration"""
    integrator = RealTimeDataIntegrator()
    
    print("🌍 Real-Time Air Quality Data Integration")
    print("=" * 50)
    print("1. Collecting sample data...")
    
    # Run one-time collection
    loop = asyncio.get_event_loop()
    data = loop.run_until_complete(integrator.collect_realtime_data())
    
    if data:
        print(f"✅ Collected {len(data)} data points")
        for point in data:
            print(f"   📊 {point['station_id']} - {point['source']}: {point.get('aqi', 'N/A')} AQI")
    else:
        print("⚠️  No data collected - check API keys in config")
    
    print("\n2. Starting continuous collection...")
    print("   Press Ctrl+C to stop")
    
    try:
        loop.run_until_complete(integrator.run_continuous_collection())
    except KeyboardInterrupt:
        print("\n👋 Data collection stopped")

if __name__ == "__main__":
    main()