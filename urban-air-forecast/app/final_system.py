import pandas as pd
import numpy as np
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class FinalSystem:
    """Final comprehensive forecasting system addressing all flaws permanently"""
    
    def __init__(self):
        self.logger = self.setup_logging()
        self.tz = pytz.timezone('Asia/Kolkata')
        self.models = {}
        self.feature_names = []
        
    def setup_logging(self):
        class ASCIIFormatter(logging.Formatter):
            def format(self, record):
                msg = super().format(record)
                replacements = {'μg/m³': 'ug/m3', '°C': 'degC', '±': '+/-', '→': '->'}
                for unicode_char, ascii_char in replacements.items():
                    msg = msg.replace(unicode_char, ascii_char)
                return msg
        
        log_dir = Path("urban-air-forecast/logs")
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger('final_system')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        file_handler = logging.FileHandler(log_dir / "final_system.log", encoding='utf-8')
        file_handler.setFormatter(ASCIIFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ASCIIFormatter('%(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        
        return logger
    
    def retrain_with_enhanced_data(self):
        """Retrain models using enhanced data"""
        self.logger.info("Retraining models with enhanced data")
        
        # Load enhanced feature table
        enhanced_data = pd.read_parquet("urban-air-forecast/data/enhanced_feature_table.parquet")
        self.logger.info(f"Enhanced data loaded: {enhanced_data.shape}")
        
        # Define expanded feature set
        feature_columns = [
            'no2', 'so2', 'co', 'o3',
            'temp_c', 'wind_speed', 'wind_dir', 'humidity', 'precip_mm',
            'traffic_idx', 'industrial_idx', 'dust_idx', 'dispersion_pm25',
            'hour', 'day_of_week', 'is_weekend', 'month', 'day_of_year',
            'pm25_lag1', 'pm25_lag24', 'pm25_lag168',
            'pm25_roll_3h', 'pm25_roll_24h', 'pm25_roll_168h',
            'pm25_roll_3h_std', 'pm25_roll_24h_std',
            'temp_roll_6h', 'wind_speed_roll_12h', 'humidity_roll_6h',
            'temp_wind_interaction', 'traffic_weather_interaction'
        ]
        
        # Filter available features
        available_features = [f for f in feature_columns if f in enhanced_data.columns]
        self.feature_names = available_features
        
        clean_data = enhanced_data.dropna(subset=available_features + ['pm25'])
        self.logger.info(f"Clean data for training: {clean_data.shape}")
        
        X = clean_data[available_features]
        y = clean_data['pm25']
        
        # Train enhanced ensemble
        models = {}
        
        # Enhanced LightGBM
        lgbm_model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
            force_row_wise=True
        )
        
        # Random Forest for variability
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42
        )
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
        
        # Train models
        for name, model in [('lgbm', lgbm_model), ('rf', rf_model), ('gb', gb_model)]:
            self.logger.info(f"Training {name} model...")
            model.fit(X, y)
            models[name] = model
        
        # Save models
        model_dir = Path("urban-air-forecast/models")
        model_dir.mkdir(exist_ok=True)
        
        for name, model in models.items():
            joblib.dump(model, model_dir / f"final_{name}_model.joblib")
        
        with open(model_dir / "final_feature_names.json", "w") as f:
            json.dump(self.feature_names, f)
        
        self.models = models
        self.logger.info("Enhanced models trained and saved")
        
        return models
    
    def generate_final_forecasts(self):
        """Generate final forecasts with all improvements"""
        self.logger.info("Generating final comprehensive forecasts")
        
        # Load enhanced data
        enhanced_data = pd.read_parquet("urban-air-forecast/data/enhanced_feature_table.parquet")
        
        # Retrain models
        self.retrain_with_enhanced_data()
        
        # Load enhanced driver data
        weather_df = pd.read_csv("urban-air-forecast/data/enhanced_weather.csv")
        simulation_df = pd.read_csv("urban-air-forecast/data/enhanced_simulation.csv")
        
        # Set issue time
        issue_time = enhanced_da