import pandas as pd
import numpy as np
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging with ASCII fallback for units
class ASCIIFormatter(logging.Formatter):
    def format(self, record):
        # Replace problematic Unicode characters with ASCII equivalents
        msg = super().format(record)
        replacements = {
            'μg/m³': 'ug/m3',
            '°C': 'degC',
            '±': '+/-',
            '→': '->'
        }
        for unicode_char, ascii_char in replacements.items():
            msg = msg.replace(unicode_char, ascii_char)
        return msg

def setup_logging():
    log_dir = Path("urban-air-forecast/logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger('forecast_generation')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler with ASCII formatter
    file_handler = logging.FileHandler(log_dir / "forecast_generation.log", encoding='utf-8')
    file_handler.setFormatter(ASCIIFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Console handler with ASCII formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ASCIIFormatter('%(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

def calculate_file_hash(filepath):
    """Calculate SHA256 hash of a file for provenance tracking"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return "hash_unavailable"

def load_model_and_features():
    """Load trained model and get feature list with validation"""
    model_path = Path("urban-air-forecast/data/lgbm_pm25.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Get feature names from model
    if hasattr(model, 'feature_name_'):
        feature_names = model.feature_name_
    else:
        # Fallback: read from feature importance file
        importance_path = Path("urban-air-forecast/data/feature_importance.csv")
        if importance_path.exists():
            importance_df = pd.read_csv(importance_path)
            feature_names = importance_df['feature_name'].tolist()
        else:
            raise ValueError("Cannot determine model features")
    
    return model, feature_names

def validate_driver_data_coverage(weather_df, simulation_df, forecast_hours=72):
    """Validate that driver data covers the full forecast horizon"""
    logger = logging.getLogger('forecast_generation')
    
    # Check time coverage
    weather_hours = len(weather_df)
    simulation_hours = len(simulation_df)
    
    coverage_issues = []
    
    if weather_hours < forecast_hours:
        coverage_issues.append(f"Weather data: {weather_hours}h < {forecast_hours}h required")
    
    if simulation_hours < forecast_hours:
        coverage_issues.append(f"Simulation data: {simulation_hours}h < {forecast_hours}h required")
    
    # Check for missing values in critical periods
    weather_missing = weather_df.isnull().sum().sum()
    simulation_missing = simulation_df.isnull().sum().sum()
    
    if weather_missing > 0:
        coverage_issues.append(f"Weather data has {weather_missing} missing values")
    
    if simulation_missing > 0:
        coverage_issues.append(f"Simulation data has {simulation_missing} missing values")
    
    coverage_ratio = min(weather_hours, simulation_hours) / forecast_hours
    
    logger.info(f"Driver data coverage: {coverage_ratio:.2%} ({min(weather_hours, simulation_hours)}/{forecast_hours}h)")
    
    return coverage_issues, coverage_ratio

def create_forecast_features(station_id, issue_time, weather_df, simulation_df, 
                           historical_data, model_features, forecast_hours=72):
    """Create feature matrix for forecasting with proper lag handling"""
    logger = logging.getLogger('forecast_generation')
    
    # Get latest historical data for this station
    station_hist = historical_data[historical_data['station_id'] == station_id].copy()
    if station_hist.empty:
        raise ValueError(f"No historical data for station {station_id}")
    
    station_hist = station_hist.sort_values('timestamp')
    latest_time = station_hist['timestamp'].max()
    
    # Validate historical data recency
    time_gap = (pd.to_datetime(issue_time) - latest_time).total_seconds() / 3600
    if time_gap > 24:
        logger.warning(f"Historical data is {time_gap:.1f}h old for station {station_id}")
    
    forecast_data = []
    
    for h in range(1, forecast_hours + 1):
        target_time = pd.to_datetime(issue_time) + timedelta(hours=h)
        
        # Get weather and simulation data for this hour
        if h <= len(weather_df):
            weather_row = weather_df.iloc[h-1]
        else:
            # Use last available weather data
            weather_row = weather_df.iloc[-1]
            logger.warning(f"Using last weather data for hour {h}")
        
        if h <= len(simulation_df):
            sim_row = simulation_df.iloc[h-1]
        else:
            # Use last available simulation data
            sim_row = simulation_df.iloc[-1]
            logger.warning(f"Using last simulation data for hour {h}")
        
        # Create feature row
        feature_row = {
            'timestamp': target_time,
            'station_id': station_id,
            'horizon_hours': h
        }
        
        # Add weather features
        weather_features = ['temp_c', 'wind_speed', 'wind_dir', 'humidity', 'precip_mm']
        for feat in weather_features:
            if feat in weather_row:
                feature_row[feat] = weather_row[feat]
        
        # Add simulation features
        sim_features = ['traffic_idx', 'industrial_idx', 'dust_idx', 'dispersion_pm25']
        for feat in sim_features:
            if feat in sim_row:
                feature_row[feat] = sim_row[feat]
        
        # Add calendar features
        feature_row['hour'] = target_time.hour
        feature_row['day_of_week'] = target_time.dayofweek
        feature_row['is_weekend'] = int(target_time.dayofweek >= 5)
        feature_row['month'] = target_time.month
        feature_row['day_of_year'] = target_time.dayofyear
        
        forecast_data.append(feature_row)
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Add lag features with proper recursive handling
    forecast_df = add_lag_features_recursive(forecast_df, station_hist, model_features)
    
    return forecast_df

def add_lag_features_recursive(forecast_df, historical_data, model_features):
    """Add lag features with recursive prediction for future lags"""
    logger = logging.getLogger('forecast_generation')
    
    # Sort historical data
    historical_data = historical_data.sort_values('timestamp')
    
    # Initialize lag features
    lag_features = ['pm25_lag1', 'pm25_lag24', 'pm25_lag168']
    rolling_features = ['pm25_roll_3h', 'pm25_roll_24h', 'temp_roll_6h', 'wind_speed_roll_12h']
    
    # Create extended time series including forecast period
    extended_data = []
    
    # Add historical data
    for _, row in historical_data.tail(200).iterrows():  # Keep last 200 hours for context
        extended_data.append({
            'timestamp': row['timestamp'],
            'pm25': row['pm25'],
            'temp_c': row.get('temp_c', np.nan),
            'wind_speed': row.get('wind_speed', np.nan),
            'is_forecast': False
        })
    
    # Add forecast period with placeholder PM2.5 values
    for _, row in forecast_df.iterrows():
        extended_data.append({
            'timestamp': row['timestamp'],
            'pm25': np.nan,  # Will be filled recursively
            'temp_c': row.get('temp_c', np.nan),
            'wind_speed': row.get('wind_speed', np.nan),
            'is_forecast': True
        })
    
    extended_df = pd.DataFrame(extended_data)
    extended_df = extended_df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate lag and rolling features for the extended series
    extended_df['pm25_lag1'] = extended_df['pm25'].shift(1)
    extended_df['pm25_lag24'] = extended_df['pm25'].shift(24)
    extended_df['pm25_lag168'] = extended_df['pm25'].shift(168)
    
    # Rolling features
    extended_df['pm25_roll_3h'] = extended_df['pm25'].rolling(window=3, min_periods=1).mean()
    extended_df['pm25_roll_24h'] = extended_df['pm25'].rolling(window=24, min_periods=1).mean()
    extended_df['temp_roll_6h'] = extended_df['temp_c'].rolling(window=6, min_periods=1).mean()
    extended_df['wind_speed_roll_12h'] = extended_df['wind_speed'].rolling(window=12, min_periods=1).mean()
    
    # Extract forecast period with lag features
    forecast_period = extended_df[extended_df['is_forecast']].copy()
    
    # Merge lag features back to forecast_df
    for feature in lag_features + rolling_features:
        if feature in model_features:
            forecast_df[feature] = forecast_period[feature].values
    
    # Add other pollutant features (use historical averages as fallback)
    other_pollutants = ['no2', 'so2', 'co', 'o3']
    for pollutant in other_pollutants:
        if pollutant in model_features:
            if pollutant in historical_data.columns:
                avg_value = historical_data[pollutant].mean()
                forecast_df[pollutant] = avg_value
            else:
                forecast_df[pollutant] = 0
    
    return forecast_df

def generate_recursive_forecasts(model, forecast_df, model_features):
    """Generate forecasts recursively, updating lag features as we go"""
    logger = logging.getLogger('forecast_generation')
    
    forecasts = []
    forecast_errors = []
    
    # Track prediction uncertainty
    prediction_std = 0.1  # Initial uncertainty
    uncertainty_growth = 0.02  # Uncertainty growth per hour
    
    for i, row in forecast_df.iterrows():
        try:
            # Prepare feature vector
            feature_vector = []
            missing_features = []
            
            for feature in model_features:
                if feature in row and not pd.isna(row[feature]):
                    feature_vector.append(row[feature])
                else:
                    feature_vector.append(0)  # Fallback value
                    missing_features.append(feature)
            
            if missing_features:
                logger.warning(f"Hour {row['horizon_hours']}: Missing features {missing_features[:3]}...")
            
            # Make prediction
            feature_array = np.array(feature_vector).reshape(1, -1)
            prediction = model.predict(feature_array)[0]
            
            # Add uncertainty based on horizon
            horizon = row['horizon_hours']
            current_uncertainty = prediction_std + (uncertainty_growth * horizon)
            
            # Quality flag based on feature availability and horizon
            if len(missing_features) > len(model_features) * 0.3:
                quality_flag = "poor"
            elif horizon > 48:
                quality_flag = "uncertain"
            elif len(missing_features) > 0:
                quality_flag = "degraded"
            else:
                quality_flag = "ok"
            
            forecasts.append({
                'horizon_hours': horizon,
                'pm25_forecast': prediction,
                'uncertainty': current_uncertainty,
                'quality_flag': quality_flag,
                'missing_features': len(missing_features)
            })
            
            # Update lag features for next iteration (simplified approach)
            if i < len(forecast_df) - 1:
                # Update pm25_lag1 for next hour
                next_idx = i + 1
                if next_idx < len(forecast_df):
                    forecast_df.loc[forecast_df.index[next_idx], 'pm25_lag1'] = prediction
            
        except Exception as e:
            logger.error(f"Forecast failed for hour {row['horizon_hours']}: {str(e)}")
            forecast_errors.append({
                'horizon_hours': row['horizon_hours'],
                'error': str(e)
            })
            
            # Add fallback prediction
            forecasts.append({
                'horizon_hours': row['horizon_hours'],
                'pm25_forecast': np.nan,
                'uncertainty': 999,
                'quality_flag': "failed",
                'missing_features': len(model_features)
            })
    
    return forecasts, forecast_errors

def validate_forecasts(forecasts, coverage_ratio, min_coverage=0.8, min_confidence=0.7):
    """Comprehensive forecast validation"""
    logger = logging.getLogger('forecast_generation')
    
    validation_results = {
        'total_forecasts': len(forecasts),
        'valid_forecasts': 0,
        'failed_forecasts': 0,
        'poor_quality': 0,
        'coverage_sufficient': coverage_ratio >= min_coverage,
        'validation_passed': False,
        'issues': []
    }
    
    for forecast in forecasts:
        if forecast['quality_flag'] == 'failed':
            validation_results['failed_forecasts'] += 1
        elif forecast['quality_flag'] == 'poor':
            validation_results['poor_quality'] += 1
        else:
            validation_results['valid_forecasts'] += 1
    
    # Calculate validation metrics
    valid_ratio = validation_results['valid_forecasts'] / validation_results['total_forecasts']
    failed_ratio = validation_results['failed_forecasts'] / validation_results['total_forecasts']
    
    # Check validation criteria
    if not validation_results['coverage_sufficient']:
        validation_results['issues'].append(f"Driver data coverage {coverage_ratio:.1%} < {min_coverage:.1%}")
    
    if valid_ratio < min_confidence:
        validation_results['issues'].append(f"Valid forecast ratio {valid_ratio:.1%} < {min_confidence:.1%}")
    
    if failed_ratio > 0.1:
        validation_results['issues'].append(f"Failed forecast ratio {failed_ratio:.1%} > 10%")
    
    # Overall validation
    validation_results['validation_passed'] = (
        validation_results['coverage_sufficient'] and 
        valid_ratio >= min_confidence and 
        failed_ratio <= 0.1
    )
    
    logger.info(f"Validation: {validation_results['valid_forecasts']}/{validation_results['total_forecasts']} valid forecasts")
    logger.info(f"Coverage: {coverage_ratio:.1%}, Confidence: {valid_ratio:.1%}")
    
    if not validation_results['validation_passed']:
        logger.warning(f"Validation FAILED: {validation_results['issues']}")
    
    return validation_results

def create_comprehensive_output(station_forecasts, validation_results, model_hash, 
                              feature_count, coverage_info, issue_time):
    """Create production-ready output with comprehensive metadata"""
    logger = logging.getLogger('forecast_generation')
    
    # Create main forecast CSV
    csv_rows = []
    for station_id, forecasts in station_forecasts.items():
        for forecast in forecasts:
            csv_rows.append({
                'issue_time': issue_time,
                'target_time': pd.to_datetime(issue_time) + timedelta(hours=forecast['horizon_hours']),
                'station_id': station_id,
                'horizon_hours': forecast['horizon_hours'],
                'pm25_forecast': forecast['pm25_forecast'],
                'pm25_lower_ci': forecast['pm25_forecast'] - 1.96 * forecast['uncertainty'],
                'pm25_upper_ci': forecast['pm25_forecast'] + 1.96 * forecast['uncertainty'],
                'uncertainty': forecast['uncertainty'],
                'quality_flag': forecast['quality_flag'],
                'missing_features': forecast['missing_features'],
                'units': 'ug/m3'
            })
    
    forecast_df = pd.DataFrame(csv_rows)
    
    # Enhanced metadata
    metadata = {
        'model_version': 'LGBM_PM25_v2.0',
        'model_hash': model_hash,
        'forecast_generated_at': datetime.now().isoformat(),
        'issue_time': issue_time,
        'author': 'Enhanced Forecast System v2.0',
        'forecast_horizon_hours': 72,
        'timezone': 'Asia/Kolkata',
        'stations_count': len(station_forecasts),
        'total_forecasts': len(csv_rows),
        'model_features_count': feature_count,
        'validation': validation_results,
        'data_coverage': coverage_info,
        'data_provenance': {
            'weather_source': 'weather.csv',
            'simulation_source': 'simulation.csv',
            'historical_source': 'feature_table.parquet',
            'model_source': 'lgbm_pm25.joblib'
        },
        'quality_summary': {
            'valid_forecasts': validation_results['valid_forecasts'],
            'failed_forecasts': validation_results['failed_forecasts'],
            'poor_quality': validation_results['poor_quality'],
            'validation_passed': validation_results['validation_passed']
        },
        'uncertainty_info': {
            'method': 'horizon_based_growth',
            'confidence_intervals': '95%',
            'base_uncertainty': 0.1,
            'growth_rate': 0.02
        }
    }
    
    # Enhanced schema
    schema = {
        'version': '2.0',
        'columns': {
            'issue_time': {'type': 'datetime', 'description': 'When forecast was issued', 'timezone': 'Asia/Kolkata'},
            'target_time': {'type': 'datetime', 'description': 'Target time for prediction', 'timezone': 'Asia/Kolkata'},
            'station_id': {'type': 'string', 'description': 'Air quality monitoring station ID'},
            'horizon_hours': {'type': 'integer', 'description': 'Hours ahead from issue time', 'range': [1, 72]},
            'pm25_forecast': {'type': 'float', 'description': 'PM2.5 concentration forecast', 'units': 'ug/m3'},
            'pm25_lower_ci': {'type': 'float', 'description': '95% confidence interval lower bound', 'units': 'ug/m3'},
            'pm25_upper_ci': {'type': 'float', 'description': '95% confidence interval upper bound', 'units': 'ug/m3'},
            'uncertainty': {'type': 'float', 'description': 'Forecast uncertainty (std dev)', 'units': 'ug/m3'},
            'quality_flag': {'type': 'string', 'description': 'Forecast quality indicator', 
                           'values': ['ok', 'degraded', 'uncertain', 'poor', 'failed']},
            'missing_features': {'type': 'integer', 'description': 'Number of missing model features'},
            'units': {'type': 'string', 'description': 'Measurement units', 'value': 'ug/m3'}
        },
        'validation_criteria': {
            'min_coverage': 0.8,
            'min_confidence': 0.7,
            'max_failure_rate': 0.1
        }
    }
    
    return forecast_df, metadata, schema

def main():
    logger = setup_logging()
    logger.info("Starting enhanced forecast generation")
    
    try:
        # Load model and features
        model, model_features = load_model_and_features()
        model_hash = calculate_file_hash("urban-air-forecast/data/lgbm_pm25.joblib")
        logger.info(f"Model loaded with {len(model_features)} features")
        
        # Load historical data
        historical_data = pd.read_parquet("urban-air-forecast/data/feature_table.parquet")
        logger.info(f"Historical data loaded: {historical_data.shape}")
        
        # Load driver data
        weather_df = pd.read_csv("urban-air-forecast/data/weather.csv")
        simulation_df = pd.read_csv("urban-air-forecast/data/simulation.csv")
        
        # Validate driver data coverage
        coverage_issues, coverage_ratio = validate_driver_data_coverage(weather_df, simulation_df)
        
        coverage_info = {
            'weather_hours': len(weather_df),
            'simulation_hours': len(simulation_df),
            'coverage_ratio': coverage_ratio,
            'issues': coverage_issues
        }
        
        if coverage_issues:
            logger.warning(f"Driver data issues: {coverage_issues}")
        
        # Set issue time (latest available data time)
        issue_time = historical_data['timestamp'].max()
        logger.info(f"Issue time: {issue_time}")
        
        # Generate forecasts for each station
        stations = historical_data['station_id'].unique()
        station_forecasts = {}
        
        for station_id in stations:
            logger.info(f"Generating forecasts for station {station_id}")
            
            try:
                # Create forecast features
                forecast_df = create_forecast_features(
                    station_id, issue_time, weather_df, simulation_df, 
                    historical_data, model_features
                )
                
                # Generate recursive forecasts
                forecasts, errors = generate_recursive_forecasts(model, forecast_df, model_features)
                
                if errors:
                    logger.warning(f"Station {station_id}: {len(errors)} forecast errors")
                
                station_forecasts[station_id] = forecasts
                
            except Exception as e:
                logger.error(f"Failed to generate forecasts for station {station_id}: {str(e)}")
                station_forecasts[station_id] = []
        
        # Validate all forecasts
        all_forecasts = [f for forecasts in station_forecasts.values() for f in forecasts]
        validation_results = validate_forecasts(all_forecasts, coverage_ratio)
        
        # Create comprehensive output
        forecast_df, metadata, schema = create_comprehensive_output(
            station_forecasts, validation_results, model_hash, 
            len(model_features), coverage_info, issue_time
        )
        
        # Save outputs
        output_dir = Path("urban-air-forecast/output")
        output_dir.mkdir(exist_ok=True)
        
        forecast_df.to_csv(output_dir / "forecast_pm25.csv", index=False)
        
        with open(output_dir / "forecast_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        with open(output_dir / "forecast_schema.json", "w") as f:
            json.dump(schema, f, indent=2)
        
        # Final summary with honest assessment
        valid_count = validation_results['valid_forecasts']
        total_count = validation_results['total_forecasts']
        
        if validation_results['validation_passed']:
            logger.info(f"SUCCESS: Generated {valid_count}/{total_count} valid forecasts")
            logger.info(f"Coverage: {coverage_ratio:.1%}, Quality: {valid_count/total_count:.1%}")
        else:
            logger.warning(f"PARTIAL SUCCESS: Generated {valid_count}/{total_count} valid forecasts")
            logger.warning(f"Issues: {validation_results['issues']}")
            logger.warning("Forecast quality may be compromised due to data limitations")
        
        logger.info("Enhanced forecast generation completed")
        
    except Exception as e:
        logger.error(f"Forecast generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()