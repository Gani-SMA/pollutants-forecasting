import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('urban-air-forecast/logs/system_validation.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('system_validation')

def analyze_forecast_quality(forecast_df):
    """Comprehensive forecast quality analysis"""
    logger = logging.getLogger('system_validation')
    
    quality_analysis = {
        'total_forecasts': len(forecast_df),
        'stations': forecast_df['station_id'].nunique(),
        'horizon_coverage': {
            'min_horizon': forecast_df['horizon_hours'].min(),
            'max_horizon': forecast_df['horizon_hours'].max(),
            'expected_hours': 72
        },
        'quality_flags': forecast_df['quality_flag'].value_counts().to_dict(),
        'missing_features_stats': {
            'mean': forecast_df['missing_features'].mean(),
            'max': forecast_df['missing_features'].max(),
            'std': forecast_df['missing_features'].std()
        },
        'uncertainty_analysis': {
            'mean_uncertainty': forecast_df['uncertainty'].mean(),
            'uncertainty_growth': forecast_df.groupby('horizon_hours')['uncertainty'].mean().to_dict()
        },
        'value_analysis': {
            'mean_forecast': forecast_df['pm25_forecast'].mean(),
            'forecast_range': [forecast_df['pm25_forecast'].min(), forecast_df['pm25_forecast'].max()],
            'unrealistic_values': len(forecast_df[forecast_df['pm25_forecast'] < 0]),
            'extreme_values': len(forecast_df[forecast_df['pm25_forecast'] > 500])
        }
    }
    
    # Check for unrealistically smooth forecasts
    station_smoothness = {}
    for station in forecast_df['station_id'].unique():
        station_data = forecast_df[forecast_df['station_id'] == station].sort_values('horizon_hours')
        if len(station_data) > 1:
            forecast_diff = station_data['pm25_forecast'].diff().abs()
            smoothness_score = forecast_diff.mean()
            station_smoothness[station] = smoothness_score
    
    quality_analysis['smoothness_analysis'] = {
        'station_smoothness': station_smoothness,
        'avg_smoothness': np.mean(list(station_smoothness.values())),
        'smoothness_threshold': 5.0  # Expected minimum variation
    }
    
    return quality_analysis

def validate_feature_consistency():
    """Check feature consistency between training and forecasting"""
    logger = logging.getLogger('system_validation')
    
    # Load training features
    feature_importance = pd.read_csv("urban-air-forecast/data/feature_importance.csv")
    training_features = set(feature_importance['feature_name'].tolist())
    
    # Load model features (from joblib model)
    import joblib
    model = joblib.load("urban-air-forecast/data/lgbm_pm25.joblib")
    if hasattr(model, 'feature_name_'):
        model_features = set(model.feature_name_)
    else:
        model_features = training_features
    
    # Load historical data features
    historical_data = pd.read_parquet("urban-air-forecast/data/feature_table.parquet")
    historical_features = set(historical_data.columns) - {'timestamp', 'station_id', 'pm25'}
    
    consistency_analysis = {
        'training_features_count': len(training_features),
        'model_features_count': len(model_features),
        'historical_features_count': len(historical_features),
        'feature_drift': {
            'missing_in_model': list(training_features - model_features),
            'extra_in_model': list(model_features - training_features),
            'missing_in_historical': list(model_features - historical_features)
        },
        'consistency_score': len(training_features & model_features) / len(training_features | model_features)
    }
    
    return consistency_analysis

def analyze_driver_data_alignment():
    """Analyze weather and simulation data alignment with forecast requirements"""
    logger = logging.getLogger('system_validation')
    
    weather_df = pd.read_csv("urban-air-forecast/data/weather.csv")
    simulation_df = pd.read_csv("urban-air-forecast/data/simulation.csv")
    
    # Convert timestamps
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    simulation_df['timestamp'] = pd.to_datetime(simulation_df['timestamp'])
    
    alignment_analysis = {
        'weather_data': {
            'total_hours': len(weather_df),
            'stations': weather_df['station_id'].nunique(),
            'time_range': [weather_df['timestamp'].min(), weather_df['timestamp'].max()],
            'missing_values': weather_df.isnull().sum().sum(),
            'coverage_for_72h': len(weather_df) >= 72
        },
        'simulation_data': {
            'total_hours': len(simulation_df),
            'stations': simulation_df['station_id'].nunique(),
            'time_range': [simulation_df['timestamp'].min(), simulation_df['timestamp'].max()],
            'missing_values': simulation_df.isnull().sum().sum(),
            'coverage_for_72h': len(simulation_df) >= 72
        },
        'alignment_issues': []
    }
    
    # Check temporal alignment
    weather_hours = len(weather_df)
    simulation_hours = len(simulation_df)
    
    if weather_hours < 72:
        alignment_analysis['alignment_issues'].append(f"Weather data insufficient: {weather_hours}/72 hours")
    
    if simulation_hours < 72:
        alignment_analysis['alignment_issues'].append(f"Simulation data insufficient: {simulation_hours}/72 hours")
    
    if weather_df['station_id'].nunique() != simulation_df['station_id'].nunique():
        alignment_analysis['alignment_issues'].append("Station count mismatch between weather and simulation")
    
    return alignment_analysis

def validate_output_schema():
    """Validate output schema completeness and correctness"""
    logger = logging.getLogger('system_validation')
    
    # Load outputs
    forecast_df = pd.read_csv("urban-air-forecast/output/forecast_pm25.csv")
    
    with open("urban-air-forecast/output/forecast_metadata.json", "r") as f:
        metadata = json.load(f)
    
    with open("urban-air-forecast/output/forecast_schema.json", "r") as f:
        schema = json.load(f)
    
    schema_validation = {
        'csv_columns': list(forecast_df.columns),
        'expected_columns': [
            'issue_time', 'target_time', 'station_id', 'horizon_hours',
            'pm25_forecast', 'pm25_lower_ci', 'pm25_upper_ci', 'uncertainty',
            'quality_flag', 'missing_features', 'units'
        ],
        'missing_columns': [],
        'extra_columns': [],
        'metadata_completeness': {
            'has_model_hash': 'model_hash' in metadata,
            'has_validation_results': 'validation' in metadata,
            'has_data_provenance': 'data_provenance' in metadata,
            'has_uncertainty_info': 'uncertainty_info' in metadata,
            'has_quality_summary': 'quality_summary' in metadata
        },
        'schema_version': schema.get('version', 'unknown'),
        'units_consistency': forecast_df['units'].nunique() == 1
    }
    
    # Check for missing/extra columns
    expected_cols = set(schema_validation['expected_columns'])
    actual_cols = set(schema_validation['csv_columns'])
    
    schema_validation['missing_columns'] = list(expected_cols - actual_cols)
    schema_validation['extra_columns'] = list(actual_cols - expected_cols)
    schema_validation['schema_complete'] = len(schema_validation['missing_columns']) == 0
    
    return schema_validation

def analyze_recursive_lag_propagation():
    """Analyze the impact of recursive lag feature propagation"""
    logger = logging.getLogger('system_validation')
    
    forecast_df = pd.read_csv("urban-air-forecast/output/forecast_pm25.csv")
    
    propagation_analysis = {}
    
    for station in forecast_df['station_id'].unique():
        station_data = forecast_df[forecast_df['station_id'] == station].sort_values('horizon_hours')
        
        # Analyze forecast stability over horizon
        forecast_values = station_data['pm25_forecast'].values
        uncertainty_values = station_data['uncertainty'].values
        
        # Calculate error propagation indicators
        forecast_diff = np.diff(forecast_values)
        uncertainty_growth = np.diff(uncertainty_values)
        
        # Detect unrealistic patterns
        low_variation = np.std(forecast_values) < 2.0  # Too smooth
        high_variation = np.std(forecast_values) > 50.0  # Too volatile
        
        propagation_analysis[station] = {
            'forecast_std': np.std(forecast_values),
            'mean_forecast_change': np.mean(np.abs(forecast_diff)),
            'uncertainty_growth_rate': np.mean(uncertainty_growth),
            'low_variation_flag': low_variation,
            'high_variation_flag': high_variation,
            'max_uncertainty': np.max(uncertainty_values),
            'error_propagation_score': np.std(forecast_diff) / np.mean(forecast_values) if np.mean(forecast_values) > 0 else 0
        }
    
    return propagation_analysis

def generate_comprehensive_report():
    """Generate comprehensive system validation report"""
    logger = setup_logging()
    logger.info("Starting comprehensive system validation")
    
    # Run all analyses
    forecast_df = pd.read_csv("urban-air-forecast/output/forecast_pm25.csv")
    
    quality_analysis = analyze_forecast_quality(forecast_df)
    feature_consistency = validate_feature_consistency()
    driver_alignment = analyze_driver_data_alignment()
    schema_validation = validate_output_schema()
    propagation_analysis = analyze_recursive_lag_propagation()
    
    # Generate comprehensive report
    report = f"""
URBAN AIR POLLUTION FORECASTING SYSTEM - COMPREHENSIVE VALIDATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== EXECUTIVE SUMMARY ===
Total Forecasts: {quality_analysis['total_forecasts']}
Stations: {quality_analysis['stations']}
Horizon Coverage: {quality_analysis['horizon_coverage']['min_horizon']}-{quality_analysis['horizon_coverage']['max_horizon']} hours
Schema Complete: {schema_validation['schema_complete']}
Feature Consistency Score: {feature_consistency['consistency_score']:.2%}

=== FORECAST QUALITY ANALYSIS ===
Quality Flag Distribution:
"""
    
    for flag, count in quality_analysis['quality_flags'].items():
        percentage = (count / quality_analysis['total_forecasts']) * 100
        report += f"  - {flag}: {count} ({percentage:.1f}%)\n"
    
    report += f"""
Missing Features:
  - Average: {quality_analysis['missing_features_stats']['mean']:.1f}
  - Maximum: {quality_analysis['missing_features_stats']['max']}
  - Standard Deviation: {quality_analysis['missing_features_stats']['std']:.1f}

Forecast Value Analysis:
  - Mean PM2.5: {quality_analysis['value_analysis']['mean_forecast']:.1f} ug/m3
  - Range: {quality_analysis['value_analysis']['forecast_range'][0]:.1f} - {quality_analysis['value_analysis']['forecast_range'][1]:.1f} ug/m3
  - Unrealistic Values (<0): {quality_analysis['value_analysis']['unrealistic_values']}
  - Extreme Values (>500): {quality_analysis['value_analysis']['extreme_values']}

Smoothness Analysis:
  - Average Smoothness: {quality_analysis['smoothness_analysis']['avg_smoothness']:.2f}
  - Threshold: {quality_analysis['smoothness_analysis']['smoothness_threshold']}
  - Unrealistically Smooth: {'YES' if quality_analysis['smoothness_analysis']['avg_smoothness'] < quality_analysis['smoothness_analysis']['smoothness_threshold'] else 'NO'}

=== FEATURE CONSISTENCY ANALYSIS ===
Feature Counts:
  - Training Features: {feature_consistency['training_features_count']}
  - Model Features: {feature_consistency['model_features_count']}
  - Historical Features: {feature_consistency['historical_features_count']}

Feature Drift Issues:
  - Missing in Model: {len(feature_consistency['feature_drift']['missing_in_model'])}
  - Extra in Model: {len(feature_consistency['feature_drift']['extra_in_model'])}
  - Missing in Historical: {len(feature_consistency['feature_drift']['missing_in_historical'])}

Consistency Score: {feature_consistency['consistency_score']:.2%}

=== DRIVER DATA ALIGNMENT ===
Weather Data:
  - Total Hours: {driver_alignment['weather_data']['total_hours']}
  - Stations: {driver_alignment['weather_data']['stations']}
  - Missing Values: {driver_alignment['weather_data']['missing_values']}
  - 72h Coverage: {'YES' if driver_alignment['weather_data']['coverage_for_72h'] else 'NO'}

Simulation Data:
  - Total Hours: {driver_alignment['simulation_data']['total_hours']}
  - Stations: {driver_alignment['simulation_data']['stations']}
  - Missing Values: {driver_alignment['simulation_data']['missing_values']}
  - 72h Coverage: {'YES' if driver_alignment['simulation_data']['coverage_for_72h'] else 'NO'}

Alignment Issues: {len(driver_alignment['alignment_issues'])}
"""
    
    for issue in driver_alignment['alignment_issues']:
        report += f"  - {issue}\n"
    
    report += f"""
=== OUTPUT SCHEMA VALIDATION ===
Schema Version: {schema_validation['schema_version']}
Schema Complete: {'YES' if schema_validation['schema_complete'] else 'NO'}
Missing Columns: {len(schema_validation['missing_columns'])}
Extra Columns: {len(schema_validation['extra_columns'])}
Units Consistent: {'YES' if schema_validation['units_consistency'] else 'NO'}

Metadata Completeness:
  - Model Hash: {'YES' if schema_validation['metadata_completeness']['has_model_hash'] else 'NO'}
  - Validation Results: {'YES' if schema_validation['metadata_completeness']['has_validation_results'] else 'NO'}
  - Data Provenance: {'YES' if schema_validation['metadata_completeness']['has_data_provenance'] else 'NO'}
  - Uncertainty Info: {'YES' if schema_validation['metadata_completeness']['has_uncertainty_info'] else 'NO'}
  - Quality Summary: {'YES' if schema_validation['metadata_completeness']['has_quality_summary'] else 'NO'}

=== RECURSIVE LAG PROPAGATION ANALYSIS ===
"""
    
    for station, analysis in propagation_analysis.items():
        report += f"""
Station {station}:
  - Forecast Std Dev: {analysis['forecast_std']:.2f}
  - Mean Change: {analysis['mean_forecast_change']:.2f}
  - Uncertainty Growth: {analysis['uncertainty_growth_rate']:.4f}
  - Low Variation Flag: {'YES' if analysis['low_variation_flag'] else 'NO'}
  - High Variation Flag: {'YES' if analysis['high_variation_flag'] else 'NO'}
  - Max Uncertainty: {analysis['max_uncertainty']:.2f}
  - Error Propagation Score: {analysis['error_propagation_score']:.4f}
"""
    
    # Overall assessment
    critical_issues = []
    warnings_list = []
    
    # Check for critical issues
    if not schema_validation['schema_complete']:
        critical_issues.append("Incomplete output schema")
    
    if feature_consistency['consistency_score'] < 0.8:
        critical_issues.append("Significant feature drift detected")
    
    if len(driver_alignment['alignment_issues']) > 0:
        critical_issues.append("Driver data alignment issues")
    
    if quality_analysis['value_analysis']['unrealistic_values'] > 0:
        critical_issues.append("Unrealistic forecast values detected")
    
    # Check for warnings
    if quality_analysis['smoothness_analysis']['avg_smoothness'] < quality_analysis['smoothness_analysis']['smoothness_threshold']:
        warnings_list.append("Forecasts may be unrealistically smooth")
    
    if quality_analysis['missing_features_stats']['mean'] > 2:
        warnings_list.append("High number of missing features in forecasts")
    
    failed_forecasts = quality_analysis['quality_flags'].get('failed', 0)
    if failed_forecasts > 0:
        warnings_list.append(f"{failed_forecasts} failed forecasts detected")
    
    report += f"""
=== OVERALL ASSESSMENT ===
Critical Issues: {len(critical_issues)}
"""
    for issue in critical_issues:
        report += f"  ❌ {issue}\n"
    
    report += f"""
Warnings: {len(warnings_list)}
"""
    for warning in warnings_list:
        report += f"  ⚠️  {warning}\n"
    
    if len(critical_issues) == 0 and len(warnings_list) == 0:
        report += "\n✅ SYSTEM VALIDATION PASSED - No critical issues or warnings detected\n"
    elif len(critical_issues) == 0:
        report += f"\n⚠️  SYSTEM VALIDATION PASSED WITH WARNINGS - {len(warnings_list)} warnings detected\n"
    else:
        report += f"\n❌ SYSTEM VALIDATION FAILED - {len(critical_issues)} critical issues detected\n"
    
    # Save report
    with open("urban-air-forecast/output/system_validation_report.txt", "w", encoding='utf-8') as f:
        f.write(report)
    
    # Save detailed analysis as JSON
    detailed_analysis = {
        'timestamp': datetime.now().isoformat(),
        'quality_analysis': quality_analysis,
        'feature_consistency': feature_consistency,
        'driver_alignment': driver_alignment,
        'schema_validation': schema_validation,
        'propagation_analysis': propagation_analysis,
        'critical_issues': critical_issues,
        'warnings': warnings_list
    }
    
    with open("urban-air-forecast/output/detailed_validation.json", "w") as f:
        json.dump(detailed_analysis, f, indent=2, default=str)
    
    logger.info(f"Validation complete: {len(critical_issues)} critical issues, {len(warnings_list)} warnings")
    print(report)

if __name__ == "__main__":
    generate_comprehensive_report()