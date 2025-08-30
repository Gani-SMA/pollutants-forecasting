import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data():
    data_path = Path("urban-air-forecast/data/feature_table.parquet")
    df = pd.read_parquet(data_path)
    return df

def baseline_model(df):
    baseline_data = df.dropna(subset=['pm25_lag24'])
    y_true = baseline_data['pm25']
    y_pred = baseline_data['pm25_lag24']
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    baseline_metrics = {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "samples": len(baseline_data)
    }
    
    with open("urban-air-forecast/data/baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    
    return baseline_metrics

def select_features(X, y, k=15):
    """Select top k features using statistical tests"""
    if len(X.columns) <= k:
        return X.columns.tolist()
    
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    return selected_features

def train_lightgbm(df):
    feature_columns = [
        'no2', 'so2', 'co', 'o3',
        'temp_c', 'wind_speed', 'wind_dir', 'humidity', 'precip_mm',
        'traffic_idx', 'industrial_idx', 'dust_idx', 'dispersion_pm25',
        'hour', 'day_of_week', 'is_weekend', 'month', 'day_of_year',
        'pm25_lag1', 'pm25_lag24',
        'pm25_roll_3h', 'pm25_roll_24h', 'temp_roll_6h', 'wind_speed_roll_12h'
    ]
    
    clean_data = df.dropna(subset=feature_columns + ['pm25'])
    
    # Feature selection to reduce overfitting
    X_full = clean_data[feature_columns]
    y = clean_data['pm25']
    
    selected_features = select_features(X_full, y, k=min(15, len(feature_columns)))
    X = clean_data[selected_features]
    
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    # Use appropriate cross-validation based on data size
    if len(X) < 100:
        # Small dataset: use simple train/test split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=False
        )
        splits = [(X_train.index, X_val.index)]
        cv_type = "train_test_split"
    else:
        # Larger dataset: use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=min(5, len(X) // 50))
        splits = list(tscv.split(X))
        cv_type = "time_series_cv"
    
    fold_metrics = {}
    feature_importance_sum = np.zeros(len(selected_features))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Adjust model complexity based on data size
        n_estimators = min(100, max(50, len(X_train) // 10))
        max_depth = min(6, max(3, int(np.log2(len(X_train)))))
        
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=max_depth,
            num_leaves=min(31, 2**max_depth - 1),
            min_child_samples=max(5, len(X_train) // 100),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            force_row_wise=True
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_val)
        
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        fold_metrics[f"fold_{fold+1}"] = {
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2": float(r2),
            "train_samples": len(X_train),
            "val_samples": len(X_val)
        }
        
        feature_importance_sum += model.feature_importances_
        models.append(model)
    
    # Save fold metrics
    fold_metrics["cv_info"] = {
        "cv_type": cv_type,
        "n_folds": len(splits),
        "total_samples": len(X)
    }
    
    with open("urban-air-forecast/data/fold_metrics.json", "w") as f:
        json.dump(fold_metrics, f, indent=2)
    
    # Feature importance
    avg_feature_importance = feature_importance_sum / len(models)
    feature_importance_df = pd.DataFrame({
        'feature_name': selected_features,
        'importance': avg_feature_importance
    }).sort_values('importance', ascending=False)
    
    feature_importance_df.to_csv("urban-air-forecast/data/feature_importance.csv", index=False)
    
    # Train final model on all data
    final_model = lgb.LGBMRegressor(
        n_estimators=min(200, max(50, len(X) // 10)),
        learning_rate=0.1,
        max_depth=min(6, max(3, int(np.log2(len(X))))),
        num_leaves=min(31, 2**min(6, max(3, int(np.log2(len(X))))) - 1),
        min_child_samples=max(5, len(X) // 100),
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        force_row_wise=True
    )
    
    final_model.fit(X, y)
    joblib.dump(final_model, "urban-air-forecast/data/lgbm_pm25.joblib")
    
    return fold_metrics, feature_importance_df

def generate_report(baseline_metrics, fold_metrics, feature_importance_df):
    # Extract fold metrics (excluding cv_info)
    fold_data = {k: v for k, v in fold_metrics.items() if k != "cv_info"}
    cv_info = fold_metrics.get("cv_info", {})
    
    mean_mse = np.mean([metrics["MSE"] for metrics in fold_data.values()])
    mean_rmse = np.mean([metrics["RMSE"] for metrics in fold_data.values()])
    mean_mae = np.mean([metrics["MAE"] for metrics in fold_data.values()])
    mean_r2 = np.mean([metrics["R2"] for metrics in fold_data.values()])
    
    std_rmse = np.std([metrics["RMSE"] for metrics in fold_data.values()])
    
    report = f"""Urban Air Pollution Forecasting - Model Evaluation Report

DATASET INFO:
- Total samples: {cv_info.get('total_samples', 'N/A')}
- Cross-validation: {cv_info.get('cv_type', 'N/A')}
- Number of folds: {cv_info.get('n_folds', 'N/A')}

BASELINE MODEL (Naive-24h):
- MSE: {baseline_metrics['MSE']:.4f}
- RMSE: {baseline_metrics['RMSE']:.4f}
- MAE: {baseline_metrics['MAE']:.4f}
- R²: {baseline_metrics['R2']:.4f}
- Samples: {baseline_metrics['samples']}

LIGHTGBM MODEL:
Fold-wise Performance:
"""
    
    for fold, metrics in fold_data.items():
        report += f"- {fold}: RMSE={metrics['RMSE']:.4f}, R²={metrics['R2']:.4f} (train: {metrics['train_samples']}, val: {metrics['val_samples']})\n"
    
    mse_improvement = ((baseline_metrics['MSE'] - mean_mse) / baseline_metrics['MSE'] * 100)
    rmse_improvement = ((baseline_metrics['RMSE'] - mean_rmse) / baseline_metrics['RMSE'] * 100)
    
    report += f"""
Mean Performance:
- MSE: {mean_mse:.4f} (±{np.std([m['MSE'] for m in fold_data.values()]):.4f})
- RMSE: {mean_rmse:.4f} (±{std_rmse:.4f})
- MAE: {mean_mae:.4f} (±{np.std([m['MAE'] for m in fold_data.values()]):.4f})
- R²: {mean_r2:.4f} (±{np.std([m['R2'] for m in fold_data.values()]):.4f})

MODEL COMPARISON:
- MSE Improvement: {mse_improvement:.2f}%
- RMSE Improvement: {rmse_improvement:.2f}%
- Model Status: {'OUTPERFORMS' if mean_rmse < baseline_metrics['RMSE'] else 'UNDERPERFORMS'} baseline
- Model Reliability: {'STABLE' if std_rmse < mean_rmse * 0.1 else 'UNSTABLE'} (RMSE std: {std_rmse:.4f})

TOP 10 FEATURE IMPORTANCE:
"""
    
    for idx, row in feature_importance_df.head(10).iterrows():
        report += f"- {row['feature_name']}: {row['importance']:.4f}\n"
    
    # Add recommendations
    report += f"""
RECOMMENDATIONS:
"""
    if mean_rmse >= baseline_metrics['RMSE']:
        report += "- Model underperforms baseline - consider more data or simpler model\n"
    if std_rmse > mean_rmse * 0.1:
        report += "- High variance detected - model may be overfitting\n"
    if cv_info.get('total_samples', 0) < 500:
        report += "- Small dataset - results may not generalize well\n"
    if mean_r2 < 0.5:
        report += "- Low R2 score - model explains limited variance\n"
    
    with open("urban-air-forecast/data/evaluation_report.txt", "w", encoding='utf-8') as f:
        f.write(report)
    
    print("✅ Model training completed successfully!")
    print(f"📊 Dataset: {cv_info.get('total_samples', 'N/A')} samples")
    print(f"📊 Baseline RMSE: {baseline_metrics['RMSE']:.4f}")
    print(f"📊 LightGBM RMSE: {mean_rmse:.4f} (±{std_rmse:.4f})")
    print(f"📈 Improvement: {rmse_improvement:.2f}%")
    print(f"🎯 Model R²: {mean_r2:.4f}")

def main():
    Path("urban-air-forecast/data").mkdir(parents=True, exist_ok=True)
    
    df = load_data()
    print(f"📊 Loaded data: {df.shape}")
    
    baseline_metrics = baseline_model(df)
    print(f"🔄 Baseline model evaluated")
    
    fold_metrics, feature_importance_df = train_lightgbm(df)
    print(f"🤖 LightGBM model trained")
    
    generate_report(baseline_metrics, fold_metrics, feature_importance_df)

if __name__ == "__main__":
    main()