import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import os
import joblib

def train_dynamic_lag_model():
    data_path = r'c:\Users\Ranuga\Data Science Project\Data Pre Processing\Stage 4\Combined_data - Main - Combined_data.csv'
    output_dir = r'c:\Users\Ranuga\Data Science Project\Data Pre Processing\Stage 10'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # 1. Feature Engineering (Inherited from 6, 7, 9)
    df['week_num'] = pd.to_numeric(df['week'].str.extract(r'(\d+)')[0])
    df['week_sin'] = np.sin(2 * np.pi * df['week_num'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_num'] / 52)
    
    regional_weather = df.groupby(['Year_Week', 'vegetable_zone'])[['rain_sum', 'mean_apparent_temperature']].mean().reset_index()
    regional_weather = regional_weather.rename(columns={'rain_sum': 'reg_rain', 'mean_apparent_temperature': 'reg_temp'})
    df = pd.merge(df, regional_weather, on=['Year_Week', 'vegetable_zone'], how='left')
    
    df['season_enc'] = LabelEncoder().fit_transform(df['seasonality'].astype(str))
    df['diesel_season_int'] = df['lanka_auto_diesel_price'] * (df['season_enc'] + 1)
    
    # 2. DYNAMIC LAGS
    # We add 8-week lags for price and regional weather to capture longer growth cycles
    print("Generating extended 8-week lags for slow-growing crop signals...")
    df = df.sort_values(['location', 'vegetable_type', 'year', 'week_num'])
    
    cols_to_lag = ['price', 'reg_rain', 'reg_temp']
    lags = [1, 2, 3, 4, 8] # Added Lag 8
    
    for col in cols_to_lag:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df.groupby(['location', 'vegetable_type'])[col].shift(lag)
            
    df['price_roll_4'] = df.groupby(['location', 'vegetable_type'])['price'].transform(lambda x: x.shift(1).rolling(4).mean())
    
    # Drop NaNs created by 8-week lag
    df_ready = df.dropna(subset=['price_lag_8']).copy()
    
    for col in ['location', 'vegetable_type', 'vegetable_zone']:
        df_ready[f'{col}_enc'] = LabelEncoder().fit_transform(df_ready[col].astype(str))
        
    # 3. Split
    train_list, test_list = [], []
    for _, group in df_ready.groupby(['location', 'vegetable_type']):
        split = int(len(group) * 0.8)
        train_list.append(group.iloc[:split])
        test_list.append(group.iloc[split:])
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)
    
    features = [
        'year', 'week_sin', 'week_cos', 
        'lanka_auto_diesel_price', 'usd_exchange_rate', 
        'reg_rain', 'reg_temp', 'diesel_season_int',
        'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_4', 'price_lag_8',
        'reg_rain_lag_1', 'reg_rain_lag_4', 'reg_rain_lag_8',
        'reg_temp_lag_1', 'reg_temp_lag_4', 'reg_temp_lag_8',
        'price_roll_4',
        'location_enc', 'vegetable_type_enc', 'vegetable_zone_enc', 'season_enc'
    ]
    
    X_train, y_train = train_df[features], train_df['price']
    X_test, y_test = test_df[features], test_df['price']
    
    # 4. Model
    print(f"Training Model with 8-week Lags ({len(features)} features)...")
    model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    
    # 5. Eval
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # Baseline from Stage 9 (since we have cyclical encoding here too)
    base_r2 = 0.8746
    
    report = f"""Stage 10: Dynamic Lag Selection Report
======================================
Extended Feature: 8-Week Lags for Weather and Price

Metrics:
--------
R2 Score: {r2:.4f} (Change vs Stage 9: {r2 - base_r2:+.4f})
Accuracy (1 - MAPE): {(1 - mape)*100:.2f}%

Top 5 Feature Importances:
---------------------------
"""
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    for feat, val in importances.head(5).items():
        report += f"- {feat}: {val:.4f}\n"

    print(report)
    with open(os.path.join(output_dir, 'dynamic_lag_performance.txt'), 'w') as f:
        f.write(report)
    joblib.dump(model, os.path.join(output_dir, 'dynamic_lag_model.joblib'))

if __name__ == "__main__":
    train_dynamic_lag_model()
