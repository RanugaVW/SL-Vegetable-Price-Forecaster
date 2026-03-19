import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from sklearn.preprocessing import LabelEncoder

def main():
    # Paths
    data_path = r'C:\Users\Ranuga\Data Science Project\5. Model Building\5.9 - Model Validation\Datasets\2024_dataset_complete.csv'
    model_path = r'C:\Users\Ranuga\Data Science Project\5. Model Building\5.8 - Retail Price Ensemble Models\XGBoost + LightBGM\Models\xgb_lgbm_advanced_ensemble_optuna_model.joblib'
    output_dir = r'C:\Users\Ranuga\Data Science Project\5. Model Building\5.9 - Model Validation\Datasets'

    print('Loading Advanced Model Bundle...')
    bundle = joblib.load(model_path)

    # Extract correct nested keys
    model_xgb = bundle['xgb']
    model_lgb = bundle['lgb']
    features = bundle['features']
    best_weight_lgb = bundle['weights']['lgb']
    best_weight_xgb = bundle['weights']['xgb']
    le_dict = bundle['label_encoders']

    print(f"Loaded! LGBM Weight: {best_weight_lgb:.4f} | XGB Weight: {best_weight_xgb:.4f}")

    print('Loading 2024 Data...')
    df = pd.read_csv(data_path)

    # Setup structural timeline
    df['week_str'] = df['week'].astype(str).str.zfill(2)
    df['Year_Week'] = df['year'].astype(str) + '_' + df['week_str']
    df['week_num'] = pd.to_numeric(df['week'].astype(str).str.extract(r'(\d+)')[0])
    df['week_sin'] = np.sin(2 * np.pi * df['week_num'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_num'] / 52)

    # Regional Weather aggregation
    regional_weather = (
        df.groupby(['Year_Week', 'vegetable_zone'])[['rain_sum', 'mean_apparent_temperature']]
        .mean().reset_index()
        .rename(columns={'rain_sum': 'reg_rain', 'mean_apparent_temperature': 'reg_temp'})
    )
    df = pd.merge(df, regional_weather, on=['Year_Week', 'vegetable_zone'], how='left')

    # Secure classification mapping for previously trained encoders
    if 'seasonality' in le_dict:
        le_season = le_dict['seasonality']
        # Safely handle 'unseen' classes to avoid crashing
        df['season_enc'] = df['seasonality'].astype(str).apply(lambda x: x if x in le_season.classes_ else le_season.classes_[0])
        df['season_enc'] = le_season.transform(df['season_enc'])
    else:
        df['season_enc'] = LabelEncoder().fit_transform(df['seasonality'].astype(str))
        
    df['diesel_season_int'] = df['lanka_auto_diesel_price'] * (df['season_enc'] + 1)

    # Critical: Time-Series Sorting
    df = df.sort_values(['retail_market', 'vegetable_type', 'year', 'week_num']).reset_index(drop=True)

    # Build physical Lags
    for col in ['retail_price', 'reg_rain', 'reg_temp']:
        for lag in [1, 2, 3, 4, 8]:
            df[f'{col}_lag_{lag}'] = df.groupby(['retail_market', 'vegetable_type'])[col].shift(lag)

    for lag in [1, 2, 3, 4, 5, 6, 8]:
        df[f'mean_farmer_price_lag_{lag}'] = df.groupby(['retail_market', 'vegetable_type'])['mean_farmer_price'].shift(lag)

    # Rolling Moments
    df['retail_price_roll_4'] = df.groupby(['retail_market', 'vegetable_type'])['retail_price'].transform(lambda x: x.shift(1).rolling(4).mean())
    grp = df.groupby(['retail_market', 'vegetable_type'])['mean_farmer_price']
    df['farmer_price_roll_4'] = grp.transform(lambda x: x.shift(1).rolling(4).mean())
    df['farmer_price_roll_8'] = grp.transform(lambda x: x.shift(1).rolling(8).mean())
    df['farmer_price_roll_std_4'] = grp.transform(lambda x: x.shift(1).rolling(4).std())
    df['farmer_price_pct_change_1'] = grp.transform(lambda x: x.shift(1).pct_change(1, fill_method=None))

    df['mean_farmer_price_filled'] = df['mean_farmer_price'].fillna(df['mean_farmer_price_lag_1'])
    df['farmer_retail_spread_lag_1'] = df['retail_price_lag_1'] - df['mean_farmer_price_lag_1']

    df['retail_price_momentum_1_4'] = df['retail_price_lag_1'] / (df['retail_price_lag_4'] + 1e-5)
    df['farmer_price_momentum_1_4'] = df['mean_farmer_price_lag_1'] / (df['mean_farmer_price_lag_4'] + 1e-5)

    # Purge initial drop gaps caused by 8-week lags
    df_ready = df.dropna(subset=['retail_price_lag_8', 'mean_farmer_price_lag_8', 'farmer_price_roll_8', 'retail_price_momentum_1_4']).copy()

    # Enforce Categorical
    for col in ['retail_market', 'vegetable_type', 'vegetable_zone']:
        if col in le_dict:
            le = le_dict[col]
            classes = list(le.classes_)
            df_ready[col] = df_ready[col].apply(lambda x: x if x in classes else classes[0])
            df_ready[f'{col}_enc'] = le.transform(df_ready[col])

    df_ready = df_ready.dropna(subset=features)
    print(f"Data Feature Engineered successfully. Final Test Rows: {len(df_ready)}")

    X_test = df_ready[features]
    y_test = df_ready['retail_price']

    # Predict Log space values natively from advanced models
    pred_xgb_log = model_xgb.predict(X_test)
    pred_lgb_log = model_lgb.predict(X_test)

    # Inverse log mapping to bring values back to real LKR scale correctly
    pred_xgb = np.expm1(pred_xgb_log)
    pred_lgb = np.expm1(pred_lgb_log)

    # Final Ensemble 
    final_pred = (best_weight_lgb * pred_lgb) + (best_weight_xgb * pred_xgb)
    df_ready['Final_Prediction'] = final_pred

    # Strict validation scores
    r2 = r2_score(y_test, final_pred)
    mape = mean_absolute_percentage_error(y_test, final_pred)
    acc = 1 - mape

    print(f"\n2024 Test Dataset Validation:")
    print("-" * 30)
    print(f"R2 Score: {r2:.4f}")
    print(f"MAPE:     {mape:.4f} ({(mape*100):.2f}%)")
    print(f"Accuracy: {acc:.4f} ({(acc*100):.2f}%)")

    # Save detailed text report
    rep_path = os.path.join(output_dir, '2024_validation_report_accurate.txt')
    with open(rep_path, 'w') as f:
        f.write('2024 Unseen Data Model Validation (Final Script Run)\n===================================\n')
        f.write(f'Test Rows Evaluated: {len(X_test)}\n')
        f.write(f'R2 Score: {r2:.4f}\n')
        f.write(f'MAPE: {mape:.4f} ({(mape*100):.2f}%)\n')
        f.write(f'Accuracy: {acc:.4f} ({(acc*100):.2f}%)\n')

    # Visualizations
    # 1. Total Scatter Mapping
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, final_pred, alpha=0.5, color='#45B3A9', edgecolors='k', label='Actual vs Predicted (Dots)')
    min_val = min(y_test.min(), final_pred.min())
    max_val = max(y_test.max(), final_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Accuracy Line')
    plt.title('Model Accuracy Distribution (Test Data)', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Retail Price (LKR)', fontsize=11)
    plt.ylabel('Predicted Retail Price (LKR)', fontsize=11)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Scatter_2024_Test_Accurate.png'), dpi=300)
    plt.close()

    # 2. Time-Series Tracking Line Graph 
    # Let's pick BEETROOT at ANURADHAPURA to perfectly match the requested styling
    veg_list = df_ready['vegetable_type'].unique()
    market_list = df_ready['retail_market'].unique()
    
    sample_veg = 'BEETROOT' if 'BEETROOT' in veg_list else veg_list[0]
    sample_market = 'ANURADHAPURA' if 'ANURADHAPURA' in market_list else market_list[0]

    df_sample = df_ready[(df_ready['vegetable_type'] == sample_veg) & (df_ready['retail_market'] == sample_market)].copy()

    if not df_sample.empty:
        # Sort chronologically for drawing continuous lines
        df_sample = df_sample.sort_values(by=['year', 'week_num'])
        df_sample = df_sample.drop_duplicates(subset=['Year_Week'])

        plt.figure(figsize=(14, 6))
        plt.plot(df_sample['Year_Week'], df_sample['retail_price'], label='Actual Price', marker='o', markersize=6, color='dodgerblue', linewidth=2)
        plt.plot(df_sample['Year_Week'], df_sample['Final_Prediction'], label='Predicted Price', marker='X', markersize=7, color='crimson', linestyle='--', linewidth=2)
        plt.title(f'Tracking Validity over Time: {sample_veg.upper()} at {sample_market.capitalize()}', fontsize=14, fontweight='bold')
        plt.xlabel('Time (Year-Week)', fontsize=12)
        plt.ylabel('Price in LKR', fontsize=12)
        
        n_ticks = len(df_sample['Year_Week'])
        step = max(1, n_ticks // 15)
        
        plt.xticks(df_sample['Year_Week'].values[::step], rotation=45, fontsize=10)
        
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='dotted', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'TimeSeries_2024_Test_{sample_veg}.png'), dpi=300)
        plt.close()
    else:
        print(f"Warning: No overlapping data available for {sample_veg} at {sample_market} to plot line chart.")

    print(f"\nSuccessfully generated Validation Reports and Visualizations in:\n{output_dir}")

if __name__ == '__main__':
    main()
