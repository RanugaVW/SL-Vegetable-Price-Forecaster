import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder

def visualize_all():
    data_path  = r'C:\Users\Ranuga\Data Science Project\3. Data Preprocessing\3.7 - Combining Datasets\Outputs\Final_Combined_data.csv'
    output_dir = r'C:\Users\Ranuga\Data Science Project\5. Model Building\5.8 - Retail Price Ensemble Models\XGBoost + LightBGM'
    
    # WE MUST USE THE REGULAR OPTUNA MODEL
    model_path = os.path.join(output_dir, 'Models', 'xgb_lgbm_ensemble_optuna_model.joblib')
    chart_dir = os.path.join(output_dir, 'Charts')
    os.makedirs(chart_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}.")
        return

    print("Reconstructing Test Set exactly as trained...")
    df = pd.read_csv(data_path)
    df.drop(columns=['code'], inplace=True, errors='ignore')

    df['week_num'] = pd.to_numeric(df['week'].str.extract(r'(\d+)')[0])
    df['week_sin'] = np.sin(2 * np.pi * df['week_num'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_num'] / 52)

    regional_weather = (
        df.groupby(['Year_Week', 'vegetable_zone'])[['rain_sum', 'mean_apparent_temperature']]
        .mean().reset_index()
        .rename(columns={'rain_sum': 'reg_rain', 'mean_apparent_temperature': 'reg_temp'})
    )
    df = pd.merge(df, regional_weather, on=['Year_Week', 'vegetable_zone'], how='left')
    df.drop(columns=['Year_Week'], inplace=True, errors='ignore')

    df['season_enc'] = LabelEncoder().fit_transform(df['seasonality'].astype(str))
    df['diesel_season_int'] = df['lanka_auto_diesel_price'] * (df['season_enc'] + 1)
    df = df.sort_values(['retail_market', 'vegetable_type', 'year', 'week_num'])

    for col in ['retail_price', 'reg_rain', 'reg_temp']:
        for lag in [1, 2, 3, 4, 8]:
            df[f'{col}_lag_{lag}'] = df.groupby(['retail_market', 'vegetable_type'])[col].shift(lag)

    for lag in [1, 2, 3, 4, 5, 6, 8]:
        df[f'mean_farmer_price_lag_{lag}'] = df.groupby(['retail_market', 'vegetable_type'])['mean_farmer_price'].shift(lag)

    df['retail_price_roll_4'] = df.groupby(['retail_market', 'vegetable_type'])['retail_price'].transform(lambda x: x.shift(1).rolling(4).mean())
    grp = df.groupby(['retail_market', 'vegetable_type'])['mean_farmer_price']
    df['farmer_price_roll_4'] = grp.transform(lambda x: x.shift(1).rolling(4).mean())
    df['farmer_price_roll_8'] = grp.transform(lambda x: x.shift(1).rolling(8).mean())
    df['farmer_price_roll_std_4'] = grp.transform(lambda x: x.shift(1).rolling(4).std())
    df['farmer_price_pct_change_1'] = grp.transform(lambda x: x.shift(1).pct_change(1, fill_method=None))

    df['mean_farmer_price_filled'] = df['mean_farmer_price'].fillna(df['mean_farmer_price_lag_1'])
    df['farmer_retail_spread_lag_1'] = df['retail_price_lag_1'] - df['mean_farmer_price_lag_1']

    df_ready = df.dropna(subset=['retail_price_lag_8', 'mean_farmer_price_lag_8', 'farmer_price_roll_8']).copy()

    for col in ['retail_market', 'vegetable_type', 'vegetable_zone']:
        df_ready[f'{col}_enc'] = LabelEncoder().fit_transform(df_ready[col].astype(str))

    test_list = []
    for _, group in df_ready.groupby(['retail_market', 'vegetable_type']):
        split = int(len(group) * 0.8)
        test_list.append(group.iloc[split:])
    test_df = pd.concat(test_list)

    print("Loading Trained Saved Model...")
    bundle = joblib.load(model_path)
    X_test = test_df[bundle['features']]
    y_test = test_df['retail_price'].values
    
    print("Executing Predictions...")
    pred_xgb = bundle['xgb'].predict(X_test)
    pred_lgb = bundle['lgb'].predict(X_test)
    
    # NO expm1. Predictions are generated securely through mapped model parameters
    final_preds = (bundle['weights']['xgb'] * pred_xgb) + (bundle['weights']['lgb'] * pred_lgb)
    test_df['predicted_price'] = final_preds

    # =====================================================================
    # Plot 1: Scatter Graph
    # =====================================================================
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, final_preds, alpha=0.5, color='teal', edgecolor='k', marker='o', label='Actual vs Predicted (Dots)')
    min_val = min(np.min(y_test), np.min(final_preds))
    max_val = max(np.max(y_test), np.max(final_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect Accuracy Line')
    plt.title('Model Accuracy Distribution (Test Data)', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Retail Price (LKR)', fontsize=12)
    plt.ylabel('Predicted Retail Price (LKR)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    chart1_path = os.path.join(chart_dir, 'Scatter_Accuracy_Test_Data.png')
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {chart1_path}")

    # =====================================================================
    # Plot 2: Validation Time-Series Tracking
    # =====================================================================
    top_combo = test_df.groupby(['retail_market', 'vegetable_type']).size().idxmax()
    sample_df = test_df[(test_df['retail_market'] == top_combo[0]) & (test_df['vegetable_type'] == 'BEETROOT')]
    if len(sample_df) == 0:
        sample_df = test_df[(test_df['retail_market'] == top_combo[0]) & (test_df['vegetable_type'] == top_combo[1])]

    sample_df = sample_df.sort_values(['year', 'week_num'])
    sample_df['Time'] = sample_df['year'].astype(str) + '-W' + sample_df['week_num'].astype(str).str.zfill(2)

    plt.figure(figsize=(14, 6))
    plt.plot(sample_df['Time'], sample_df['retail_price'], 
             marker='o', markersize=6, linestyle='-', color='dodgerblue', linewidth=2.5, label='Actual Price')
    plt.plot(sample_df['Time'], sample_df['predicted_price'], 
             marker='X', markersize=8, linestyle='--', color='crimson', linewidth=2.5, label='Predicted Price')
    
    plt.title(f'Tracking Validity over Time: {sample_df["vegetable_type"].iloc[0]} at {sample_df["retail_market"].iloc[0]}', fontsize=14, fontweight='bold')
    plt.xlabel('Time (Year-Week)', fontsize=12)
    plt.ylabel('Price in LKR', fontsize=12)
    
    ticks = np.arange(0, len(sample_df), max(1, len(sample_df)//15))
    plt.xticks(ticks, sample_df['Time'].iloc[ticks], rotation=45)
    
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    chart2_path = os.path.join(chart_dir, 'Validation_TimeSeries_Tracking.png')
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {chart2_path}")

    # =====================================================================
    # Plot 3: ALL Vegetables
    # =====================================================================
    anu_df = test_df[test_df['retail_market'] == 'Anuradhapura'].copy()
    anu_df = anu_df.sort_values(['vegetable_type', 'year', 'week_num'])
    anu_df['Time'] = anu_df['year'].astype(str) + '-W' + anu_df['week_num'].astype(str).str.zfill(2)

    vegetables = anu_df['vegetable_type'].unique()
    num_vegs = len(vegetables)
    cols = 2
    rows = math.ceil(num_vegs / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6 * rows))
    fig.suptitle('Validation Tracking Over Time: ALL Vegetables at Anuradhapura', fontsize=22, fontweight='bold', y=0.99)
    axes = axes.flatten()

    for i, veg in enumerate(vegetables):
        ax = axes[i]
        v_df = anu_df[anu_df['vegetable_type'] == veg]
        ax.plot(v_df['Time'], v_df['retail_price'], marker='o', markersize=4, linestyle='-', color='dodgerblue', linewidth=2, label='Actual Price')
        ax.plot(v_df['Time'], v_df['predicted_price'], marker='X', markersize=5, linestyle='--', color='crimson', linewidth=2, label='Predicted Price')
        ax.set_title(veg, fontsize=16, fontweight='bold')
        ax.set_ylabel('Price in LKR', fontsize=12)
        ticks = np.arange(0, len(v_df), max(1, len(v_df)//12))
        ax.set_xticks(ticks)
        ax.set_xticklabels(v_df['Time'].iloc[ticks], rotation=45)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    chart3_path = os.path.join(chart_dir, 'Anuradhapura_All_Vegetables_Validation.png')
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {chart3_path}")

if __name__ == '__main__':
    visualize_all()