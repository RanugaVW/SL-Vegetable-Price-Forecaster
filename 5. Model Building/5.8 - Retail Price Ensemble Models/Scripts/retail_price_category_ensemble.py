import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import os
import joblib


def train_category_ensemble():
    data_path  = r'C:\Users\Ranuga\Data Science Project\Final Complete Dataset\Data Pre Process\Combining DataSets\Final_Combined_data.csv'
    output_dir = r'C:\Users\Ranuga\Data Science Project\Final Complete Dataset\Model Building'
    os.makedirs(output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Load Data
    # ─────────────────────────────────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(data_path)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Category Mapping (Up Country vs Low Country)
    # ─────────────────────────────────────────────────────────────────────────
    up_country_veggies = ['BEETROOT', 'CABBAGE', 'CARROT', 'LEEKS', 'GREEN BEANS']
    low_country_veggies = ['ASH PLANTAINS', 'BRINJALS', 'GREEN CHILLIES', 'LADIES FINGERS', 'PUMPKIN', 'SNAKE GOURD', 'TOMATOES']

    def map_category(veg):
        v = str(veg).strip().upper()
        if v in up_country_veggies:
            return 'Up Country'
        elif v in low_country_veggies:
            return 'Low Country'
        else:
            return 'Other'

    df['veg_category'] = df['vegetable_type'].apply(map_category)
    print(f"  Vegetable Categories: {df['veg_category'].value_counts().to_dict()}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Simple Feature Engineering
    # ─────────────────────────────────────────────────────────────────────────
    # Cyclic week
    df['week_num'] = pd.to_numeric(df['week'].str.extract(r'(\d+)')[0])
    df['week_sin'] = np.sin(2 * np.pi * df['week_num'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_num'] / 52)

    # Regional rain (Year_Week-level aggregate)
    regional_rain = df.groupby(['Year_Week', 'vegetable_zone'])['rain_sum'].mean().reset_index().rename(columns={'rain_sum': 'reg_rain'})
    df = pd.merge(df, regional_rain, on=['Year_Week', 'vegetable_zone'], how='left')

    # Sort and Lag
    df = df.sort_values(['retail_market', 'vegetable_type', 'year', 'week_num'])

    # ── Focus: 4-week rainfall lag ──
    df['reg_rain_lag_4'] = df.groupby(['retail_market', 'vegetable_type'])['reg_rain'].shift(4)

    # ── Focus: Primary Farmer Price features (Lagged & Rolling) ──
    df['farmer_price_lag_1'] = df.groupby(['retail_market', 'vegetable_type'])['mean_farmer_price'].shift(1)
    df['farmer_price_lag_4'] = df.groupby(['retail_market', 'vegetable_type'])['mean_farmer_price'].shift(4)
    # Strongest signal: current-week farmer price (imputed with lag_1)
    df['farmer_price_filled'] = df['mean_farmer_price'].fillna(df['farmer_price_lag_1'])
    # Retail price lag
    df['retail_price_lag_1'] = df.groupby(['retail_market', 'vegetable_type'])['retail_price'].shift(1)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Preparation
    # ─────────────────────────────────────────────────────────────────────────
    # Drop rows without the 4-week lag
    df_ready = df.dropna(subset=['reg_rain_lag_4', 'farmer_price_lag_4', 'retail_price_lag_1']).copy()

    # Encode features
    le_cat = LabelEncoder()
    df_ready['veg_category_enc'] = le_cat.fit_transform(df_ready['veg_category'])
    
    le_veg = LabelEncoder()
    df_ready['vegetable_type_enc'] = le_veg.fit_transform(df_ready['vegetable_type'])

    le_mkt = LabelEncoder()
    df_ready['retail_market_enc'] = le_mkt.fit_transform(df_ready['retail_market'])

    # Simple Feature List
    features = [
        'year', 'week_sin', 'week_cos',
        'veg_category_enc', 'vegetable_type_enc', 'retail_market_enc',
        'farmer_price_filled', 'farmer_price_lag_1', 'farmer_price_lag_4',
        'reg_rain_lag_4', 'retail_price_lag_1'
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Split (80/20 Time-ordered)
    # ─────────────────────────────────────────────────────────────────────────
    train_list, test_list = [], []
    for _, group in df_ready.groupby(['retail_market', 'vegetable_type']):
        split = int(len(group) * 0.8)
        train_list.append(group.iloc[:split])
        test_list.append(group.iloc[split:])

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    X_train, y_train = train_df[features], train_df['retail_price']
    X_test, y_test = test_df[features], test_df['retail_price']

    print(f"\n  Final Features ({len(features)}): {features}")
    print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Training (XGBoost + LightGBM only)
    # ─────────────────────────────────────────────────────────────────────────
    print("\nTraining Models...")

    # Model 1: XGBoost
    print("  [1/2] XGBoost...")
    m_xgb = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, early_stopping_rounds=50)
    m_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Model 2: LightGBM
    print("  [2/2] LightGBM...")
    m_lgb = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
    m_lgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50, verbose=False)])

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Ensemble & Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    p_xgb = m_xgb.predict(X_test)
    p_lgb = m_lgb.predict(X_test)
    y_pred = (0.5 * p_xgb) + (0.5 * p_lgb)

    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    report = f"""Category-Based (Up/Low Country) Ensemble Report
=================================================
Target  : retail_price
Design  : Simplest XGBoost + LightGBM (50/50)
Grouping: veg_category (Up Country vs Low Country)

Top-Weighted Feature: farmer_price_filled
Primary Weather Feed: reg_rain_lag_4 (4-week lag)

Final Performance:
------------------
  R2 Score  : {r2:.4f}
  Accuracy  : {(1 - mape)*100:.2f}%
  MAPE      : {mape*100:.2f}%

Features Used:
--------------
{features}
"""
    print("\n" + report)

    # Save
    report_path = os.path.join(output_dir, 'retail_category_ensemble_performance.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    bundle = {
        'xgb': m_xgb,
        'lgb': m_lgb,
        'features': features,
        'encoders': {'category': le_cat, 'vegetable': le_veg, 'market': le_mkt}
    }
    joblib.dump(bundle, os.path.join(output_dir, 'retail_category_ensemble_model.joblib'))
    print(f"Success! Model bundle saved to {output_dir}")


if __name__ == "__main__":
    train_category_ensemble()
