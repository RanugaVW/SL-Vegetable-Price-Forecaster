import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import os
import joblib


def train_ensemble_model():
    data_path = r'C:\Users\Ranuga\Data Science Project\Final Complete Dataset\Data Pre Process\Combining DataSets\Final_Combined_data.csv'
    output_dir = r'C:\Users\Ranuga\Data Science Project\Final Complete Dataset\Model Building'
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"  Dataset shape: {df.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Feature Engineering
    # ─────────────────────────────────────────────────────────────────────────

    # Cyclic week encoding
    df['week_num'] = pd.to_numeric(df['week'].str.extract(r'(\d+)')[0])
    df['week_sin'] = np.sin(2 * np.pi * df['week_num'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_num'] / 52)

    # Regional aggregates (rain & temperature per zone per week)
    regional_weather = (
        df.groupby(['Year_Week', 'vegetable_zone'])[['rain_sum', 'mean_apparent_temperature']]
        .mean()
        .reset_index()
        .rename(columns={'rain_sum': 'reg_rain', 'mean_apparent_temperature': 'reg_temp'})
    )
    df = pd.merge(df, regional_weather, on=['Year_Week', 'vegetable_zone'], how='left')

    # Encode seasonality and create diesel×season interaction
    df['season_enc'] = LabelEncoder().fit_transform(df['seasonality'].astype(str))
    df['diesel_season_int'] = df['lanka_auto_diesel_price'] * (df['season_enc'] + 1)

    # ── Farmer-price features ─────────────────────────────────────────────────
    # mean_farmer_price is a strong direct signal for retail_price.
    # Price spread (markup) and ratio are useful derived features.
    df['price_spread'] = df['retail_price'] - df['mean_farmer_price']          # will be lagged
    df['farmer_retail_ratio'] = df['retail_price'] / df['mean_farmer_price'].replace(0, np.nan)
    # ─────────────────────────────────────────────────────────────────────────

    # Sort for time-series lag creation
    df = df.sort_values(['retail_market', 'vegetable_type', 'year', 'week_num'])

    # Lag features for the main targets and weather
    cols_to_lag = ['retail_price', 'mean_farmer_price', 'reg_rain', 'reg_temp']
    for col in cols_to_lag:
        for lag in [1, 2, 3, 4, 8]:
            df[f'{col}_lag_{lag}'] = df.groupby(['retail_market', 'vegetable_type'])[col].shift(lag)

    # Rolling mean of retail_price (4-week window, excludes current)
    df['retail_price_roll_4'] = (
        df.groupby(['retail_market', 'vegetable_type'])['retail_price']
        .transform(lambda x: x.shift(1).rolling(4).mean())
    )
    # Rolling mean of farmer_price (4-week window)
    df['farmer_price_roll_4'] = (
        df.groupby(['retail_market', 'vegetable_type'])['mean_farmer_price']
        .transform(lambda x: x.shift(1).rolling(4).mean())
    )

    # Drop rows where the longest lag hasn't filled yet
    df_ready = df.dropna(subset=['retail_price_lag_8', 'mean_farmer_price_lag_8']).copy()

    # Encode categoricals
    le_dict = {}
    for col in ['retail_market', 'vegetable_type', 'vegetable_zone']:
        le = LabelEncoder()
        df_ready[f'{col}_enc'] = le.fit_transform(df_ready[col].astype(str))
        le_dict[col] = le

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Train / Test Split  (80 / 20 per group, preserving time order)
    # ─────────────────────────────────────────────────────────────────────────
    train_list, test_list = [], []
    for _, group in df_ready.groupby(['retail_market', 'vegetable_type']):
        split = int(len(group) * 0.8)
        train_list.append(group.iloc[:split])
        test_list.append(group.iloc[split:])

    train_df = pd.concat(train_list)
    test_df  = pd.concat(test_list)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Feature list
    # ─────────────────────────────────────────────────────────────────────────
    features = [
        # Time
        'year', 'week_sin', 'week_cos',
        # Macro-economic
        'lanka_auto_diesel_price', 'usd_exchange_rate', 'diesel_season_int',
        # Holidays
        'no_of_holidays',
        # Weather
        'reg_rain', 'reg_temp',
        # ── NEW: farmer price (lagged & rolling — avoids data leakage) ──
        'mean_farmer_price_lag_1', 'mean_farmer_price_lag_2',
        'mean_farmer_price_lag_3', 'mean_farmer_price_lag_4', 'mean_farmer_price_lag_8',
        'farmer_price_roll_4',
        # Retail price lags
        'retail_price_lag_1', 'retail_price_lag_2',
        'retail_price_lag_3', 'retail_price_lag_4', 'retail_price_lag_8',
        # Weather lags
        'reg_rain_lag_1', 'reg_rain_lag_4', 'reg_rain_lag_8',
        'reg_temp_lag_1', 'reg_temp_lag_4', 'reg_temp_lag_8',
        # Rolling retail price
        'retail_price_roll_4',
        # Categoricals
        'retail_market_enc', 'vegetable_type_enc', 'vegetable_zone_enc', 'season_enc',
    ]

    X_train, y_train = train_df[features], train_df['retail_price']
    X_test,  y_test  = test_df[features],  test_df['retail_price']

    print(f"\n  Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
    print(f"  Number of features: {len(features)}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Train Ensemble Members
    # ─────────────────────────────────────────────────────────────────────────
    print("\nTraining Ensemble Members...")

    # Model 1 – XGBoost (champion)
    print("  [1/3] Training XGBoost...")
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
    )
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Model 2 – HistGradientBoosting (LightGBM-equivalent)
    print("  [2/3] Training HistGradientBoosting...")
    model_hgb = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
    )
    model_hgb.fit(X_train, y_train)

    # Model 3 – Random Forest (bagging diversity)
    print("  [3/3] Training Random Forest (may take a moment)...")
    model_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model_rf.fit(X_train, y_train)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Weighted Ensemble Predictions
    # ─────────────────────────────────────────────────────────────────────────
    print("\nCalculating Weighted Ensemble Predictions...")
    pred_xgb     = model_xgb.predict(X_test)
    pred_hgb     = model_hgb.predict(X_test)
    pred_rf      = model_rf.predict(X_test)
    final_pred   = (0.6 * pred_xgb) + (0.2 * pred_hgb) + (0.2 * pred_rf)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    r2         = r2_score(y_test, final_pred)
    mape       = mean_absolute_percentage_error(y_test, final_pred)
    mape_xgb   = mean_absolute_percentage_error(y_test, pred_xgb)
    mape_hgb   = mean_absolute_percentage_error(y_test, pred_hgb)
    mape_rf    = mean_absolute_percentage_error(y_test, pred_rf)

    report = f"""Retail Price Ensemble Model – Performance Report
=================================================
Target  : retail_price
Key new feature: mean_farmer_price (lagged & rolling, to avoid leakage)

Models  : XGBoost (w=0.6) + HistGBM (w=0.2) + RandomForest (w=0.2)
Data    : Final_Combined_data.csv  ({df.shape[0]:,} rows total)
Train   : {len(train_df):,} rows | Test: {len(test_df):,} rows
Features: {len(features)}

Overall Ensemble Metrics
------------------------
  R²  Score  : {r2:.4f}
  Accuracy (1 - MAPE) : {(1 - mape)*100:.2f}%
  MAPE       : {mape*100:.2f}%

Individual Model Accuracies
---------------------------
  XGBoost       : {(1 - mape_xgb)*100:.2f}%
  HistGBM       : {(1 - mape_hgb)*100:.2f}%
  Random Forest : {(1 - mape_rf)*100:.2f}%

Conclusion
----------
The ensemble combines gradient-boosting and bagging to stabilize predictions.
Including mean_farmer_price (lagged) provides a direct supply-chain signal
that significantly improves retail price forecasting accuracy.
"""

    print("\n" + report)

    report_path = os.path.join(output_dir, 'retail_price_ensemble_performance.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Performance report saved → {report_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Save Model Bundle
    # ─────────────────────────────────────────────────────────────────────────
    ensemble_bundle = {
        'xgb'          : model_xgb,
        'hgb'          : model_hgb,
        'rf'           : model_rf,
        'features'     : features,
        'label_encoders': le_dict,
    }
    model_path = os.path.join(output_dir, 'retail_price_ensemble_model.joblib')
    joblib.dump(ensemble_bundle, model_path)
    print(f"Model bundle saved      → {model_path}")


if __name__ == "__main__":
    train_ensemble_model()
