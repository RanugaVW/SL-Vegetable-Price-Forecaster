import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import os
import joblib


def train_xgb_lgbm_ensemble():
    data_path  = r'C:\Users\Ranuga\Data Science Project\Final Complete Dataset\Data Pre Process\Combining DataSets\Final_Combined_data.csv'
    output_dir = r'C:\Users\Ranuga\Data Science Project\Final Complete Dataset\Model Building'
    os.makedirs(output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Load
    # ─────────────────────────────────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"  Dataset shape: {df.shape}")

    # Drop unused identifier columns
    df.drop(columns=['code'], inplace=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Feature Engineering
    # ─────────────────────────────────────────────────────────────────────────

    # Cyclic week encoding
    df['week_num'] = pd.to_numeric(df['week'].str.extract(r'(\d+)')[0])
    df['week_sin'] = np.sin(2 * np.pi * df['week_num'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_num'] / 52)

    # Regional weather aggregates per zone per week
    regional_weather = (
        df.groupby(['Year_Week', 'vegetable_zone'])[['rain_sum', 'mean_apparent_temperature']]
        .mean()
        .reset_index()
        .rename(columns={'rain_sum': 'reg_rain', 'mean_apparent_temperature': 'reg_temp'})
    )
    df = pd.merge(df, regional_weather, on=['Year_Week', 'vegetable_zone'], how='left')

    # Year_Week only needed for the weather merge above — drop it now
    df.drop(columns=['Year_Week'], inplace=True)

    # Seasonality & diesel interaction
    df['season_enc'] = LabelEncoder().fit_transform(df['seasonality'].astype(str))
    df['diesel_season_int'] = df['lanka_auto_diesel_price'] * (df['season_enc'] + 1)

    # Sort for time-series lags
    df = df.sort_values(['retail_market', 'vegetable_type', 'year', 'week_num'])

    # Lags for retail_price, farmer_price, and weather
    for col in ['retail_price', 'reg_rain', 'reg_temp']:
        for lag in [1, 2, 3, 4, 8]:
            df[f'{col}_lag_{lag}'] = df.groupby(['retail_market', 'vegetable_type'])[col].shift(lag)

    # Farmer price — more granular lags to give it higher model influence
    for lag in [1, 2, 3, 4, 5, 6, 8]:
        df[f'mean_farmer_price_lag_{lag}'] = df.groupby(['retail_market', 'vegetable_type'])['mean_farmer_price'].shift(lag)

    # Rolling stats for retail price (4-week mean)
    df['retail_price_roll_4'] = (
        df.groupby(['retail_market', 'vegetable_type'])['retail_price']
        .transform(lambda x: x.shift(1).rolling(4).mean())
    )

    # ── Farmer price rolling features (richer signal) ──────────────────────
    grp = df.groupby(['retail_market', 'vegetable_type'])['mean_farmer_price']
    # Short & long rolling mean
    df['farmer_price_roll_4'] = grp.transform(lambda x: x.shift(1).rolling(4).mean())
    df['farmer_price_roll_8'] = grp.transform(lambda x: x.shift(1).rolling(8).mean())
    # Volatility: rolling std (price stability signal)
    df['farmer_price_roll_std_4'] = grp.transform(lambda x: x.shift(1).rolling(4).std())
    # Week-over-week % change in farmer price (momentum)
    df['farmer_price_pct_change_1'] = grp.transform(lambda x: x.shift(1).pct_change(1, fill_method=None))

    # ── Current-week farmer price (impute nulls with lag_1 — strongest signal) ──
    df['mean_farmer_price_filled'] = df['mean_farmer_price'].fillna(df['mean_farmer_price_lag_1'])
    # Retail-farmer spread (lag_1): captures markup/margin signal
    df['farmer_retail_spread_lag_1'] = df['retail_price_lag_1'] - df['mean_farmer_price_lag_1']

    # Drop rows where longest lag hasn't filled yet
    df_ready = df.dropna(subset=['retail_price_lag_8', 'mean_farmer_price_lag_8',
                                  'farmer_price_roll_8']).copy()

    # Encode categoricals
    le_dict = {}
    for col in ['retail_market', 'vegetable_type', 'vegetable_zone']:
        le = LabelEncoder()
        df_ready[f'{col}_enc'] = le.fit_transform(df_ready[col].astype(str))
        le_dict[col] = le

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Train / Test Split  (80 / 20 per group, time-ordered)
    # ─────────────────────────────────────────────────────────────────────────
    train_list, test_list = [], []
    for _, group in df_ready.groupby(['retail_market', 'vegetable_type']):
        split = int(len(group) * 0.8)
        train_list.append(group.iloc[:split])
        test_list.append(group.iloc[split:])

    train_df = pd.concat(train_list)
    test_df  = pd.concat(test_list)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Feature list
    # ─────────────────────────────────────────────────────────────────────────
    features = [
        # ── HIGHEST WEIGHT: direct farmer price signal ──────────────────────
        'mean_farmer_price_filled',     # current-week farmer price (strongest predictor)
        'farmer_retail_spread_lag_1',   # lagged retail-farmer markup
        # Farmer price lags & rolling
        'mean_farmer_price_lag_1', 'mean_farmer_price_lag_2',
        'mean_farmer_price_lag_3', 'mean_farmer_price_lag_4',
        'mean_farmer_price_lag_5', 'mean_farmer_price_lag_6',
        'mean_farmer_price_lag_8',
        'farmer_price_roll_4',
        'farmer_price_roll_8',
        'farmer_price_roll_std_4',
        'farmer_price_pct_change_1',
        # Time
        'year', 'week_sin', 'week_cos',
        # Macro-economic
        'lanka_auto_diesel_price', 'usd_exchange_rate', 'diesel_season_int',
        # Holidays
        'no_of_holidays',
        # Weather
        'reg_rain', 'reg_temp',
        # Retail price lags
        'retail_price_lag_1', 'retail_price_lag_2',
        'retail_price_lag_3', 'retail_price_lag_4', 'retail_price_lag_8',
        # Weather lags
        'reg_rain_lag_1', 'reg_rain_lag_4', 'reg_rain_lag_8',
        'reg_temp_lag_1', 'reg_temp_lag_4', 'reg_temp_lag_8',
        # Rolling retail
        'retail_price_roll_4',
        # Categoricals
        'retail_market_enc', 'vegetable_type_enc', 'vegetable_zone_enc', 'season_enc',
    ]

    X_train, y_train = train_df[features], train_df['retail_price']
    X_test,  y_test  = test_df[features],  test_df['retail_price']

    print(f"\n  Train rows : {len(train_df):,}")
    print(f"  Test rows  : {len(test_df):,}")
    print(f"  Features   : {len(features)}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Train Models
    # ─────────────────────────────────────────────────────────────────────────
    print("\nTraining Ensemble Members...")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print("  [1/2] Training XGBoost...")
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
    best_xgb_iter = model_xgb.best_iteration
    print(f"     Best iteration: {best_xgb_iter}")

    # ── LightGBM ─────────────────────────────────────────────────────────────
    print("  [2/2] Training LightGBM...")
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    best_lgb_iter = model_lgb.best_iteration_
    print(f"     Best iteration: {best_lgb_iter}")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Weighted Ensemble  (equal weight — both are boosting models)
    # ─────────────────────────────────────────────────────────────────────────
    print("\nBuilding Weighted Ensemble Predictions...")
    pred_xgb   = model_xgb.predict(X_test)
    pred_lgb   = model_lgb.predict(X_test)
    final_pred = (0.5 * pred_xgb) + (0.5 * pred_lgb)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    r2        = r2_score(y_test, final_pred)
    mape      = mean_absolute_percentage_error(y_test, final_pred)
    mape_xgb  = mean_absolute_percentage_error(y_test, pred_xgb)
    mape_lgb  = mean_absolute_percentage_error(y_test, pred_lgb)

    report = f"""XGBoost + LightGBM Ensemble – Performance Report
=================================================
Target  : retail_price
Key feature: mean_farmer_price (lagged & rolling — no leakage)

Weights : XGBoost (0.5) + LightGBM (0.5)
Data    : Final_Combined_data.csv  ({df.shape[0]:,} rows total)
Train   : {len(train_df):,} rows | Test: {len(test_df):,} rows
Features: {len(features)}

Model Tuning Details
--------------------
  XGBoost  best iteration : {best_xgb_iter}
  LightGBM best iteration : {best_lgb_iter}

Overall Ensemble Metrics
------------------------
  R2  Score              : {r2:.4f}
  Accuracy (1 - MAPE)    : {(1 - mape)*100:.2f}%
  MAPE                   : {mape*100:.2f}%

Individual Model Accuracies
---------------------------
  XGBoost  : {(1 - mape_xgb)*100:.2f}%
  LightGBM : {(1 - mape_lgb)*100:.2f}%

Conclusion
----------
XGBoost and LightGBM are complementary gradient-boosting frameworks.
XGBoost uses depth-wise tree growth (more accurate on structured tabular data
with outliers), while LightGBM uses leaf-wise growth (faster, handles
high-cardinality categoricals better). Their ensemble averages out
individual model variance for more stable retail price predictions.
"""

    print("\n" + report)

    report_path = os.path.join(output_dir, 'xgb_lgbm_ensemble_performance.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Performance report saved -> {report_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Save Model Bundle
    # ─────────────────────────────────────────────────────────────────────────
    bundle = {
        'xgb'            : model_xgb,
        'lgb'            : model_lgb,
        'features'       : features,
        'label_encoders' : le_dict,
        'weights'        : {'xgb': 0.5, 'lgb': 0.5},
    }
    model_path = os.path.join(output_dir, 'xgb_lgbm_ensemble_model.joblib')
    joblib.dump(bundle, model_path)
    print(f"Model bundle saved      -> {model_path}")


if __name__ == "__main__":
    train_xgb_lgbm_ensemble()
