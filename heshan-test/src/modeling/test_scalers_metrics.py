import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
import xgboost as xgb

TRAIN_YEARS = [2013, 2014, 2015, 2016, 2017]
TEST_YEARS  = [2018, 2019]

df = pd.read_csv('data/main_filled.csv')
if 'week' not in df.columns:
    df['week'] = df['Year_Week'].str.extract(r'-w(\d+)')[0]

df['week'] = df['week'].astype(str)
df['week_num'] = df['week'].str.extract(r'(\d+)').astype(float).fillna(1).astype(int)

df['week_start'] = pd.to_datetime(df['year'].astype(str) + '-01-01') + pd.to_timedelta((df['week_num'] - 1) * 7, unit='D')

for col in ['vegetable_type','location','vegetable_zone','seasonality']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

df_clean = df.dropna(subset=['price']).copy()
df_clean = df_clean.sort_values(['vegetable_type','location','week_start']).reset_index(drop=True)

for col in ['rain_sum', 'mean_apparent_temperature', 'lanka_auto_diesel_price']:
    if col not in df_clean.columns: df_clean[col] = 0

df_clean['rain_lag3']  = df_clean.groupby(['vegetable_type','location'])['rain_sum'].shift(3)
df_clean['rain_lag6']  = df_clean.groupby(['vegetable_type','location'])['rain_sum'].shift(6)
df_clean['temp_lag6']  = df_clean.groupby(['vegetable_type','location'])['mean_apparent_temperature'].shift(6)
df_clean['price_lag1'] = df_clean.groupby(['vegetable_type','location'])['price'].shift(1)
df_clean['price_lag4'] = df_clean.groupby(['vegetable_type','location'])['price'].shift(4)

df_clean['is_upcountry']    = (df_clean.get('vegetable_zone', '') == 'UP Country').astype(int)
df_clean['is_maha']         = (df_clean.get('seasonality', '') == 'Maha Season').astype(int)

df_clean = df_clean.fillna(0)

FEATURES = [
    'week_num', 'is_maha','is_upcountry', 'lanka_auto_diesel_price',
    'mean_apparent_temperature','rain_sum',
    'rain_lag3', 'rain_lag6', 'temp_lag6', 'price_lag1', 'price_lag4'
]
TARGET = 'price'

FEATURES = [f for f in FEATURES if f in df_clean.columns]

train = df_clean[df_clean['year'].isin(TRAIN_YEARS)]
test  = df_clean[df_clean['year'].isin(TEST_YEARS)]

if len(test) == 0:
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df_clean, test_size=0.2, random_state=42)

scalers = {
    'None': 'passthrough',
    'Standard': StandardScaler(),
    'MinMax': MinMaxScaler(),
    'Robust': RobustScaler(),
    'MaxAbs': MaxAbsScaler(),
    'Quantile': QuantileTransformer(),
    'Power': PowerTransformer()
}

results = []

for name, scaler in scalers.items():
    X_train = train[FEATURES].copy()
    X_test  = test[FEATURES].copy()
    y_train = train[TARGET].copy()
    y_test  = test[TARGET].copy()

    if name != 'None':
        X_train[FEATURES] = scaler.fit_transform(X_train[FEATURES])
        X_test[FEATURES] = scaler.transform(X_test[FEATURES])
    
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    
    preds = xgb_model.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - preds[mask]) / y_test[mask])) * 100
    
    results.append((name, r2, mae, rmse, mape))

print("\n--- Extended Metrics (XGBoost) ---")
print(f"{'Scaler':<15} | {'R2 Score':<10} | {'MAE (Rs)':<10} | {'RMSE (Rs)':<10} | {'MAPE (%)':<10}")
print("-" * 65)
results.sort(key=lambda x: x[1], reverse=True)
for r in results:
    print(f"{r[0]:<15} | {r[1]:<10.4f} | {r[2]:<10.2f} | {r[3]:<10.2f} | {r[4]:<10.2f}")
