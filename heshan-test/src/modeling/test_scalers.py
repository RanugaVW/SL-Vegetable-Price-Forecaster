import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
import xgboost as xgb

TRAIN_YEARS = [2013, 2014, 2015, 2016, 2017]
TEST_YEARS  = [2018, 2019]

# Use main_filled.csv instead of Kaggle path
df = pd.read_csv('data/main_filled.csv')
if 'week' not in df.columns:
    df['week'] = df['Year_Week'].str.extract(r'-w(\d+)')[0]

df['week'] = df['week'].astype(str)
df['week_num'] = df['week'].str.extract(r'(\d+)').astype(float).fillna(1).astype(int)

# Create arbitrary dates for grouping if needed
df['week_start'] = pd.to_datetime(df['year'].astype(str) + '-01-01') + pd.to_timedelta((df['week_num'] - 1) * 7, unit='D')

for col in ['vegetable_type','location','vegetable_zone','seasonality']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Simplify missing handling for this test to avoid structural null loop complexity
df_clean = df.dropna(subset=['price']).copy()

# Add Lag features (using the pipeline logic)
df_clean = df_clean.sort_values(['vegetable_type','location','week_start']).reset_index(drop=True)

for col in ['rain_sum', 'mean_apparent_temperature', 'lanka_auto_diesel_price']:
    if col not in df_clean.columns: df_clean[col] = 0

df_clean['rain_lag3']  = df_clean.groupby(['vegetable_type','location'])['rain_sum'].shift(3)
df_clean['rain_lag6']  = df_clean.groupby(['vegetable_type','location'])['rain_sum'].shift(6)
df_clean['temp_lag6']  = df_clean.groupby(['vegetable_type','location'])['mean_apparent_temperature'].shift(6)
df_clean['price_lag1'] = df_clean.groupby(['vegetable_type','location'])['price'].shift(1)
df_clean['price_lag4'] = df_clean.groupby(['vegetable_type','location'])['price'].shift(4)

df_clean['is_upcountry']    = (df_clean['vegetable_zone'] == 'UP Country').astype(int)
df_clean['is_maha']         = (df_clean['seasonality'] == 'Maha Season').astype(int)

# Fill nas from lag shifts
df_clean = df_clean.fillna(0)

FEATURES = [
    'week_num', 'is_maha','is_upcountry', 'lanka_auto_diesel_price',
    'mean_apparent_temperature','rain_sum',
    'rain_lag3', 'rain_lag6', 'temp_lag6', 'price_lag1', 'price_lag4'
]
TARGET = 'price'

# Filter available features
FEATURES = [f for f in FEATURES if f in df_clean.columns]

train = df_clean[df_clean['year'].isin(TRAIN_YEARS)]
test  = df_clean[df_clean['year'].isin(TEST_YEARS)]

# If some test years are empty, just use train_test_split as fallback
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

results_rf = {}
results_xgb = {}

print(f'Training on {len(train)} rows, Testing on {len(test)} rows...')
print('Evaluating Normalization Methods...\n')

for name, scaler in scalers.items():
    X_train = train[FEATURES].copy()
    X_test  = test[FEATURES].copy()
    y_train = train[TARGET].copy()
    y_test  = test[TARGET].copy()

    if name != 'None':
        X_train[FEATURES] = scaler.fit_transform(X_train[FEATURES])
        X_test[FEATURES] = scaler.transform(X_test[FEATURES])
    
    # RF
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    r2_rf = r2_score(y_test, rf.predict(X_test))
    results_rf[name] = r2_rf
    
    # XGB
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    r2_xgb = r2_score(y_test, xgb_model.predict(X_test))
    results_xgb[name] = r2_xgb

print('--- R2 Scores (Random Forest) ---')
for name, score in sorted(results_rf.items(), key=lambda item: item[1], reverse=True):
    print(f'{name}: {score:.4f}')

print('\n--- R2 Scores (XGBoost) ---')
for name, score in sorted(results_xgb.items(), key=lambda item: item[1], reverse=True):
    print(f'{name}: {score:.4f}')
