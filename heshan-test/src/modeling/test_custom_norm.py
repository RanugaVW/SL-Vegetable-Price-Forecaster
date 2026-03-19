import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

train = df_clean[df_clean['year'].isin(TRAIN_YEARS)].copy()
test  = df_clean[df_clean['year'].isin(TEST_YEARS)].copy()

if len(test) == 0:
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df_clean, test_size=0.2, random_state=42)

X_train_raw = train[FEATURES].copy()
X_test_raw  = test[FEATURES].copy()
y_train = train[TARGET].copy()
y_test  = test[TARGET].copy()

class VegetablePriceNormalizer:
    def __init__(self):
        self.scalers = {}
        
    def analyze_price_distribution(self, df):
        veg_stats = df.groupby('vegetable_type')['price'].agg(['count', 'min', 'max', 'mean', 'std', 'median']).round(2)
        price_groups = {
            'low_price': veg_stats[veg_stats['mean'] < 50].index.tolist(),
            'medium_price': veg_stats[(veg_stats['mean'] >= 50) & (veg_stats['mean'] < 100)].index.tolist(),
            'high_price': veg_stats[veg_stats['mean'] >= 100].index.tolist()
        }
        return veg_stats, price_groups
    
    def analyze_location_effects(self, df):
        loc_stats = df.groupby('location')['price'].agg(['mean', 'std', 'count']).round(2)
        national_avg = df['price'].mean()
        loc_stats['price_index'] = (loc_stats['mean'] / national_avg * 100).round(1)
        return loc_stats
    
    def analyze_seasonal_patterns(self, df):
        df['month'] = pd.to_datetime(df['week_start']).dt.month
        monthly_stats = df.groupby('month')['price'].agg(['mean', 'std']).round(2)
        overall_mean = df['price'].mean()
        monthly_stats['seasonal_index'] = (monthly_stats['mean'] / overall_mean * 100).round(1)
        return monthly_stats
    
    def get_vegetable_specific_scaler(self, vegetable_type, price_range):
        if price_range == 'low_price': return {'scaler': MinMaxScaler(feature_range=(0, 1))}
        elif price_range == 'medium_price': return {'scaler': RobustScaler(quantile_range=(10, 90))}
        else: return {'scaler': PowerTransformer(method='yeo-johnson', standardize=True)}

    def fit_transform_vegetable_aware(self, X_train, X_test, df_train, df_test):
        veg_stats, price_groups = self.analyze_price_distribution(pd.concat([df_train, df_test]))
        X_train_norm, X_test_norm = X_train.copy(), X_test.copy()
        for group_name, vegetables in price_groups.items():
            if not vegetables: continue
            train_mask = df_train['vegetable_type'].isin(vegetables)
            test_mask = df_test['vegetable_type'].isin(vegetables)
            if not train_mask.any(): continue
            scaler = self.get_vegetable_specific_scaler(group_name, group_name)['scaler']
            numeric_features = [c for c in X_train.columns if c not in ['is_maha', 'is_upcountry', 'upcountry_x_maha']]
            train_data = X_train.loc[train_mask, numeric_features]
            if len(train_data) > 0:
                scaler.fit(train_data)
                X_train_norm.loc[train_mask, numeric_features] = scaler.transform(train_data)
                if len(X_test.loc[test_mask, numeric_features]) > 0:
                    X_test_norm.loc[test_mask, numeric_features] = scaler.transform(X_test.loc[test_mask, numeric_features])
        return X_train_norm, X_test_norm

    def fit_transform_location_aware(self, X_train, X_test, df_train, df_test):
        loc_stats = self.analyze_location_effects(pd.concat([df_train, df_test]))
        X_train_norm, X_test_norm = X_train.copy(), X_test.copy()
        high_price_locs = loc_stats[loc_stats['price_index'] > 105].index.tolist()
        medium_price_locs = loc_stats[(loc_stats['price_index'] >= 95) & (loc_stats['price_index'] <= 105)].index.tolist()
        low_price_locs = loc_stats[loc_stats['price_index'] < 95].index.tolist()
        location_groups = {'high': high_price_locs, 'medium': medium_price_locs, 'low': low_price_locs}
        numeric_features = [c for c in X_train.columns if c not in ['is_maha', 'is_upcountry', 'upcountry_x_maha']]
        for group_name, locations in location_groups.items():
            if not locations: continue
            train_mask = df_train['location'].isin(locations)
            test_mask = df_test['location'].isin(locations)
            if not train_mask.any(): continue
            scaler = RobustScaler(quantile_range=(5, 95))
            train_data = X_train.loc[train_mask, numeric_features]
            if len(train_data) > 0:
                scaler.fit(train_data)
                X_train_norm.loc[train_mask, numeric_features] = scaler.transform(train_data)
                if len(X_test.loc[test_mask, numeric_features]) > 0:
                    X_test_norm.loc[test_mask, numeric_features] = scaler.transform(X_test.loc[test_mask, numeric_features])
        return X_train_norm, X_test_norm

    def fit_transform_hybrid(self, X_train, X_test, df_train, df_test):
        X_train_norm, X_test_norm = X_train.copy(), X_test.copy()
        df_train['veg_loc'] = df_train['vegetable_type'] + '_' + df_train['location']
        df_test['veg_loc'] = df_test['vegetable_type'] + '_' + df_test['location']
        veg_loc_stats = pd.concat([df_train, df_test]).groupby('veg_loc')['price'].agg(['mean', 'std', 'count']).round(2)
        veg_loc_stats = veg_loc_stats[veg_loc_stats['count'] > 10]
        numeric_features = [c for c in X_train.columns if c not in ['is_maha', 'is_upcountry', 'upcountry_x_maha']]
        for veg_loc in veg_loc_stats.index:
            train_mask = df_train['veg_loc'] == veg_loc
            test_mask = df_test['veg_loc'] == veg_loc
            if not train_mask.any(): continue
            mean_price = veg_loc_stats.loc[veg_loc, 'mean']
            std_price  = veg_loc_stats.loc[veg_loc, 'std']
            cv = std_price / mean_price if mean_price > 0 else 0
            if cv < 0.2: scaler = StandardScaler()
            elif cv < 0.5: scaler = RobustScaler()
            else: scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            if len(X_train.loc[train_mask, numeric_features]) > 0:
                scaler.fit(X_train.loc[train_mask, numeric_features])
                X_train_norm.loc[train_mask, numeric_features] = scaler.transform(X_train.loc[train_mask, numeric_features])
                if len(X_test.loc[test_mask, numeric_features]) > 0:
                    X_test_norm.loc[test_mask, numeric_features] = scaler.transform(X_test.loc[test_mask, numeric_features])
        return X_train_norm, X_test_norm

normalizer = VegetablePriceNormalizer()
methods = {
    'Vegetable-Aware': normalizer.fit_transform_vegetable_aware(X_train_raw, X_test_raw, train, test),
    'Location-Aware': normalizer.fit_transform_location_aware(X_train_raw, X_test_raw, train, test),
    'Hybrid': normalizer.fit_transform_hybrid(X_train_raw, X_test_raw, train, test)
}

print("\n--- Testing Custom Normalization Architectures (Random Forest) ---")
print(f"{'Method':<20} | {'R2 Score':<10} | {'MAE (Rs)':<10} | {'RMSE (Rs)':<10} | {'MAPE (%)':<10}")
print("-" * 70)

for name, (X_tr, X_ts) in methods.items():
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_train)
    preds = model.predict(X_ts)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - preds[mask]) / y_test[mask])) * 100
    
    print(f"{name:<20} | {r2:<10.4f} | {mae:<10.2f} | {rmse:<10.2f} | {mape:<10.2f}")
