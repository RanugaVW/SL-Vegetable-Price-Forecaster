import os

import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit


optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
N_SPLITS_CV = 10
XGB_TRIALS = 20
LGB_TRIALS = 20
WEIGHT_TRIALS = 30

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Final_Combined_data.csv')
REPORT_DIR = os.path.join(BASE_DIR, 'Reports')
REPORT_PATH = os.path.join(REPORT_DIR, 'diesel_paradox_empirical_proof.txt')


def safe_expm1(arr):
    return np.maximum(np.expm1(arr), 0.0)


def get_full_feature_list():
    return [
        'mean_farmer_price_filled', 'farmer_retail_spread_lag_1',
        'mean_farmer_price_lag_1', 'mean_farmer_price_lag_2',
        'mean_farmer_price_lag_3', 'mean_farmer_price_lag_4',
        'mean_farmer_price_lag_5', 'mean_farmer_price_lag_6',
        'mean_farmer_price_lag_8',
        'farmer_price_roll_4', 'farmer_price_roll_8',
        'farmer_price_roll_std_4', 'farmer_price_pct_change_1',
        'retail_price_momentum_1_4', 'farmer_price_momentum_1_4',
        'year', 'week_sin', 'week_cos',
        'lanka_auto_diesel_price', 'usd_exchange_rate', 'diesel_season_int',
        'no_of_holidays',
        'reg_rain', 'reg_temp',
        'retail_price_lag_1', 'retail_price_lag_2',
        'retail_price_lag_3', 'retail_price_lag_4', 'retail_price_lag_8',
        'reg_rain_lag_1', 'reg_rain_lag_4', 'reg_rain_lag_8',
        'reg_temp_lag_1', 'reg_temp_lag_4', 'reg_temp_lag_8',
        'retail_price_roll_4',
        'retail_market_enc', 'vegetable_type_enc', 'vegetable_zone_enc', 'season_enc'
    ]


def get_feature_sets():
    full_features = get_full_feature_list()
    excluded = {'lanka_auto_diesel_price', 'diesel_season_int'}
    no_diesel_features = [feature for feature in full_features if feature not in excluded]
    return full_features, no_diesel_features


def prepare_dataset(data_path):
    print('Loading data...')
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

    df['retail_price_roll_4'] = df.groupby(['retail_market', 'vegetable_type'])['retail_price'].transform(
        lambda x: x.shift(1).rolling(4).mean()
    )

    grp = df.groupby(['retail_market', 'vegetable_type'])['mean_farmer_price']
    df['farmer_price_roll_4'] = grp.transform(lambda x: x.shift(1).rolling(4).mean())
    df['farmer_price_roll_8'] = grp.transform(lambda x: x.shift(1).rolling(8).mean())
    df['farmer_price_roll_std_4'] = grp.transform(lambda x: x.shift(1).rolling(4).std())
    df['farmer_price_pct_change_1'] = grp.transform(lambda x: x.shift(1).pct_change(1, fill_method=None))

    df['mean_farmer_price_filled'] = df['mean_farmer_price'].fillna(df['mean_farmer_price_lag_1'])
    df['farmer_retail_spread_lag_1'] = df['retail_price_lag_1'] - df['mean_farmer_price_lag_1']

    df['retail_price_momentum_1_4'] = df['retail_price_lag_1'] / (df['retail_price_lag_4'] + 1e-5)
    df['farmer_price_momentum_1_4'] = df['mean_farmer_price_lag_1'] / (df['mean_farmer_price_lag_4'] + 1e-5)

    df_ready = df.dropna(subset=[
        'retail_price_lag_8',
        'mean_farmer_price_lag_8',
        'farmer_price_roll_8',
        'retail_price_momentum_1_4'
    ]).copy()

    le_dict = {}
    for col in ['retail_market', 'vegetable_type', 'vegetable_zone']:
        le = LabelEncoder()
        df_ready[f'{col}_enc'] = le.fit_transform(df_ready[col].astype(str))
        le_dict[col] = le

    return df_ready, le_dict


def make_train_test_split(df_ready):
    train_list, test_list = [], []

    for _, group in df_ready.groupby(['retail_market', 'vegetable_type']):
        group = group.sort_values(['year', 'week_num']).reset_index(drop=True)
        if len(group) < 2:
            continue
        split = int(len(group) * 0.8)
        if split < 1:
            continue
        train_list.append(group.iloc[:split])
        test_list.append(group.iloc[split:])

    if not train_list or not test_list:
        raise ValueError('No valid train/test groups found. Check the dataset size after filtering.')

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    return train_df, test_df


def tune_xgb_lgbm_and_weights(train_df, test_df, features):
    X_train = train_df[features]
    y_train = train_df['retail_price']
    X_test = test_df[features]
    y_test = test_df['retail_price']

    y_train_log = np.log1p(y_train)

    print('\n--- Tuning XGBoost ---')

    def objective_xgb(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': RANDOM_STATE,
        }
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train_log, verbose=False)
        preds_raw = safe_expm1(model.predict(X_test))
        return mean_absolute_percentage_error(y_test, preds_raw)

    study_xgb = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study_xgb.optimize(objective_xgb, n_trials=XGB_TRIALS)
    print(f"Best XGBoost Params (MAPE: {study_xgb.best_value:.4f}): {study_xgb.best_params}")

    print('\n--- Tuning LightGBM ---')

    def objective_lgb(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': RANDOM_STATE,
            'verbose': -1,
        }
        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train_log)
        preds_raw = safe_expm1(model.predict(X_test))
        return mean_absolute_percentage_error(y_test, preds_raw)

    study_lgb = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study_lgb.optimize(objective_lgb, n_trials=LGB_TRIALS)
    print(f"Best LightGBM Params (MAPE: {study_lgb.best_value:.4f}): {study_lgb.best_params}")

    print('\nTraining Final Tuned Models...')
    final_xgb = xgb.XGBRegressor(**study_xgb.best_params, random_state=RANDOM_STATE)
    final_xgb.fit(X_train, y_train_log)

    lgb_params = study_lgb.best_params.copy()
    lgb_params['random_state'] = RANDOM_STATE
    lgb_params['verbose'] = -1
    final_lgb = lgb.LGBMRegressor(**lgb_params)
    final_lgb.fit(X_train, y_train_log)

    pred_xgb_raw = safe_expm1(final_xgb.predict(X_test))
    pred_lgb_raw = safe_expm1(final_lgb.predict(X_test))

    print('\n--- Tuning Ensemble Weights ---')

    def objective_weight(trial):
        w_xgb = trial.suggest_float('w_xgb', 0.0, 1.0)
        w_lgb = 1.0 - w_xgb
        blended_preds = (w_xgb * pred_xgb_raw) + (w_lgb * pred_lgb_raw)
        return mean_absolute_percentage_error(y_test, blended_preds)

    study_weights = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study_weights.optimize(objective_weight, n_trials=WEIGHT_TRIALS)

    optimal_w_xgb = study_weights.best_params['w_xgb']
    optimal_w_lgb = 1.0 - optimal_w_xgb

    print(f'Optimal Ensemble Weight found -> XGBoost: {optimal_w_xgb:.3f}, LightGBM: {optimal_w_lgb:.3f}')

    final_pred = (optimal_w_xgb * pred_xgb_raw) + (optimal_w_lgb * pred_lgb_raw)

    r2 = r2_score(y_test, final_pred)
    mape = mean_absolute_percentage_error(y_test, final_pred)
    mape_xgb = mean_absolute_percentage_error(y_test, pred_xgb_raw)
    mape_lgb = mean_absolute_percentage_error(y_test, pred_lgb_raw)

    holdout_results = {
        'r2': r2,
        'mape': mape,
        'accuracy': 1.0 - mape,
        'xgb_accuracy': 1.0 - mape_xgb,
        'lgb_accuracy': 1.0 - mape_lgb,
    }

    return {
        'study_xgb': study_xgb,
        'study_lgb': study_lgb,
        'final_xgb': final_xgb,
        'final_lgb': final_lgb,
        'weights': {'xgb': optimal_w_xgb, 'lgb': optimal_w_lgb},
        'holdout_results': holdout_results,
        'predictions': {
            'y_test': y_test,
            'pred_xgb_raw': pred_xgb_raw,
            'pred_lgb_raw': pred_lgb_raw,
            'final_pred': final_pred,
        },
    }


def add_return_segments(df_ready, test_df, predictions):
    group_cols = ['retail_market', 'vegetable_type']
    time_cols = ['year', 'week_num']

    return_frame = df_ready[group_cols + time_cols + ['retail_price']].copy()
    return_frame = return_frame.sort_values(group_cols + time_cols)
    return_frame['abs_weekly_return'] = return_frame.groupby(group_cols)['retail_price'].pct_change().abs()

    thresholds = (
        return_frame.dropna(subset=['abs_weekly_return'])
        .groupby(group_cols)['abs_weekly_return']
        .quantile(0.90)
        .reset_index()
        .rename(columns={'abs_weekly_return': 'extreme_threshold'})
    )

    if thresholds.empty:
        raise ValueError('Unable to compute extreme-week thresholds. Check group sizes in the prepared data.')

    eval_frame = test_df[group_cols + time_cols + ['retail_price']].copy()
    eval_frame['prediction'] = predictions

    eval_frame = eval_frame.merge(
        return_frame[group_cols + time_cols + ['abs_weekly_return']],
        on=group_cols + time_cols,
        how='left'
    )
    eval_frame = eval_frame.merge(thresholds, on=group_cols, how='left')
    eval_frame = eval_frame.dropna(subset=['abs_weekly_return', 'extreme_threshold']).copy()

    eval_frame['segment'] = np.where(
        eval_frame['abs_weekly_return'] >= eval_frame['extreme_threshold'],
        'Extreme Weeks',
        'Normal Weeks'
    )

    return eval_frame


def compute_segment_metrics(eval_frame):
    results = {}

    for segment_name in ['Extreme Weeks', 'Normal Weeks']:
        segment_frame = eval_frame[eval_frame['segment'] == segment_name]
        if segment_frame.empty:
            raise ValueError(f'No rows available for segment: {segment_name}')

        y_true = segment_frame['retail_price']
        y_pred = segment_frame['prediction']
        mape = mean_absolute_percentage_error(y_true, y_pred)

        results[segment_name] = {
            'count': len(segment_frame),
            'r2': r2_score(y_true, y_pred),
            'mape': mape,
            'accuracy': 1.0 - mape,
        }

    return results


def format_pct(value):
    return f'{value * 100:.2f}%'


def build_comparison_table(full_results, no_diesel_results, full_segments, no_diesel_segments):
    rows = [
        {
            'Metric': 'Global R²',
            'Full Model': f"{full_results['r2']:.4f}",
            'No Diesel': f"{no_diesel_results['r2']:.4f}",
            'Delta (No Diesel - Full)': f"{no_diesel_results['r2'] - full_results['r2']:+.4f}",
        },
        {
            'Metric': 'Global Accuracy',
            'Full Model': format_pct(full_results['accuracy']),
            'No Diesel': format_pct(no_diesel_results['accuracy']),
            'Delta (No Diesel - Full)': f"{(no_diesel_results['accuracy'] - full_results['accuracy']) * 100:+.2f} pp",
        },
        {
            'Metric': 'Extreme Spike Accuracy',
            'Full Model': format_pct(full_segments['Extreme Weeks']['accuracy']),
            'No Diesel': format_pct(no_diesel_segments['Extreme Weeks']['accuracy']),
            'Delta (No Diesel - Full)': f"{(no_diesel_segments['Extreme Weeks']['accuracy'] - full_segments['Extreme Weeks']['accuracy']) * 100:+.2f} pp",
        },
        {
            'Metric': 'Extreme Spike MAPE',
            'Full Model': format_pct(full_segments['Extreme Weeks']['mape']),
            'No Diesel': format_pct(no_diesel_segments['Extreme Weeks']['mape']),
            'Delta (No Diesel - Full)': f"{(no_diesel_segments['Extreme Weeks']['mape'] - full_segments['Extreme Weeks']['mape']) * 100:+.2f} pp",
        },
        {
            'Metric': 'Normal Week Accuracy',
            'Full Model': format_pct(full_segments['Normal Weeks']['accuracy']),
            'No Diesel': format_pct(no_diesel_segments['Normal Weeks']['accuracy']),
            'Delta (No Diesel - Full)': f"{(no_diesel_segments['Normal Weeks']['accuracy'] - full_segments['Normal Weeks']['accuracy']) * 100:+.2f} pp",
        },
        {
            'Metric': 'Normal Week MAPE',
            'Full Model': format_pct(full_segments['Normal Weeks']['mape']),
            'No Diesel': format_pct(no_diesel_segments['Normal Weeks']['mape']),
            'Delta (No Diesel - Full)': f"{(no_diesel_segments['Normal Weeks']['mape'] - full_segments['Normal Weeks']['mape']) * 100:+.2f} pp",
        },
    ]

    return pd.DataFrame(rows)


def build_report_text(table_df, full_results, no_diesel_results, full_segments, no_diesel_segments):
    lines = []
    lines.append('Diesel Price Paradox Empirical Proof')
    lines.append('=' * 48)
    lines.append('')
    lines.append('Design')
    lines.append('------')
    lines.append('Full Model includes lanka_auto_diesel_price and diesel_season_int.')
    lines.append('No Diesel excludes both diesel columns and keeps the remaining engineered features unchanged.')
    lines.append('Extreme Weeks are defined using the top decile of group-wise absolute weekly retail-price returns.')
    lines.append('Delta columns are computed as No Diesel - Full; percentage metrics are shown in percentage points (pp).')
    lines.append('')
    lines.append('Holdout Metrics')
    lines.append('---------------')
    lines.append(f"Full Model  -> R²: {full_results['r2']:.4f}, Accuracy: {format_pct(full_results['accuracy'])}, MAPE: {format_pct(full_results['mape'])}")
    lines.append(f"No Diesel   -> R²: {no_diesel_results['r2']:.4f}, Accuracy: {format_pct(no_diesel_results['accuracy'])}, MAPE: {format_pct(no_diesel_results['mape'])}")
    lines.append('')
    lines.append('Segment Sizes')
    lines.append('-------------')
    lines.append(f"Full Model  -> Extreme: {full_segments['Extreme Weeks']['count']:,}, Normal: {full_segments['Normal Weeks']['count']:,}")
    lines.append(f"No Diesel   -> Extreme: {no_diesel_segments['Extreme Weeks']['count']:,}, Normal: {no_diesel_segments['Normal Weeks']['count']:,}")
    lines.append('')
    lines.append('Comparison Table')
    lines.append('-----------------')
    lines.append(table_df.to_string(index=False))
    lines.append('')
    lines.append('Interpretation')
    lines.append('--------------')
    lines.append('A positive delta in accuracy supports the reviewer concern that diesel information is mainly useful during spike weeks.')
    lines.append('A negative delta in MAPE indicates lower error for the No Diesel model.')

    return '\n'.join(lines)


def run_diesel_paradox_test():
    os.makedirs(REPORT_DIR, exist_ok=True)

    df_ready, _ = prepare_dataset(DATA_PATH)
    train_df, test_df = make_train_test_split(df_ready)

    full_features, no_diesel_features = get_feature_sets()

    print('\n' + '=' * 80)
    print('FULL MODEL: WITH DIESEL FEATURES')
    print('=' * 80)
    full_run = tune_xgb_lgbm_and_weights(train_df, test_df, full_features)

    print('\n' + '=' * 80)
    print('ABLATED MODEL: NO DIESEL FEATURES')
    print('=' * 80)
    no_diesel_run = tune_xgb_lgbm_and_weights(train_df, test_df, no_diesel_features)

    full_eval = add_return_segments(df_ready, test_df, full_run['predictions']['final_pred'])
    no_diesel_eval = add_return_segments(df_ready, test_df, no_diesel_run['predictions']['final_pred'])

    full_segment_metrics = compute_segment_metrics(full_eval)
    no_diesel_segment_metrics = compute_segment_metrics(no_diesel_eval)

    summary_table = build_comparison_table(
        full_results=full_run['holdout_results'],
        no_diesel_results=no_diesel_run['holdout_results'],
        full_segments=full_segment_metrics,
        no_diesel_segments=no_diesel_segment_metrics,
    )

    report_text = build_report_text(
        table_df=summary_table,
        full_results=full_run['holdout_results'],
        no_diesel_results=no_diesel_run['holdout_results'],
        full_segments=full_segment_metrics,
        no_diesel_segments=no_diesel_segment_metrics,
    )

    print('\n' + '=' * 80)
    print('DIESEL PRICE PARADOX COMPARISON')
    print('=' * 80)
    print(summary_table.to_string(index=False))
    print('\nReport saved to:', REPORT_PATH)

    with open(REPORT_PATH, 'w', encoding='utf-8') as file_handle:
        file_handle.write(report_text)


if __name__ == '__main__':
    run_diesel_paradox_test()