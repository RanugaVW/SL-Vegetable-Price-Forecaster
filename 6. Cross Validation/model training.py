import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import os
import joblib
import optuna
import scipy.stats as stats

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
N_SPLITS_CV = 10
XGB_TRIALS = 20
LGB_TRIALS = 20
WEIGHT_TRIALS = 30

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Final_Combined_data.csv')
OUTPUT_DIR = BASE_DIR


def safe_expm1(arr):
    return np.maximum(np.expm1(arr), 0.0)


def slugify(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def get_feature_list():
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


def prepare_dataset(data_path):
    print("Loading data...")
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
        raise ValueError("No valid train/test groups found. Check the dataset size after filtering.")

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    return train_df, test_df


def tune_xgb_lgbm_and_weights(train_df, test_df, features):
    X_train = train_df[features]
    y_train = train_df['retail_price']
    X_test = test_df[features]
    y_test = test_df['retail_price']

    y_train_log = np.log1p(y_train)

    print("\n--- Tuning XGBoost ---")

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
            'random_state': RANDOM_STATE
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

    print("\n--- Tuning LightGBM ---")

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
            'verbose': -1
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

    print("\nTraining Final Tuned Models...")
    final_xgb = xgb.XGBRegressor(**study_xgb.best_params, random_state=RANDOM_STATE)
    final_xgb.fit(X_train, y_train_log)

    lgb_params = study_lgb.best_params.copy()
    lgb_params['random_state'] = RANDOM_STATE
    lgb_params['verbose'] = -1
    final_lgb = lgb.LGBMRegressor(**lgb_params)
    final_lgb.fit(X_train, y_train_log)

    pred_xgb_raw = safe_expm1(final_xgb.predict(X_test))
    pred_lgb_raw = safe_expm1(final_lgb.predict(X_test))

    print("\n--- Tuning Ensemble Weights ---")

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

    print(f"Optimal Ensemble Weight found -> XGBoost: {optimal_w_xgb:.3f}, LightGBM: {optimal_w_lgb:.3f}")

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
        'lgb_accuracy': 1.0 - mape_lgb
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
            'final_pred': final_pred
        }
    }


def group_aware_timeseries_cv(df_ready, features, xgb_params, lgb_params, w_xgb, w_lgb, label):
    print("\n" + "=" * 60)
    print(f"  STEP 7: Group-Aware 5-Fold TimeSeries CV - {label}")
    print("=" * 60)

    fold_results = []

    group_splits = {}
    for group_key, group in df_ready.groupby(['retail_market', 'vegetable_type']):
        group = group.sort_values(['year', 'week_num']).reset_index(drop=True)
        if len(group) < (N_SPLITS_CV + 1):
            continue
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
        splits = list(tscv.split(group))
        if len(splits) == N_SPLITS_CV:
            group_splits[group_key] = (group, splits)

    if not group_splits:
        raise ValueError(f"No valid groups found for CV in {label}.")

    print(f"\nRunning {N_SPLITS_CV} folds for {label}...\n")

    for fold_idx in range(N_SPLITS_CV):
        train_parts = []
        test_parts = []

        for group, splits in group_splits.values():
            train_idx, test_idx = splits[fold_idx]
            train_parts.append(group.iloc[train_idx])
            test_parts.append(group.iloc[test_idx])

        if not train_parts or not test_parts:
            raise ValueError(f"Fold {fold_idx + 1} in {label} has no training/testing data.")

        train_df_fold = pd.concat(train_parts, ignore_index=True)
        test_df_fold = pd.concat(test_parts, ignore_index=True)

        X_train_f = train_df_fold[features]
        y_train_f = train_df_fold['retail_price']
        X_test_f = test_df_fold[features]
        y_test_f = test_df_fold['retail_price']

        y_train_f_log = np.log1p(y_train_f)

        fold_xgb = xgb.XGBRegressor(**xgb_params)
        fold_xgb.fit(X_train_f, y_train_f_log, verbose=False)
        pred_xgb = safe_expm1(fold_xgb.predict(X_test_f))

        fold_lgb = lgb.LGBMRegressor(**lgb_params)
        fold_lgb.fit(X_train_f, y_train_f_log)
        pred_lgb = safe_expm1(fold_lgb.predict(X_test_f))

        pred_ens = (w_xgb * pred_xgb) + (w_lgb * pred_lgb)

        r2_val = r2_score(y_test_f, pred_ens)
        mape_val = mean_absolute_percentage_error(y_test_f, pred_ens)
        acc_val = 1.0 - mape_val

        xgb_mape_val = mean_absolute_percentage_error(y_test_f, pred_xgb)
        lgb_mape_val = mean_absolute_percentage_error(y_test_f, pred_lgb)

        fold_results.append({
            'fold': fold_idx + 1,
            'r2': r2_val,
            'mape': mape_val,
            'accuracy': acc_val,
            'xgb_mape': xgb_mape_val,
            'lgb_mape': lgb_mape_val,
            'xgb_accuracy': 1.0 - xgb_mape_val,
            'lgb_accuracy': 1.0 - lgb_mape_val,
            'train_rows': len(train_df_fold),
            'test_rows': len(test_df_fold)
        })

        print(
            f"  Fold {fold_idx + 1}: "
            f"R²={r2_val:.4f}  "
            f"MAPE={mape_val * 100:.2f}%  "
            f"Accuracy={acc_val * 100:.2f}%  "
            f"[Train rows: {len(train_df_fold):,} | Test rows: {len(test_df_fold):,}]"
        )

    fold_df = pd.DataFrame(fold_results)

    k = len(fold_df)
    t_crit = stats.t.ppf(0.975, df=k - 1)

    def mean_ci(arr, is_r2=False):
        arr = np.array(arr, dtype=float)
        mean = arr.mean()
        std = arr.std(ddof=1)
        margin = t_crit * std / np.sqrt(k)
        low_bound = mean - margin
        high_bound = mean + margin
        
        if is_r2:
            high_bound = min(high_bound, 1.0)
            
        return mean, std, margin, low_bound, high_bound

    r2_mean, r2_std, r2_margin, r2_low, r2_high = mean_ci(fold_df['r2'], is_r2=True)
    
    mape_mean, mape_std, mape_margin, mape_low, mape_high = mean_ci(fold_df['mape'])
    acc_mean, acc_std, acc_margin, acc_low, acc_high = mean_ci(fold_df['accuracy'])
    xgb_acc_mean, xgb_acc_std, xgb_acc_margin, xgb_acc_low, xgb_acc_high = mean_ci(fold_df['xgb_accuracy'])
    lgb_acc_mean, lgb_acc_std, lgb_acc_margin, lgb_acc_low, lgb_acc_high = mean_ci(fold_df['lgb_accuracy'])

    print("\n" + "-" * 60)
    print(f"  CROSS-VALIDATION SUMMARY ({label})")
    print("-" * 60)
    print(f"R²       : {r2_mean:.4f} ± {r2_margin:.4f}  (95% CI: [{r2_low:.4f}, {r2_high:.4f}])")
    print(f"MAPE     : {mape_mean * 100:.2f}% ± {mape_margin * 100:.2f}%  (95% CI: [{mape_low * 100:.2f}%, {mape_high * 100:.2f}%])")
    print(f"Accuracy : {acc_mean * 100:.2f}% ± {acc_margin * 100:.2f}%  (95% CI: [{acc_low * 100:.2f}%, {acc_high * 100:.2f}%])")
    print(f"XGB Accuracy : {xgb_acc_mean * 100:.2f}% ± {xgb_acc_margin * 100:.2f}%")
    print(f"LGB Accuracy : {lgb_acc_mean * 100:.2f}% ± {lgb_acc_margin * 100:.2f}%")
    print("-" * 60)

    summary = {
        'label': label,
        'r2': {'mean': r2_mean, 'std': r2_std, 'margin': r2_margin, 'low': r2_low, 'high': r2_high},
        'mape': {'mean': mape_mean, 'std': mape_std, 'margin': mape_margin, 'low': mape_low, 'high': mape_high},
        'accuracy': {'mean': acc_mean, 'std': acc_std, 'margin': acc_margin, 'low': acc_low, 'high': acc_high},
        'xgb_accuracy': {'mean': xgb_acc_mean, 'std': xgb_acc_std, 'margin': xgb_acc_margin, 'low': xgb_acc_low, 'high': xgb_acc_high},
        'lgb_accuracy': {'mean': lgb_acc_mean, 'std': lgb_acc_std, 'margin': lgb_acc_margin, 'low': lgb_acc_low, 'high': lgb_acc_high},
        'fold_df': fold_df
    }

    return summary


def save_report(report_path, content):
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)


def run_experiment(df_all, season_filter, label, model_dir, report_dir, le_dict):
    if season_filter is None:
        df_ready = df_all.copy()
    else:
        df_ready = df_all[df_all['seasonality'] == season_filter].copy()

    if df_ready.empty:
        raise ValueError(f"No data found for experiment: {label}")

    features = get_feature_list()

    train_df, test_df = make_train_test_split(df_ready)

    tuning_results = tune_xgb_lgbm_and_weights(train_df, test_df, features)

    study_xgb = tuning_results['study_xgb']
    study_lgb = tuning_results['study_lgb']
    final_xgb = tuning_results['final_xgb']
    final_lgb = tuning_results['final_lgb']
    weights = tuning_results['weights']
    holdout_results = tuning_results['holdout_results']

    xgb_params_cv = {**study_xgb.best_params, 'random_state': RANDOM_STATE}
    lgb_params_cv = {**study_lgb.best_params, 'random_state': RANDOM_STATE, 'verbose': -1}

    cv_summary = group_aware_timeseries_cv(
        df_ready=df_ready,
        features=features,
        xgb_params=xgb_params_cv,
        lgb_params=lgb_params_cv,
        w_xgb=weights['xgb'],
        w_lgb=weights['lgb'],
        label=label
    )

    slug = slugify(label)
    cv_csv_path = os.path.join(report_dir, f'cv_fold_results_{slug}.csv')
    cv_summary['fold_df'].to_csv(cv_csv_path, index=False)

    cv_report = f"""5-Fold Group-Aware TimeSeries Cross-Validation Report
====================================================
Model Type   : {label}
CV Strategy  : TimeSeriesSplit (n_splits={N_SPLITS_CV})
CI Method    : t-distribution, 95%, df={N_SPLITS_CV - 1}
Hyperparams  : Fixed from Optuna tuned on holdout split
Weights      : Fixed from Optuna tuned on holdout split

Holdout Metrics (original single split)
---------------------------------------
R²           : {holdout_results['r2']:.4f}
Accuracy     : {holdout_results['accuracy'] * 100:.2f}%
MAPE         : {holdout_results['mape'] * 100:.2f}%

Cross-Validation Summary
------------------------
R²           : {cv_summary['r2']['mean']:.4f} ± {cv_summary['r2']['margin']:.4f}
             : 95% CI [{cv_summary['r2']['low']:.4f}, {cv_summary['r2']['high']:.4f}]

MAPE         : {cv_summary['mape']['mean'] * 100:.2f}% ± {cv_summary['mape']['margin'] * 100:.2f}%
             : 95% CI [{cv_summary['mape']['low'] * 100:.2f}%, {cv_summary['mape']['high'] * 100:.2f}%]

Accuracy     : {cv_summary['accuracy']['mean'] * 100:.2f}% ± {cv_summary['accuracy']['margin'] * 100:.2f}%
             : 95% CI [{cv_summary['accuracy']['low'] * 100:.2f}%, {cv_summary['accuracy']['high'] * 100:.2f}%]

XGB Accuracy : {cv_summary['xgb_accuracy']['mean'] * 100:.2f}% ± {cv_summary['xgb_accuracy']['margin'] * 100:.2f}%
LGB Accuracy : {cv_summary['lgb_accuracy']['mean'] * 100:.2f}% ± {cv_summary['lgb_accuracy']['margin'] * 100:.2f}%

Per-Fold Results
----------------
{cv_summary['fold_df'].to_string(index=False)}
"""
    report_path = os.path.join(report_dir, f'cv_report_{slug}.txt')
    save_report(report_path, cv_report)
    print(f"\nCV report saved to: {report_path}")
    print(f"Fold results CSV saved to: {cv_csv_path}")

    dataset_name = os.path.basename(DATA_PATH)
    final_report = f"""Advanced XGBoost + LightGBM Ensemble - {label}
===========================================================
Model built from: {dataset_name}
Target  : np.log1p(retail_price) -> Expm1 inverted logic
Optuna Trials: {XGB_TRIALS} for XGBoost, {LGB_TRIALS} for LightGBM, {WEIGHT_TRIALS} for ensemble weights

Tuned Hyperparameters
--------------------
  XGBoost  : {study_xgb.best_params}
  LightGBM : {study_lgb.best_params}

Ensemble Blending
-----------------
  XGBoost Weight : {weights['xgb']:.4f}
  LightGBM Weight: {weights['lgb']:.4f}

Overall Ensemble Metrics (single holdout split)
-----------------------------------------------
  R2  Score              : {holdout_results['r2']:.4f}
  Accuracy (1 - MAPE)    : {holdout_results['accuracy'] * 100:.2f}%
  MAPE                   : {holdout_results['mape'] * 100:.2f}%

Individual Accuracies
---------------------
  XGBoost Accuracy       : {(1.0 - mean_absolute_percentage_error(test_df['retail_price'], safe_expm1(final_xgb.predict(test_df[features])))) * 100:.2f}%
  LightGBM Accuracy      : {(1.0 - mean_absolute_percentage_error(test_df['retail_price'], safe_expm1(final_lgb.predict(test_df[features])))) * 100:.2f}%

Cross-Validation Results (5-Fold Group-Aware TimeSeriesSplit)
-------------------------------------------------------------
  R²       : {cv_summary['r2']['mean']:.4f} ± {cv_summary['r2']['margin']:.4f}
  Accuracy : {cv_summary['accuracy']['mean'] * 100:.2f}% ± {cv_summary['accuracy']['margin'] * 100:.2f}%
  MAPE     : {cv_summary['mape']['mean'] * 100:.2f}% ± {cv_summary['mape']['margin'] * 100:.2f}%
"""
    report_name = os.path.join(report_dir, f'xgb_lgbm_advanced_ensemble_optuna_performance_{slug}.txt')
    save_report(report_name, final_report)
    print(f"Main report saved to: {report_name}")

    bundle = {
        'xgb': final_xgb,
        'lgb': final_lgb,
        'features': features,
        'label_encoders': le_dict,
        'weights': {'xgb': weights['xgb'], 'lgb': weights['lgb']},
        'target_transform': 'log1p'
    }
    model_name = os.path.join(model_dir, f'xgb_lgbm_advanced_ensemble_optuna_model_{slug}.joblib')
    joblib.dump(bundle, model_name)
    print(f"Artifacts saved to: {model_name}")

    return {
        'label': label,
        'holdout_results': holdout_results,
        'cv_summary': cv_summary
    }


def train_xgb_lgbm_advanced():
    model_dir = os.path.join(OUTPUT_DIR, 'Models')
    report_dir = os.path.join(OUTPUT_DIR, 'Reports')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    df_all, le_dict = prepare_dataset(DATA_PATH)

    results = {}

    print("\n" + "#" * 70)
    print("RUNNING UNIFIED MODEL (MAHA + YALA)")
    print("#" * 70)
    results['Unified'] = run_experiment(
        df_all=df_all,
        season_filter=None,
        label='Unified (Maha + Yala)',
        model_dir=model_dir,
        report_dir=report_dir,
        le_dict=le_dict
    )

    print("\n" + "#" * 70)
    print("RUNNING MAHA MODEL")
    print("#" * 70)
    results['Maha'] = run_experiment(
        df_all=df_all,
        season_filter='Maha Season',
        label='Maha',
        model_dir=model_dir,
        report_dir=report_dir,
        le_dict=le_dict
    )

    print("\n" + "#" * 70)
    print("RUNNING YALA MODEL")
    print("#" * 70)
    results['Yala'] = run_experiment(
        df_all=df_all,
        season_filter='Yala Season',
        label='Yala',
        model_dir=model_dir,
        report_dir=report_dir,
        le_dict=le_dict
    )

    summary_rows = []
    for key, res in results.items():
        cv = res['cv_summary']
        summary_rows.append({
            'model': key,
            'r2_mean': cv['r2']['mean'],
            'r2_ci': cv['r2']['margin'],
            'mape_mean': cv['mape']['mean'],
            'mape_ci': cv['mape']['margin'],
            'accuracy_mean': cv['accuracy']['mean'],
            'accuracy_ci': cv['accuracy']['margin']
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(report_dir, 'cv_summary_all_models.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nCombined summary saved to: {summary_csv}")
    print("\nFinal summary:")
    print(summary_df.to_string(index=False))

    return results


if __name__ == "__main__":
    train_xgb_lgbm_advanced()