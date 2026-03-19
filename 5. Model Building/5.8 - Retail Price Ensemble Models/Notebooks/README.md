# 5.8 — XGBoost + LightGBM Ensemble for Retail Vegetable Price Prediction

## Overview

This model predicts the **weekly retail price** of 12 vegetable types across 14 Sri Lankan market locations by ensembling two complementary gradient-boosting algorithms — **XGBoost** and **LightGBM**. The core innovation is the heavy use of **farmer (producer) price** as a leading indicator for retail prices, combined with weather, economic, and seasonal features.

| Metric | Value |
|:---|:---|
| **R² Score** | 0.9300 |
| **Accuracy (1 − MAPE)** | 90.75% |
| **MAPE** | 9.25% |
| **Training Rows** | 45,579 |
| **Test Rows** | 11,546 |
| **Total Features** | 38 |

---

## Pipeline: Step-by-Step Breakdown

### Step 1 — Data Loading

- **Input file:** `Final_Combined_data.csv` (61,152 rows) — the fully merged dataset from phase 3.7 containing retail prices, farmer prices, weather, economic indicators, holidays, and seasonality.
- **Immediate cleanup:** Drops the `code` column (an internal identifier with no predictive value).

---

### Step 2 — Feature Engineering

This is the most critical step. The script creates **5 categories** of engineered features:

#### 2.1 — Cyclic Week Encoding

```
week_num = extract numeric week (1–52) from the string column "week"
week_sin = sin(2π × week_num / 52)
week_cos = cos(2π × week_num / 52)
```

**Why?** Raw week numbers (1, 2, ... 52) create a false discontinuity — the model would treat Week 52 and Week 1 as maximally far apart, when they are actually adjacent. Sine/cosine encoding places them on a smooth circle, so the model understands that December → January is continuous, not a jump.

#### 2.2 — Regional Weather Aggregation

```python
regional_weather = df.groupby(['Year_Week', 'vegetable_zone'])[['rain_sum', 'mean_apparent_temperature']].mean()
```

Instead of using weather at each individual market, this averages the weather across the entire **vegetable growing zone** (Upcountry vs. Lowcountry) per week. This captures **supply-side shocks** — e.g., a drought in the upcountry zone affects prices in ALL upcountry markets, not just one.

- Produces: `reg_rain`, `reg_temp`

#### 2.3 — Economic × Season Interaction

```python
season_enc = LabelEncoder(seasonality)   # Maha=0, Yala=1 (or vice versa)
diesel_season_int = diesel_price × (season_enc + 1)
```

**Why?** Diesel cost (transport from farms to markets) impacts prices differently depending on the growing season. During the "Yala" season (dry), transport routes may differ, and supply patterns change. This interaction term lets the model learn season-dependent cost sensitivity.

#### 2.4 — Lag Features (Time-Series Memory)

The data is sorted by `[retail_market, vegetable_type, year, week_num]` before creating lags. All lags use `shift()` to avoid data leakage — only past values are used.

##### Retail Price Lags
| Feature | Look-back |
|:---|:---|
| `retail_price_lag_1` | 1 week ago |
| `retail_price_lag_2` | 2 weeks ago |
| `retail_price_lag_3` | 3 weeks ago |
| `retail_price_lag_4` | 4 weeks ago |
| `retail_price_lag_8` | 8 weeks ago |

##### Farmer Price Lags (more granular — 7 lags)
| Feature | Look-back |
|:---|:---|
| `mean_farmer_price_lag_1` to `lag_6` | 1 to 6 weeks ago |
| `mean_farmer_price_lag_8` | 8 weeks ago |

**Why more farmer lags?** Farmer prices are a leading indicator — changes in wholesale/producer prices at the farm gate propagate to retail markets with varying delays. By giving the model lags at weeks 1 through 6 AND 8, it can learn the exact propagation delay for different vegetables and markets.

##### Weather Lags
| Feature | Look-back |
|:---|:---|
| `reg_rain_lag_1`, `reg_temp_lag_1` | 1 week ago |
| `reg_rain_lag_4`, `reg_temp_lag_4` | 4 weeks ago (1 month — growth cycle) |
| `reg_rain_lag_8`, `reg_temp_lag_8` | 8 weeks ago (2 months — planting impact) |

**Why 4 and 8 weeks?** Weather doesn't affect prices immediately. Rain 4 weeks ago affected crops during their growth; weather 8 weeks ago affected planting decisions. These lags capture the agricultural growth cycle.

#### 2.5 — Rolling & Momentum Features

##### Retail Price Rolling Mean
```python
retail_price_roll_4 = shift(1).rolling(4).mean()
```
4-week moving average of past retail prices (shifted by 1 to avoid leakage). Captures the **medium-term price trend**.

##### Farmer Price Rolling Features (4 features)
| Feature | Formula | Signal |
|:---|:---|:---|
| `farmer_price_roll_4` | 4-week rolling mean (shifted) | Short-term farm price trend |
| `farmer_price_roll_8` | 8-week rolling mean (shifted) | Long-term farm price trend |
| `farmer_price_roll_std_4` | 4-week rolling standard deviation | **Price volatility/stability** |
| `farmer_price_pct_change_1` | Week-over-week % change (shifted) | **Price momentum** (rising/falling) |

##### Derived Features
| Feature | Formula | Signal |
|:---|:---|:---|
| `mean_farmer_price_filled` | Current farmer price, NaN filled with lag_1 | Strongest direct predictor |
| `farmer_retail_spread_lag_1` | `retail_lag_1 - farmer_lag_1` | Markup/margin between farm and retail |

#### 2.6 — Categorical Encoding

Three categorical columns are label-encoded:

| Column | Purpose |
|:---|:---|
| `retail_market_enc` | Which of the 14 markets (location-specific pricing patterns) |
| `vegetable_type_enc` | Which of the 12 vegetables (price level & volatility differ) |
| `vegetable_zone_enc` | Upcountry vs. Lowcountry (supply zone) |

---

### Step 3 — Data Cleaning (Pre-Split)

```python
df_ready = df.dropna(subset=['retail_price_lag_8', 'mean_farmer_price_lag_8', 'farmer_price_roll_8'])
```

Rows where the longest lag (8 weeks) hasn't filled yet are dropped. This means the first ~8 weeks of each market-vegetable group are sacrificed to ensure all features have valid values. **No imputation is used for lag features** — only real historical data.

**Result:** 61,152 → 57,125 rows (the lost rows are the initial 8 weeks per group where lags can't be computed).

---

### Step 4 — Train/Test Split

```
For each (retail_market, vegetable_type) group:
    Train = first 80% of rows (chronologically)
    Test  = last 20% of rows (chronologically)
```

**This is NOT a random split.** It is a **time-ordered split per group**, which is critical for time-series forecasting:
- Ensures the model never "sees the future"
- Each market-vegetable pair has its own 80/20 boundary
- Preserves temporal ordering within each group

| Split | Rows |
|:---|:---|
| Train | 45,579 |
| Test | 11,546 |

---

### Step 5 — Model Training

#### XGBoost (Depth-Wise Tree Growth)

```python
XGBRegressor(
    n_estimators=1000,       # Max 1000 boosting rounds
    learning_rate=0.05,       # Small step size for fine-grained learning
    max_depth=6,              # Moderate tree depth (avoids overfitting)
    subsample=0.8,            # Row sampling (80% per tree — regularization)
    colsample_bytree=0.8,    # Feature sampling (80% per tree — regularization)
    early_stopping_rounds=50, # Stop if no improvement for 50 rounds
)
```

- **Best iteration:** 168 (out of 1000 max)
- **Individual accuracy:** 90.69%

#### LightGBM (Leaf-Wise Tree Growth)

```python
LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=63,            # Leaf-wise growth with ≤63 leaves per tree
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,                # Parallel training (all CPU cores)
    verbose=-1,               # Silent output
    # Early stopping via callback: stop at 50 rounds
)
```

- **Best iteration:** 180 (out of 1000 max)
- **Individual accuracy:** 90.71%

#### Key Difference Between XGBoost and LightGBM

| Property | XGBoost | LightGBM |
|:---|:---|:---|
| Tree Growth | **Depth-wise** (level by level) | **Leaf-wise** (best leaf first) |
| Speed | Slower, more compute | Faster, less memory |
| Strength | Better with outliers & structured data | Better with high-cardinality categoricals |
| Risk | More conservative, less overfitting | Can overfit on small data |

**Why ensemble both?** They make different types of errors. Averaging their predictions cancels out individual biases, producing more stable and generalizable results.

---

### Step 6 — Ensemble Prediction

```python
final_pred = 0.5 × XGBoost_prediction + 0.5 × LightGBM_prediction
```

Equal 50/50 weighting — since both models achieve nearly identical individual accuracy (90.69% vs 90.71%), there's no reason to favor one over the other.

---

### Step 7 — Evaluation Metrics

| Metric | Formula | Ensemble Result |
|:---|:---|:---|
| **R² Score** | 1 − (SS_res / SS_tot) | **0.9300** — 93% of price variance explained |
| **MAPE** | mean(\|actual − predicted\| / actual) | **9.25%** |
| **Accuracy** | 1 − MAPE | **90.75%** |

---

### Step 8 — Model Saving

The model is saved as a **single `.joblib` bundle** containing:

```python
{
    'xgb':            trained XGBoost model,
    'lgb':            trained LightGBM model,
    'features':       list of 38 feature names (in order),
    'label_encoders': {'retail_market': le, 'vegetable_type': le, 'vegetable_zone': le},
    'weights':        {'xgb': 0.5, 'lgb': 0.5},
}
```

This makes inference simple — load the bundle, encode new data using the saved label encoders, compute the features in the same order, and average both model predictions.

---

## Complete Feature List (38 Features)

| # | Feature | Category | Description |
|:---|:---|:---|:---|
| 1 | `mean_farmer_price_filled` | Farmer Price | Current-week farmer price (NaN → lag_1 fallback) |
| 2 | `farmer_retail_spread_lag_1` | Derived | Retail − Farmer markup from last week |
| 3–9 | `mean_farmer_price_lag_1..6,8` | Farmer Lags | 7 lag values (1–6 and 8 weeks back) |
| 10 | `farmer_price_roll_4` | Farmer Rolling | 4-week moving average |
| 11 | `farmer_price_roll_8` | Farmer Rolling | 8-week moving average |
| 12 | `farmer_price_roll_std_4` | Farmer Volatility | 4-week rolling standard deviation |
| 13 | `farmer_price_pct_change_1` | Farmer Momentum | Week-over-week % change |
| 14 | `year` | Time | Calendar year |
| 15 | `week_sin` | Time (Cyclic) | Sine-encoded week of year |
| 16 | `week_cos` | Time (Cyclic) | Cosine-encoded week of year |
| 17 | `lanka_auto_diesel_price` | Economic | Diesel price (transport cost proxy) |
| 18 | `usd_exchange_rate` | Economic | USD rate (imported input cost proxy) |
| 19 | `diesel_season_int` | Economic × Season | Diesel × season interaction |
| 20 | `no_of_holidays` | Calendar | Number of holidays in the week |
| 21 | `reg_rain` | Weather | Regional average rainfall |
| 22 | `reg_temp` | Weather | Regional average temperature |
| 23–27 | `retail_price_lag_1..4,8` | Retail Lags | 5 lag values (1–4 and 8 weeks back) |
| 28–30 | `reg_rain_lag_1,4,8` | Weather Lags | Rainfall lag at 1, 4, 8 weeks |
| 31–33 | `reg_temp_lag_1,4,8` | Weather Lags | Temperature lag at 1, 4, 8 weeks |
| 34 | `retail_price_roll_4` | Retail Rolling | 4-week rolling mean |
| 35 | `retail_market_enc` | Categorical | Encoded market location |
| 36 | `vegetable_type_enc` | Categorical | Encoded vegetable type |
| 37 | `vegetable_zone_enc` | Categorical | Encoded growing zone |
| 38 | `season_enc` | Categorical | Encoded season (Maha/Yala) |

### Feature Breakdown by Category

```
Farmer Price Features  : 13 (34%)  ← dominant signal
Retail Price Features  :  6 (16%)
Weather Features       :  8 (21%)
Time Features          :  3  (8%)
Economic Features      :  3  (8%)
Categorical Features   :  4 (11%)
Calendar Features      :  1  (2%)
                         ──
Total                  : 38
```

---

## Data Leakage Prevention

| Technique | Implementation |
|:---|:---|
| All lags use `shift()` | Only past values are used — no current-week retail price in features |
| Rolling features use `shift(1).rolling()` | The rolling window starts from **last week**, not current week |
| Time-ordered train/test split | Model never trains on future data |
| `pct_change` is shifted | Momentum calculated from past values only |
| `mean_farmer_price_filled` | Uses current farmer price (available before retail) — not leakage because farmer prices are published before retail |

---

## Script: `retail_price_xgb_lgbm_ensemble.py`

**Location:** [retail_price_xgb_lgbm_ensemble.py](file:///c:/Users/Ranuga/Data%20Science%20Project/5.%20Model%20Building/5.8%20-%20Retail%20Price%20Ensemble%20Models/Scripts/retail_price_xgb_lgbm_ensemble.py)

**Outputs:**
- [xgb_lgbm_ensemble_performance.txt](file:///c:/Users/Ranuga/Data%20Science%20Project/5.%20Model%20Building/5.8%20-%20Retail%20Price%20Ensemble%20Models/Reports/xgb_lgbm_ensemble_performance.txt) — Full performance report
- `xgb_lgbm_ensemble_model.joblib` — Saved model bundle (in `Models/`)

---

*Generated: 2026-03-19*
