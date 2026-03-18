# Stage 6: Advanced Modeling & Regional Weather

This stage explores techniques to move beyond the baseline model by capturing regional supply-side signals and testing alternative mathematical transformations.

## Experiments Conducted

### 1. Target Transformation (Log-Price)
- **Goal**: stabilize variance and handle price spikes better.
- **Results**: Accuracy dropped from 87.22% to 86.52%.
- **Conclusion**: The linear relationship in the raw price data is stronger than the log-log relationship for this specific dataset.

### 2. Regional Weather Enrichment (Successful)
- **Goal**: Capture "Supply-Side" shocks by looking at the weather across the entire production zone (Up Country vs. Low Country).
- **Features**: `regional_rain_avg`, `regional_temp_avg` and their lags.
- **Results**: Improved **R2 Score to 0.8732**.
- **Insight**: `regional_rain_avg_lag_4` became a top 5 feature, confirming that regional droughts/floods 4 weeks ago significantly impact current market prices.

## Files In This Directory

| File | Description |
| :--- | :--- |
| `regional_optimized_model.joblib` | XGBoost model with regional weather features. |
| `regional_weather_performance.txt` | Performance report for the regional enrichment. |
| `optimized_price_model.joblib` | The (less accurate) log-transformed model for reference. |
| `advanced_performance_report.txt` | Comparison report for log-transformation vs baseline. |

---
*Created on: 2026-03-15*
