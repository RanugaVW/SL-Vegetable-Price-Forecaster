# Stage 5: Modeling and Evaluation

This directory contains the final datasets, the trained model, and the performance reports for the vegetable price prediction project.

## Files In This Directory

| File | Description |
| :--- | :--- |
| `Train_forecasting.csv` | Training dataset with lag features and encoded categories. |
| `Test_forecasting.csv` | Test dataset (last 20% of time series) for evaluation. |
| `vegetable_price_model.joblib` | The trained XGBoost Regressor model. |
| `model_performance_report.txt` | Technical metrics (R2, RMSE, MAPE). |
| `price_missingness_report.txt` | Analysis of price completeness relative to raw data. |

## Model Performance Summary

- **R2 Score**: 0.8628 (The model explains 86.2% of price variance)
- **Accuracy**: 87.22% (Based on 1 - MAPE)
- **RMSE**: 36.15

## Understanding Feature Importances

The model uses the following features to make its predictions. The "Importance" score represents how much impact each feature has on the final price estimate.

### 1. Market History (The strongest signals)
*   **Price Lags (1 & 2 weeks)**: The most critical factors. Current prices are highly dependent on what happened in the last 14 days.
*   **4-Week Rolling Mean**: Provides the broader monthly trend, helping the model understand if we are in a high-price or low-price cycle.

### 2. Delayed Weather Impact
*   **Rain and Temperature Lags (3-4 weeks)**: The model found that weather from a month ago is more important than immediate weather. 
    *   *Why?* It accounts for crop growth cycles. Extreme weather 4 weeks ago affects the supply available in the market *today*.

### 3. Economic and Transport Costs
*   **Diesel Price**: Tracks the cost of moving vegetables from farms (e.g., Nuwara Eliya) to central markets (e.g., Colombo).
*   **USD Exchange Rate**: Tracks the cost of imported inputs like fertilizers and seeds.

---
*Created on: 2026-03-15*
