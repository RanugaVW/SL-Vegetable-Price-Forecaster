# Stage 7: Economic-Seasonal Interactions & Feature Optimization

This stage represents the current **Champion Model**, achieving the project's highest accuracy by combining regional supply signals with the interaction of economic costs and growing seasons.

## Key Improvements

### 1. Economic-Seasonal Interactions
- **Hypothesis**: Diesel prices and USD exchange rates affect the market differently depending on the growing season (Maha vs. Yala).
- **Implementation**: Created interaction terms (`diesel_price * seasonality`).
- **Result**: These features became highly impactful, improving the model's ability to handle economic fluctuations.

### 2. Feature Optimization & Cleanup
- **Numeric Week conversion**: Converted `week` (string) to `week_num` (1-52). This became one of the top 3 signals, as it allows the model to understand the exact point in the annual price cycle.
- **Redundancy removal**: Dropped redundant text columns like `year` and `week` to force the model to focus on the numeric signals.

## Performance Snapshot

- **Accuracy (1-MAPE)**: **87.57%** (Current Peak)
- **R2 Score**: **0.8732**
- **RMSE**: **34.75**

## Files In This Directory

| File | Description |
| :--- | :--- |
| `economic_seasonal_optimized_model.joblib` | **Final Champion Model**. |
| `economic_seasonal_performance.txt` | Technical metrics and feature importance for Step 7. |

---
*Created on: 2026-03-15*
