# Stage 10: Dynamic Lag Selection

In this stage, we extended the model's memory by adding 8-week lags for regional weather and prices.

## Results

- **Accuracy (1-MAPE)**: **87.69%** (Improved from Stage 7/9)
- **R2 Score**: **0.8727**

## Key Insight
The model's performance was boosted by the inclusion of **Lag 8 (2 months ago)** features. Specifically, `reg_temp_lag_8` became one of the top 5 most important features. 

This confirms the "growth cycle" hypothesis: The weather during the early stages of a vegetable's growth (about 8 weeks before harvesting) has a measurable impact on the market price today. By giving the model a longer "memory," we improved its ability to anticipate supply fluctuations for slower-growing crops.

---
*Created on: 2026-03-15*
