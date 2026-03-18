# Stage 8: Error Analysis

This stage focuses on identifying the specific "weak points" of the Stage 7 champion model.

## Error Breakdown

### 1. By Vegetable Type (MAPE %)
The model struggles most with highly volatile vegetables:
- **TOMATOES**: 18.49%
- **GREEN CHILLIES**: 16.27%
- **GREEN BEANS**: 15.57%

The most stable predictions (lowest error) are for:
- **ASH PLANTAINS**: 6.70%
- **SNAKE GOURD**: 10.17%

### 2. By Location (MAPE %)
- **Hambanthota**: 14.74%
- **Thambuththegama**: 14.58%
- **Embilipitiya**: 13.84%

## Key Conclusions
- **Supply Shocks**: The high error in Tomatoes and Chillies is likely due to their sensitivity to immediate supply shocks (pests, sudden transport blocks) which aren't fully captured by 4-week lags.
- **Seasonality**: The error in certain locations suggests that regional micro-climates might need more granular weather data or cyclical year-end adjustments.

---
*Created on: 2026-03-15*
