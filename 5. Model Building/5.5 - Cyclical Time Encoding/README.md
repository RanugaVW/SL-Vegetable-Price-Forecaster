# Stage 9: Cyclical Time Encoding

In this stage, we treated the week number as a cyclical feature (Sine and Cosine) rather than a linear number.

## Results

- **R2 Score**: **0.8746** (Improved from Stage 7 baseline of 0.8732)
- **Accuracy (1-MAPE)**: **87.38%** (Slight decrease from 87.57%)

## Key Insight
The improvement in R2 indicates that the model is better at explaining the overall price variance across the year boundary (Week 52 to Week 1). However, the slight dip in MAPE suggests that for specific high-volatility weeks, the sine/cosine transition might be too smooth.

The cyclical features remain valuable as they prevent the model from treating December and January as fundamentally "disconnected" time periods.

---
*Created on: 2026-03-15*
