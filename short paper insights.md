# Agricultural Price Prediction Modeling: Short Paper Insights

*An analysis of machine learning approaches applied to forecasting agricultural retail prices using weather, macroeconomic dynamics, and seasonal trends.*

## 1. The Diesel Price Paradox: Model Insights

During hyperparameter tuning and feature selection for our XGBoost and LightGBM ensemble models, we noticed an unexpected phenomenon. If we strictly look at the raw data and correlation matrices, **Diesel Prices do not show a strong, direct linear correlation** with weekly vegetable prices. Vegetable prices fluctuate wildly with seasons and weather, while diesel prices remain mostly flat for long periods before making sudden, rigid jumps.

Because of this weak visual correlation, we hypothesized that removing the Diesel Price feature (`lanka_auto_diesel_price`) might simplify the model without hurting performance. However, **when we removed the diesel feature and ran Optuna tuning, our overall model accuracy decreased.**

### Why does a feature with weak linear correlation improve our predictive power so much?

Tree-based models are uniquely positioned to extract value from features that linear models would discard. There are three primary reasons why XGBoost and LightGBM benefit significantly from the Diesel Price data:

1. **Non-Linear "Regime" Shifting (Step Functions):** 
   Unlike traditional linear models, tree-based algorithms do not map straight-line relationships. Instead, they make decisions by "splitting" data. Since diesel prices move in sudden macroeconomic step-functions, the decision trees use the diesel price to split the timeline into different "epochs" or "cost regimes" (e.g., *Era of 100 LKR Diesel* vs *Era of 400+ LKR Diesel*). The tree then sets a different baseline expectation for retail prices in each era.
2. **Feature Interaction with the Farmer-to-Retail Spread:** 
   The gap between the **Farmer Price** and the **Retail Price** is heavily driven by logistics, transportation, and middleman margins. By keeping the diesel price, the model learns complex feature interactions: it understands that when the farmer price is 'X' and diesel is 'Y', the retail markup needs to be 'Z'. Without diesel, the model cannot mathematically explain why the transportation markup systematically widened during later years.
3. **Proxy for Macroeconomic Inflation:** 
   Diesel acts as an excellent proxy for broader country-wide inflation. When diesel jumps, parallel costs (packaging, physical labor, market fees) often inflate shortly after. The model relies on this step-change behavior, alongside the USD Exchange Rate, to scale prices appropriately across the multi-year dataset.

---

## 2. Statistical Outliers and Noise Analysis

To understand the dataset's volatility, calculating the Interquartile Range (IQR) on the complete 60,000+ row dataset reveals the extreme price spikes for each vegetable type.

### How Many Outliers Exist?
Based on our multi-year bounding box calculations, the overall noise and statistical outlier percentage is **exceedingly small**. Examples from the distribution:
* **Green Chillies:** ~3.85% outliers (Highest volatility)
* **Leeks:** ~3.65% outliers
* **Tomatoes:** ~2.90% outliers 
* **Cabbage & Snake Gourd:** < 0.20% outliers

In almost all cases, outlier density sits safely between **1% and 4%** of the total observations.

### Should We Consider or Drop These Outliers?
A common intuition is: *If outliers represent such a tiny fraction of the dataset (< 4%), shouldn't we just delete them to get cleaner regression curves?*

**The absolute answer is: We must keep them.** Here is why:

1. **They represent reality, not errors:** In agricultural economics, these mathematical "outliers" are rarely sensor errors or typos. They represent **extreme, real-world supply shocks** (e.g., historic floods destroying a harvest, extreme drought, or sudden logistical freezes). Deleting these points would mean blinding the AI to the agricultural realities it is supposed to predict.
2. **Algorithmic Immunity:** Because our primary predictive architecture relies on gradient-boosted trees (XGBoost and LightGBM), the model is natively highly resistant to extreme values. Trees split at threshold boundaries and aren't skewed by the *magnitude* of an extreme value the way standard Linear Regression would be. 
3. **Logarithmic Scaling Transformation (`np.log1p`):** In our pipeline, we applied a log transformation to the target variable (`retail_price`). This naturally compresses violent price spikes (e.g., a sudden jump from 150 LKR to 800 LKR for Green Chillies), ensuring that the predictive gradients aren't overly penalized by learning from the outliers.

By preserving this "noise", the ensemble manages to achieve **~90.8% MAPE accuracy** because it learns the exact boundaries of extreme events rather than pretending they don't exist.