# Model Insights: The Diesel Price Paradox

## The Problem 1

During our hyperparameter tuning and feature selection with the XGBoost and LightGBM ensemble models, we noticed an interesting phenomenon:
If we look at the raw data and line charts, **Diesel Prices do not show a strong, direct linear correlation** with weekly vegetable prices. Vegetable prices fluctuate wildly with seasons and weather, while diesel prices remain mostly flat for long periods before making sudden, rigid jumps. 

Because of this weak visual correlation, we hypothesized that removing the Diesel Price feature (`lanka_auto_diesel_price`) might simplify the model without hurting performance. However, **when we removed the diesel feature and ran Optuna tuning, our overall model accuracy decreased.**

Why does a feature with weak linear correlation improve our predictive power so much?

---

## The Explanation (Why this happens)

Tree-based models are able to extract value from features that linear models would struggle with. There are three primary reasons why XGBoost and LightGBM benefit significantly from the Diesel Price data:

### 1. Non-Linear "Regime" Shifting (Step Functions)
Unlike traditional linear models, tree-based models (XGBoost/LightGBM) do not look for straight-line relationships. Instead, they make decisions by "splitting" data. Since diesel prices move in sudden macroeconomic step-functions, the decision trees use the diesel price to split the timeline into different "epochs" or "cost regimes" (e.g., *Era of 100 LKR Diesel* vs *Era of 400+ LKR Diesel*). The tree then sets a different baseline expectation for retail prices in each era.

### 2. Feature Interaction with the Farmer-to-Retail Spread
The gap between the **Farmer Price** and the **Retail Price** is largely driven by logistics, transportation, and middleman margins. Even if diesel doesn't dictate the week-to-week weather supply shocks, it heavily influences the transportation cost. 
By keeping the diesel price, the model learns the interaction between features: it figures out that when the farmer price is 'X' and diesel is 'Y', the retail markup needs to be 'Z'. Without diesel, the model cannot mathematically explain why the transportation markup suddenly widened during later years.

### 3. Proxy for Macroeconomic Inflation
Diesel is an excellent proxy for broader country-wide inflation. When diesel jumps, everything else (packaging, physical labor, market fees) goes up shortly after. The model relies on this step-change behavior, alongside the USD Exchange Rate, to scale prices appropriately across the multi-year dataset.

---

## Visual Proof

Look at the chart below for Beetroot. Notice that while the retail price (green line) goes through violent seasonal cycles, the Diesel Price (red dotted line) behaves like a staircase. 

The models aren't trying to trace the green line directly using the red line. Instead, the model uses the red steps to establish the **floor cost** (the underlying base cost of the economy) at that exact point in time. When you remove diesel, you blind the model to when those "floor" shifts occurred.

![Macroeconomic Indicators vs Retail & Farmer Prices: Beetroot](../4.%20Data%20Visualization/USD_Diesel_Correlation/Charts/BEETROOT_trends.png)
