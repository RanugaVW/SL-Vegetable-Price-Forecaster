# Agricultural Price Prediction Modeling: Short Paper Insights

*An analysis of machine learning approaches applied to forecasting agricultural retail prices using weather, macroeconomic dynamics, and seasonal trends.*

## 1. Initial Data Reduction & Missing Value Mitigation

Before deploying any advanced predictive models, we faced a catastrophic missing data problem in the raw agricultural dataset. The initial dataset containing the historical retail prices was vast, spanning numerous years, minor vegetables, and smaller regional markets. 

However, a strict density analysis revealed severe data sparsity:
* **Total Price Cells** (52 weeks per row): 746,304
* **Missing Price Cells**: 196,310
* **Initial Missing Percentage**: **26.30%**

Proceeding with over a quarter of the core target variable missing would severely compromise any machine learning algorithm. Instead of aggressively attempting to synthesize or blindly impute 196k data points, we implemented a rigorous **Data Reduction Strategy**. 

We systematically profiled the entire dataset and filtered it down strictly to the **12 most structurally consistent vegetables** and the **14 most robust regional market locations** spanning a heavily documented timeframe (2013–2019). 

This strategic Data Reduction Plan is paring successfully pruned the unreliable "fringe" data. By narrowing our focus to this core, high-fidelity grid, we achieved an immediate and massive stabilization:
* **Revised Total Rows**: 61,152
* **Remaining Missing Rows**: 3,286
* **Revised Missing Percentage**: **5.37%**

By reducing the missing matrix from 26.3% down to 5.37% natively (before applying any imputation), we established a fundamentally sound foundation for our gradient-boosted trees.

---

## 2. The Diesel Price Paradox: Model Insights

During hyperparameter tuning and feature selection for our XGBoost and LightGBM ensemble models, we noticed an unexpected phenomenon. If we strictly look at the raw data and correlation matrices, **Diesel Prices do not show a strong, direct linear correlation** with weekly vegetable prices. Vegetable prices fluctuate wildly with seasons and weather, while diesel prices remain mostly flat for long periods before making sudden, rigid jumps.

Because of this weak visual correlation, we hypothesized that removing the Diesel Price feature (`lanka_auto_diesel_price`) might simplify the model without hurting performance. However, **when we removed the diesel feature and ran Optuna tuning, our overall model accuracy decreased.**

![Macroeconomic Indicators vs Retail & Farmer Prices: Beetroot](./4.%20Data%20Visualization/USD_Diesel_Correlation/Charts/BEETROOT_trends.png)

### Why does a feature with weak linear correlation improve our predictive power so much?

Tree-based models are uniquely positioned to extract value from features that linear models would discard. There are three primary reasons why XGBoost and LightGBM benefit significantly from the Diesel Price data:

1. **Non-Linear "Regime" Shifting (Step Functions):** 
   Unlike traditional linear models, tree-based algorithms do not map straight-line relationships. Instead, they make decisions by "splitting" data. Since diesel prices move in sudden macroeconomic step-functions, the decision trees use the diesel price to split the timeline into different "epochs" or "cost regimes" (e.g., *Era of 100 LKR Diesel* vs *Era of 400+ LKR Diesel*). The tree then sets a different baseline expectation for retail prices in each era.
2. **Feature Interaction with the Farmer-to-Retail Spread:** 
   The gap between the **Farmer Price** and the **Retail Price** is heavily driven by logistics, transportation, and middleman margins. By keeping the diesel price, the model learns complex feature interactions: it understands that when the farmer price is 'X' and diesel is 'Y', the retail markup needs to be 'Z'. Without diesel, the model cannot mathematically explain why the transportation markup systematically widened during later years.
3. **Proxy for Macroeconomic Inflation:** 
   Diesel acts as an excellent proxy for broader country-wide inflation. When diesel jumps, parallel costs (packaging, physical labor, market fees) often inflate shortly after. The model relies on this step-change behavior, alongside the USD Exchange Rate, to scale prices appropriately across the multi-year dataset.

---

## 3. Statistical Outliers and Noise Analysis

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

---

## 4. Resolving Spatial Weather Mismatches: The Origin-Averaging Approach

One of the most critical and mathematically challenging steps in our preprocessing pipeline was correctly aligning meteorological data with market prices. 

### The Naive Trap
A standard, naive approach to this problem would be to fetch the local weather data for the specific retail market location. For example, trying to predict the price of carrots at the Colombo retail market by looking at the rainfall in Colombo. **This approach is fundamentally flawed and virtually useless** because the commercial vegetables are not grown in the urban retail centers where they are sold. 

### The Origin-Averaging Solution
To handle this geographic anomaly, we reverse-engineered the supply chain to map the actual weather affecting the crops:
1. **Identify True Origins ($X$):** For every single retail market, we tracked down precisely where each specific vegetable type is actually cultivated and brought from. If a specific retail market imports a crop from multiple different agricultural zones, we mapped all of those supply locations (locations $X_1, X_2, \dots, X_n$).
2. **Fetch Daily Origin Weather:** We pulled the complete historical daily weather metrics (such as `rain_sum` and `mean_apparent_temperature`) exclusively for those true cultivation source locations ($X$).
3. **Compute the Geographic Mean:** We then aggregated the data by taking the mathematical mean of the daily weather across *all* supplying locations $X$ for that specific vegetable and destination market. 

### Why This Was a Game-Changer
This strong data-engineering step dramatically shifted the model's accuracy. By calculating the average weather conditions strictly across the agricultural origins, the AI could finally accurately measure the true vegetative supply shocks. The model no longer cared if it was raining in Colombo; instead, it was able to mathematically link the fact that a severe drought concentrated across the mean origin zones ($X$) directly collapses the harvest supply chain, resulting in an accurate prediction of a massive retail price spike in Colombo weeks later.

---

## 5. Recovering the "Ghost" Farmer Prices: The Dual-Origin Imputation

A massive structural problem surfaced while processing the initial producer dataset (`weekly_producer_vegetable_all_location(2008-2024).xlsx`). 

### The Void in Urban Centers
Just like the weather anomaly, major commercial hubs (most notably Colombo) have an extensive log of **Retail Prices**, but absolutely zero raw **Farmer/Producer Prices**. This is simply because mass-scale agricultural production doesn't happen inside these major cities. Consequently, the initial raw dataset faced devastating data sparsity. Looking at the raw producer tracking document (`weekly_producer_vegetable_all_location(2008-2024).xlsx`), an astonishing **33.82% of the entire dataset was missing** (equating to 42,317 empty cells out of a total 125,115 cells), with some specific urban location columns missing upwards of **43%** of their required data.

Leaving these gaps empty would disconnect the algorithms from arguably the most powerful predictive feature: the Farmer-to-Retail Spread.

### The Dual-Origin Solution
To resolve this safely and accurately without guessing synthetic data, we designed a targeted imputation strategy:

1. **Identify Top Logistics Corridors:** For every retail market block missing a direct farmer price, we queried the historical records to identify the **top two main agricultural producer markets** supplying that exact vegetable to the specific retail location.
2. **Mathematical Aggregation:** We took the mean of the observed Farmer Prices from those exact two top producing locations.
3. **Target Imputation:** We projected that calculated mean into the blank value of the urban retail center, simulating the actual initial wholesale price point for that city.

### The Impact: From 33.82% to 3.85% Missing
By employing this direct supply-chain bridging, we effectively solved the devastating initial missing value challenge. 
1. **The Dual-Origin Transformation:** Instead of trying to build complex mathematical imputations mimicking unknown farmer interactions, we simply mapped the actual "import cost" mathematically. This massive structural fix immediately reduced the dataset's missing farmer prices from a crippling **33.82% down to just 7.59%** (leaving only 4,643 missing rows out of 61,152 total records).
2. **Targeted Linear Interpolation:** To fix the remaining 7.59% gap (usually caused by short-term logistical breaks like strikes or minor recording errors), we applied strict Time-Series Linear Interpolation, capping it strictly at $\le 4$ consecutive weeks. This ensured no synthetic "guesses" were made for long-term data deaths. This final phase mathematically fused the remaining data, reducing the final missing value footprint down to an incredibly stable **3.85%** (only 2,357 remaining unfillable rows).

By deploying this dual-layered strategy, we successfully transformed a fractured timeline into a nearly flawless unified chain. This gave the AI the continuous, robust Farmer Price lags required for our high-impact predictive momentums.

---

## 6. Real-World Applications: Decoding the Retail Markup Spread

The ultimate goal of forecasting agricultural economics is not merely to predict a number, but to understand the systemic forces driving that number. Because our model architecture explicitly bridges the **Farmer Price** to the predicted **Retail Price**, we can deploy it as an anomaly detection and policy-planning tool.

### Identifying the Root Cause of Price Inflation
If we feed the current Farmer Price into the model along with our engineered macro-features, and the model predicts a Retail Price that is **exponentially higher** than the usual historical baseline, we can deconstruct the decision tree paths (using tools like SHAP or feature importance weights) to isolate the exact cause of the markup shock:

1. **Logistical Paralysis (Diesel & USD):** If the spread between the farmer and retail cost balloons, the model can explicitly trace this back to recent sudden spikes in the `lanka_auto_diesel_price` or `usd_exchange_rate`. It proves that the farmers are not earning more money; rather, the transportation and middleman costs have geometrically expanded.
2. **Lagged Weather Traumas:** Because our model tracks 1-week, 4-week, and 8-week lagged weather averages over the origin zones, the model can identify if an unusual predicted price spike is the "echo effect" of a massive drought or systemic flooding that occurred a month ago across rural farms. 
3. **Momentum Cascades:** Using our custom mathematical features (`retail_price_momentum_1_4`), the model can highlight cases where consumer panic or seasonal scarcity has generated an unnatural, temporary hyper-inflation bubble that is independent of baseline inflation.

By analyzing *why* the algorithm predicted a massive price difference, stakeholders can determine if urban food inflation is being caused by a true agricultural supply shortage, a volatile macroeconomic bottleneck, or mere logistical exploitation.

---

## 7. Conclusion: Solving a National-Level Economic Enigma

When this project was initially conceptualized alongside national agricultural institutions like HARTI, the problem was framed as exceptionally challenging, if not completely volatile. Because Sri Lanka's core commercial vegetable market operates entirely recursively—meaning these specific crop types are practically *never* imported from other countries to offset shortages—the retail prices are at the absolute mercy of localized hyper-volatility. Every local drought, every national fuel strike, and every domestic monsoon directly and permanently alters the price of food. 

The core problem we set out to solve was: **Can we bring mathematical predictability to an isolated, natively volatile food supply chain?**

Through methodical data engineering, we unified a completely fractured national timeline. By aggregating true *Origin Zone weather* instead of urban rainfall, executing *Dual-Market Imputations* to recover 33%+ missing farmer base costs, and recognizing macroeconomic step-functions like Diesel prices, we constructed an agricultural diagnostic engine.

Running our finalized, tuned LightGBM & XGBoost ensemble architecture over our strictly curated master dataset yielded exceptional results for an economic forecasting system:
* **R2 Validation Score:** 0.9281
* **Overall Accuracy (1 - MAPE):** 90.84%

![Scatter Accuracy on Test Data](./5.%20Model%20Building/5.8%20-%20Retail%20Price%20Ensemble%20Models/XGBoost%20%2B%20LightBGM/Charts/Scatter_Accuracy_Test_Data.png)

![Validation Time-Series Tracking](./5.%20Model%20Building/5.8%20-%20Retail%20Price%20Ensemble%20Models/XGBoost%20%2B%20LightBGM/Charts/Validation_TimeSeries_Tracking.png)

### Final Thoughts
We effectively proved that agricultural prices in an isolated national economy are not random. The ~91% accuracy threshold achieved by the ensemble model verifies that violent price spikes are highly predictable cascading events. It demonstrates that the gap between what farmers earn and what the public pays is mathematically governed by strict underlying laws of momentum, lagged biological weather trauma, and macroeconomic transport costs. Ultimately, this research provides a predictive blueprint capable of forecasting national food security risks weeks before they hit the retail markets.