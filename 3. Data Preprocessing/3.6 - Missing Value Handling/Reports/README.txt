
================================================================================
  MISSING VALUE HANDLING — FOLDER README
  Last Updated : 2026-03-18 23:56:19
================================================================================

This folder contains processed datasets and reports related to missing value
analysis and imputation for the Sri Lanka Vegetable Price Forecasting project.

--------------------------------------------------------------------------------
FILES IN THIS FOLDER
--------------------------------------------------------------------------------

  1. Combined_data.csv
     - Main combined dataset (retail prices + features).
     - farmer_price column was removed (reconstructed separately below).
     - All numeric columns rounded to 2 decimal places.

  2. Farmer Produce Locaation.csv
     - Lookup table mapping each Retail Market + Vegetable pair to:
         * Primary Producer Market Target
         * Secondary Producer Market Target (Fallback)
     - Used to derive mean_farmer_price in the calculated dataset below.

  3. Calculated_Retail_Producer_Prices.csv
     - Derived dataset containing estimated farmer prices for each
       Retail Market, Vegetable, and Week.
     - mean_farmer_price = average of Primary + Secondary producer market
       farmer prices from weekly_producer_vegetable_long_format.csv.
     - Columns: Year_Week, year, week, Retail Market, Vegetable, mean_farmer_price
     - Total rows: 61,152 | Markets: 14 | Vegetables: 12 | Years: 2013-2019

  4. missing price summary report.txt
     - Report generated BEFORE missing value imputation.
     - Covers:
         * retail_price missing % in Combined_data.csv
         * mean_farmer_price overall missing % (7.59%)
         * Breakdown by Vegetable and by Retail Market

  5. after_linear_interpolation_summary.txt
     - Report generated AFTER linear interpolation imputation.
     - Covers:
         * Before: 4,643 missing values (7.59%)
         * After : 2,357 missing values (3.85%)
         * Filled : 2,286 values
         * Breakdown by Vegetable and Retail Market after imputation

  6. seasonal_imputation_validation.png
     - Visual validation of the Seasonal Weekly Mean approach (tested first).
     - Tested on TOMATOES @ Colombo (highest missing %).
     - Result: MAPE = 70.79% — too unreliable. Approach was rejected.

  7. linear_interpolation_validation.png
     - Visual validation of the Linear Interpolation approach.
     - Tested across ALL (Retail Market x Vegetable) groups.
     - Simulated gaps of 1-4 consecutive weeks on known data.

  8. README.txt (this file)

--------------------------------------------------------------------------------
IMPUTATION APPROACH SUMMARY
--------------------------------------------------------------------------------

  STEP 1 — Approach Tested: Seasonal Weekly Mean
    Fill NaN with average of same week-number across all other years
    (same market + vegetable).
    Result on TOMATOES @ Colombo:
      MAE  = 25.67 LKR
      RMSE = 33.07 LKR
      MAPE = 70.79%
    Decision: REJECTED — too inaccurate for volatile vegetables.

  STEP 2 — Approach Selected: Linear Interpolation (≤4 week gaps)
    Fill NaN by linear interpolation within each (Retail Market x Vegetable)
    group, sorted by year and week number. Maximum gap limit = 4 weeks.
    Gaps > 4 weeks are left as NaN intentionally.

    Validation Results (tested across all groups, 1,257 samples):
      Overall MAE  : 11.92 LKR
      Overall RMSE : 22.62 LKR
      Overall MAPE : 17.21%

    Accuracy by Gap Size:
      Gap 1 week  — MAE:  6.42 | MAPE: 12.11%
      Gap 2 weeks — MAE:  9.98 | MAPE: 15.77%
      Gap 3 weeks — MAE: 12.98 | MAPE: 16.49%
      Gap 4 weeks — MAE: 13.39 | MAPE: 19.63%

    Decision: ACCEPTED — error stays moderate and increases predictably with gap.

  OUTCOME:
    Missing % reduced from 7.59% → 3.85% (3.74 percentage point improvement).
    Remaining 2,357 NaN values are large-gap rows (mainly TOMATOES & GREEN BEANS)
    that will be handled at the modelling stage.

================================================================================
