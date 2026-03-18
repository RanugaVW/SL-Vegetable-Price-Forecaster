# Initial Dataset Summary

This folder contains the preliminary exploratory data analysis (EDA) of the raw vegetable price dataset.

## Contents

- `analyze_initial_missing.py`: Calculates the overall missing price percentage for the raw 2008-2020 dataset.
- `initial_data_missing_report.txt`: A summary report of price missingness (Baseline: 26.30%).
- `missing_cells_complete_summary.csv`: A row-by-row breakdown of missing weeks for every year/location/item.
- `total_cells_by_vegetable.csv`: Data points count for each vegetable type.

## Key Findings

The raw dataset (`weekly_vegetable_all_location HARTI - weekly.csv`) suffers from significant missingness in the early years. This led to the strategy of focusing on the 2013-2019 window and specific vegetables to ensure a high-quality dataset for time-series forecasting.
