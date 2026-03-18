# SL-Vegetable-Price-Forecaster

**National Level Project:** This is a national-level project where 12 vegetable type data, 14 market locations, and prices are collected from HARTI (Horticultural and Agronomic Research and Training Institute).

A Data Science project focused on forecasting weekly vegetable prices in Sri Lanka using historical market data and high-resolution weather variables.

## Project Overview

This project aims to build a robust predictive model that estimates the weekly price of upcountry and low-country vegetables across 14 major economic centers in Sri Lanka. It leverages 7 years (2013-2019) of historical price data combined with meteorological data (Mean Temperature and Rain Sum) aggregated from the specific growing regions (origins) for each market.

## Repository Structure

```
├── Raw Data/                     → Original HARTI CSV files
├── 1. Initial Data Summary/      → Missingness analysis & reports
├── 2. Weather Data/              → Weather fetching, geocoding & aggregation
├── 3. Data Preprocessing/        → 7-step data transformation pipeline
│   ├── 3.1 - Pivot to Long Format
│   ├── 3.2 - Filtering and Cleanup
│   ├── 3.3 - Weather Merge
│   ├── 3.4 - External Data Merge
│   ├── 3.5 - Producer Price Processing
│   ├── 3.6 - Missing Value Handling
│   └── 3.7 - Combining Datasets
├── 4. Data Visualization/        → EDA & analysis charts
│   ├── 4.1 - Initial EDA
│   ├── 4.2 - Farmer vs Retail Price Analysis
│   └── 4.3 - Rain Lag Analysis
└── 5. Model Building/            → 9-step modeling pipeline
    ├── 5.1 - Baseline XGBoost Model
    ├── 5.2 - Regional Weather Enrichment
    ├── 5.3 - Economic-Seasonal Interactions
    ├── 5.4 - Error Analysis
    ├── 5.5 - Cyclical Time Encoding
    ├── 5.6 - Dynamic Lag Selection
    ├── 5.7 - Ensemble Model (Producer Prices)
    ├── 5.8 - Retail Price Ensemble Models
    └── 5.9 - Model Validation
```

Each sub-folder is organized by file type: `Scripts/`, `Outputs/`, `Reports/`, `Models/`, `Datasets/`, `Charts/`, `Notebooks/`.

## Current Progress

- **Data Integration:** Historical price data (2008-2020) has been cleaned and filtered to a target window (2013-2019).
- **Feature Engineering:** Integrated weather variables based on a custom "Origin Mapping Matrix" which maps markets to the specific districts where their vegetables are actually grown.
- **Completeness:** Verified that the dataset contains exactly 52 weeks per year for every location and vegetable type (61,152 rows).
- **Analysis:** Baseline missing price analysis completed (Initial: 26.3% missing, Current Merged: 5.37% missing).

## Challenges & Strategies

### 1. Inconsistent Location Data

**Challenge:** Many rural locations from the price dataset did not return results in standard geocoding APIs.
**Strategy:** Implemented a manual coordinate mapping fallback for rural areas (e.g., Ambewela, Sooriyawewa) using exact Lat/Lon coordinates to ensure 100% weather data coverage.

### 2. High Initial Missingness

**Challenge:** The raw dataset had 26.3% missing price values, predominantly in earlier years (2008-2012).
**Strategy:** Focused the project scope on 2013-2019 and filtered for the 12 most consistent vegetable types, reducing missingness to 5.37% while maintaining data integrity.

### 3. Date & Week Alignment

**Challenge:** Standard calendars don't always align with the "Friday-Thursday" week cycle used in Sri Lankan market reporting.
**Strategy:** Created a custom mathematical week calculator that resets and starts from the first Friday of every year, strictly enforcing a 52-week annual structure.

### 4. Market Complexity (Origin Mapping)

**Challenge:** Markets in Colombo or Kandy do not rely on local weather; their prices depend on weather in the "Growing Origins" (e.g., Nuwara Eliya for Leeks).
**Strategy:** Developed an "Origin Mapping Matrix" to aggregate Mean Temperature and Rain Sum specifically from the districts that supply each specific market center.

## How to Run

All scripts use **relative paths**, ensuring the project is portable. Simply clone the repository and run scripts from their respective directories or the root.
