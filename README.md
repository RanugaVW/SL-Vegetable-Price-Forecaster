# SL-Vegetable-Price-Forecaster

**National Level Project:** This is a national-level project where 12 vegetable type data, 14 market locations, and prices are collected from HARTI (Horticultural and Agronomic Research and Training Institute).

A Data Science project focused on forecasting weekly vegetable prices in Sri Lanka using historical market data and high-resolution weather variables.

## Project Overview

This project aims to build a robust predictive model that estimates the weekly price of upcountry and low-country vegetables across 14 major economic centers in Sri Lanka. It leverages 7 years (2013-2019) of historical price data combined with meteorological data (Mean Temperature and Rain Sum) aggregated from the specific growing regions (origins) for each market.

## Current Progress

- **Data Integration:** Historical price data (2008-2020) has been cleaned and filtered to a target window (2013-2019).
- **Feature Engineering:** Integrated weather variables based on a custom "Origin Mapping Matrix" which maps markets to the specific districts where their vegetables are actually grown.
- **Completeness:** Verified that the dataset contains exactly 52 weeks per year for every location and vegetable type (61,152 rows).
- **Analysis:** Baseline missing price analysis completed (Initial: 26.3% missing, Current Merged: 5.37% missing).

## Repository Structure

- `Initial Dataset Summary/`: Preliminary analysis scripts and reports for raw data.
- `Weather Data/`: Scripts for geocoding, fetching Open-Meteo historical data, and weekly aggregation.
- `Data Pre Processing/`: The three-stage processing pipeline (Pivot -> Filter -> Merge).
- `analyze_removals.py`: Analysis of data dropped during the filtering process.
- `.gitignore`: Configured to exclude Python system files and cache.

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

