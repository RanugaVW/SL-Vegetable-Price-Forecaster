# Weather Data

Everything related to meteorological data acquisition and processing.

## Folder Structure

- `Daily Data/`: Contains the high-resolution daily weather variables (Rain Sum, Mean Apparent Temperature).
- `Weekly Data/`: Contains the aggregated weekly weather metrics aligned with the vegetable reporting cycle.

## Automation & Fetching

- `fetch_weather.py`: Uses Open-Meteo Geocoding to find locations and download historical daily data from the Archive API.
- `fetch_missing_weather.py`: Handles hardcoded fallback coordinates for rural Sri Lankan locations that the Geocoding API missed.
- `aggregate_weekly.py`: Converts daily metrics into weekly ones.

## The Friday-Thursday Concept

A critical requirement was aligning weather data with the vegetable price weeks.

- **Concept:** Every week starts on a Friday and ends on a Thursday.
- **Implementation:** The `aggregate_weekly.py` script identifies the first Friday of each year and groups daily data into 52 strict 7-day blocks.

## Challenges Fixed

- **Geocoding Rural Areas:** Locations like Ambewela were geocoded manually to ensure precision.
- **Missing Origins:** Fetched additional weather data for "Growing Origin" locations that weren't in the original market list.
