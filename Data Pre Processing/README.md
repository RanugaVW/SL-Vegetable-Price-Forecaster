# Data Pre Processing Pipeline

This directory manages the transformation of raw vegetable prices into a model-ready dataset.

## Pipeline Stages

### Stage 1: Pivot Matrix

- **Input:** Wide-format monthly/weekly CSV.
- **Process:** Pivots the 52 week columns into a "Long Format" (Year_Week ID format).
- **Result:** `Weekly_vegetable_long_format.csv`.

### Stage 2: Filtering & Cleanup

- **Process:** Filters for the target years (2013-2019), specific 14 locations, and 12 vegetable types.
- **Logic:** `filter_dataset.py` ensures we only train models on high-quality, relevant data segments.
- **Findings:** `removal_analysis.txt` documents the missingness of data that was either kept or discarded.

### Stage 3: Feature Engineering (Weather Merge)

- **Process:** `merge_weather_vegetables.py` performs a complex merge based on the **Origin Mapping Matrix**.
- **Challenges:** Vegetables in Colombo are affected by weather in Nuwara Eliya or Jaffna, not Colombo itself.
- **Solution:** The merge logic identifies if a vegetable is "Upcountry" (UP) or "Lowcountry" (LOW) and then aggregates weather from the specific growing regions mapped for that destination market.

## Final Output

The final product is `Final_Merged_Dataset.csv`, which contains 61,152 rows (Exactly 52 weeks per Year/Loc/Veg group) with price and origin-weather features.
