import pandas as pd
import numpy as np

def parse_week(w):
    try:
        return int(str(w).lower().replace('w', '').strip())
    except:
        return np.nan

df = pd.read_csv('data/main_filled.csv')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['week_num'] = df['week'].apply(parse_week)
df['is_missing'] = df['price'].isna()

# Check local seasonal absence (missing > 60% of years for a specific vegetable, location, and week)
loc_veg_week_missing = df.groupby(['location', 'vegetable_type', 'week_num'])['is_missing'].mean().reset_index()

seasonal_absences = loc_veg_week_missing[loc_veg_week_missing['is_missing'] >= 0.60]

print(f"Number of (location+vegetable+week) combinations strongly missing year-over-year: {len(seasonal_absences)}")

if len(seasonal_absences) > 0:
    print(seasonal_absences.head(20))
    df = df.merge(loc_veg_week_missing.rename(columns={'is_missing': 'loc_missing_rate'}), on=['location', 'vegetable_type', 'week_num'], how='left')
    
    def classify(row):
        if not row['is_missing']: return 'Present'
        if row['loc_missing_rate'] >= 0.60: return 'Local Seasonal Absence'
        return 'True Missing Data'
        
    df['classification'] = df.apply(classify, axis=1)
    print("\nRevised Classification:")
    print(df['classification'].value_counts())
else:
    print("No localized seasonal absences detected either.")
