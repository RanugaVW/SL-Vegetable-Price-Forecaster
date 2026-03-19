import pandas as pd
import numpy as np

print('Loading producer dataset...')
df_prod = pd.read_csv('data/weekly_producer_vegetable_all_location(2008-2024) - weekly.csv')

weeks = [f'W{i}' for i in range(1, 53)]
valid_weeks = [w for w in weeks if w in df_prod.columns]

# Total possible entries vs actual entries
total_cells = len(df_prod) * len(valid_weeks)
null_cells = df_prod[valid_weeks].isna().sum().sum()
print(f"\nMissing Data Snapshot:")
print(f"Total Weekly Data Points Possible: {total_cells:,}")
print(f"Null Values Present: {null_cells:,}")
print(f"Missing Proportion: {(null_cells/total_cells)*100:.2f}%\n")

# Missing per vegetable
print("Top 10 Most Missing Vegetables:")
df_prod['null_count'] = df_prod[valid_weeks].isna().sum(axis=1)
veg_nulls = df_prod.groupby('Items')[['null_count']].sum()
veg_total = df_prod.groupby('Items').size() * len(valid_weeks)
veg_summary = pd.DataFrame({'Total_Slots': veg_total, 'Nulls': veg_nulls['null_count']})
veg_summary['Missing_%'] = (veg_summary['Nulls'] / veg_summary['Total_Slots'] * 100).round(2)
print(veg_summary.sort_values('Missing_%', ascending=False).head(10))

# Missing per location
print("\nTop 10 Most Missing Locations:")
loc_nulls = df_prod.groupby('Location')[['null_count']].sum()
loc_total = df_prod.groupby('Location').size() * len(valid_weeks)
loc_summary = pd.DataFrame({'Total_Slots': loc_total, 'Nulls': loc_nulls['null_count']})
loc_summary['Missing_%'] = (loc_summary['Nulls'] / loc_summary['Total_Slots'] * 100).round(2)
print(loc_summary.sort_values('Missing_%', ascending=False).head(10))
