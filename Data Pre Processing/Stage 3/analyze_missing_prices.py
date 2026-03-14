import pandas as pd
import os

# Define relative paths
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "Final_Merged_Dataset.csv")
output_report = os.path.join(base_dir, "missing_prices_report.txt")

print(f"Loading {file_path}...")
df = pd.read_csv(file_path)

# Calculate metrics
total_rows = len(df)
missing_count = df['price'].isna().sum()
missing_percent = (missing_count / total_rows) * 100

# Breakdown by Year
year_missing = df.groupby('year')['price'].apply(lambda x: x.isna().sum()).reset_index()
year_total = df.groupby('year').size().reset_index(name='total')
year_stats = year_missing.merge(year_total, on='year')
year_stats['missing_percent'] = (year_stats['price'] / year_stats['total'] * 100).round(2)

# Breakdown by Vegetable Type
veg_missing = df.groupby('vegetable_type')['price'].apply(lambda x: x.isna().sum()).reset_index()
veg_total = df.groupby('vegetable_type').size().reset_index(name='total')
veg_stats = veg_missing.merge(veg_total, on='vegetable_type')
veg_stats['missing_percent'] = (veg_stats['price'] / veg_stats['total'] * 100).round(2)
veg_stats = veg_stats.sort_values(by='missing_percent', ascending=False)

# Breakdown by Location
loc_missing = df.groupby('location')['price'].apply(lambda x: x.isna().sum()).reset_index()
loc_total = df.groupby('location').size().reset_index(name='total')
loc_stats = loc_missing.merge(loc_total, on='location')
loc_stats['missing_percent'] = (loc_stats['price'] / loc_stats['total'] * 100).round(2)
loc_stats = loc_stats.sort_values(by='missing_percent', ascending=False)

# Writing the report
with open(output_report, 'w', encoding='utf-8') as f:
    f.write("MISSING PRICE ANALYSIS (STAGE 3 MERGED DATASET)\n")
    f.write("==============================================\n\n")
    f.write(f"Total Rows: {total_rows}\n")
    f.write(f"Missing Price Rows: {missing_count}\n")
    f.write(f"Overall Price Missing Percentage: {missing_percent:.2f}%\n\n")
    
    f.write("--- MISSING PRICE BY YEAR ---\n")
    f.write(year_stats[['year', 'price', 'missing_percent']].to_string(index=False))
    f.write("\n\n")
    
    f.write("--- MISSING PRICE BY VEGETABLE TYPE ---\n")
    f.write(veg_stats[['vegetable_type', 'price', 'missing_percent']].to_string(index=False))
    f.write("\n\n")
    
    f.write("--- MISSING PRICE BY LOCATION ---\n")
    f.write(loc_stats[['location', 'price', 'missing_percent']].to_string(index=False))
    f.write("\n")

print(f"Price-only report saved to {output_report}")
print(f"Stage 3 Missing Price: {missing_percent:.2f}%")
