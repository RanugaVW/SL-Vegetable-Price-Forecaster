import pandas as pd
import os

# Base directory relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(base_dir, "Final_Merged_Dataset.csv")
output_report = os.path.join(base_dir, "completeness_report.txt")

print(f"Loading {file_path}...")
df = pd.read_csv(file_path)

# 1. Basic Stats
total_rows = len(df)
locations = df['location'].unique()
vegetables = df['vegetable_type'].unique()
years = df['year'].unique()

print(f"Total rows: {total_rows}")
print(f"Locations ({len(locations)}): {sorted(locations)}")
print(f"Vegetables ({len(vegetables)}): {sorted(vegetables)}")
print(f"Years ({len(years)}): {sorted(years)}")

# 2. Detailed check
print("\nVerifying 52 weeks per year/location/vegetable group...")
# Group and count rows
group_counts = df.groupby(['year', 'location', 'vegetable_type']).size().reset_index(name='week_count')

# Identify problematic groups
missing_data = group_counts[group_counts['week_count'] != 52]

# Writing report
with open(output_report, 'w') as f:
    f.write("DATASET COMPLETENESS REPORT\n")
    f.write("===========================\n\n")
    f.write(f"File: {file_path}\n")
    f.write(f"Total Rows: {total_rows}\n")
    f.write(f"Expected Rows (7 years * 14 locations * 12 veggies * 52 weeks): {7*14*12*52}\n\n")
    
    f.write(f"Locations found ({len(locations)}):\n{', '.join(sorted(locations))}\n\n")
    f.write(f"Vegetables found ({len(vegetables)}):\n{', '.join(sorted(vegetables))}\n\n")
    f.write(f"Years found ({len(years)}):\n{', '.join(map(str, sorted(years)))}\n\n")
    
    if missing_data.empty:
        f.write("VERIFICATION SUCCESS: Every single combination of Year, Location, and Vegetable Type has exactly 52 weeks.\n")
    else:
        f.write("VERIFICATION WARNING: Some groups do not have 52 weeks!\n")
        f.write(missing_data.to_string())
        f.write("\n")

print(f"Report saved to {output_report}")
if missing_data.empty:
    print("Verification SUCCESS: No missing weeks found.")
else:
    print(f"Verification WARNING: {len(missing_data)} groups have missing weeks. See report for details.")
