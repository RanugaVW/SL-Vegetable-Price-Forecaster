import pandas as pd
import os

# Base directory relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(base_dir))

# Load the comprehensive missing cells summary
file_path = os.path.join(project_root, "Initial Dataset Summary", "missing_cells_complete_summary.csv")
df = pd.read_csv(file_path)

# Filter out the "Items" row which was a metadata row
df = df[df['Vegetable_Type'] != 'Items']

# Add Total Cells (Each row in this summary represents 1 year for 1 location for 1 veggie = 52 weeks)
df['Total_Cells'] = 52

output_path = os.path.join(base_dir, "removal_analysis.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("=== OVERALL METRICS ===\n")
    total_possible = df['Total_Cells'].sum()
    total_missing = df['Missing_Cells'].sum()
    f.write(f"Overall Missing Percentage: {(total_missing / total_possible * 100):.2f}%\n\n")

    f.write("=== MISSING DATA PERCENTAGE BY YEAR ===\n")
    year_summary = df.groupby('Year')[['Missing_Cells', 'Total_Cells']].sum()
    year_summary['Missing_Percentage'] = (year_summary['Missing_Cells'] / year_summary['Total_Cells'] * 100).round(2)
    year_summary = year_summary.sort_values(by='Missing_Percentage', ascending=False)
    f.write(year_summary.to_string())
    f.write("\n\n")
    
    f.write("=== MISSING DATA PERCENTAGE BY VEGETABLE TYPE ===\n")
    veg_summary = df.groupby('Vegetable_Type')[['Missing_Cells', 'Total_Cells']].sum()
    veg_summary['Missing_Percentage'] = (veg_summary['Missing_Cells'] / veg_summary['Total_Cells'] * 100).round(2)
    veg_summary = veg_summary.sort_values(by='Missing_Percentage', ascending=False)
    f.write(veg_summary.to_string())
    f.write("\n\n")

    f.write("=== MISSING DATA PERCENTAGE BY LOCATION ===\n")
    loc_summary = df.groupby('Location')[['Missing_Cells', 'Total_Cells']].sum()
    loc_summary['Missing_Percentage'] = (loc_summary['Missing_Cells'] / loc_summary['Total_Cells'] * 100).round(2)
    loc_summary = loc_summary.sort_values(by='Missing_Percentage', ascending=False)
    f.write(loc_summary.to_string())
    f.write("\n")

print("Analysis successfully saved to removal_analysis.txt")
