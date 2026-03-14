import pandas as pd
import os

# Define relative paths
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
file_path = os.path.join(project_root, "weekly_vegetable_all_location HARTI - weekly.csv")
output_report = os.path.join(base_dir, "initial_data_missing_report.txt")

print(f"Loading {file_path}...")
# Skip 3 lines as they are headers/notes
df = pd.read_csv(file_path, skiprows=3, header=None)

# Columns 0-4 are metadata, 5+ are weekly prices (52 weeks)
price_columns = df.iloc[:, 5:]

total_cells = price_columns.size
missing_cells = price_columns.isna().sum().sum()
missing_percent = (missing_cells / total_cells) * 100

# Metadata for context
num_rows = len(df)
num_veggies = df[4].nunique()
num_locations = df[2].nunique()
years = sorted(df[0].unique())

with open(output_report, 'w', encoding='utf-8') as f:
    f.write("INITIAL DATASET MISSING PRICE ANALYSIS\n")
    f.write("======================================\n\n")
    f.write(f"Source File: {os.path.basename(file_path)}\n")
    f.write(f"Total Metadata Rows: {num_rows}\n\n")
    
    f.write("--- MISSING PRICE METRICS ---\n")
    f.write(f"Total Price Cells (52 weeks per row): {total_cells:,}\n")
    f.write(f"Missing Price Cells: {missing_cells:,}\n")
    f.write(f"Overall Price Missing Percentage: {missing_percent:.2f}%\n\n")
    
    f.write("Note: This analysis focuses exclusively on the price columns (w1-w52).\n")

print(f"Price-only report saved to {output_report}")
print(f"Initial Dataset Missing Price: {missing_percent:.2f}%")
