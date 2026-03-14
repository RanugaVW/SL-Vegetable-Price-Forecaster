import pandas as pd
import os

# Define base directory relative to script
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

file_path = os.path.join(project_root, "weekly_vegetable_all_location HARTI - weekly.csv")

# Load the dataset, skipping the first 3 rows which contain header/notes
df = pd.read_csv(file_path, skiprows=3, header=None)

# The columns: 
# 0: Year, 1: LocationID, 2: Location, 3: ItemID, 4: Item (Vegetable Type)
# The weekly prices start at column 5
weekly_prices = df.iloc[:, 5:]

# Count the number of missing and available values in the price columns
df['MissingCount'] = weekly_prices.isnull().sum(axis=1)

# Group by Year (0), Location (2), and Vegetable Type (4)
result = df.groupby([0, 2, 4])['MissingCount'].sum().reset_index()
result.columns = ['Year', 'Location', 'Vegetable_Type', 'Missing_Cells']

# Sort the summary
result = result.sort_values(by=['Year', 'Location', 'Vegetable_Type'])

# Save to a CSV file
output_path = os.path.join(base_dir, "missing_cells_complete_summary.csv")
result.to_csv(output_path, index=False)

print(f"Summary successfully saved to {output_path}")
