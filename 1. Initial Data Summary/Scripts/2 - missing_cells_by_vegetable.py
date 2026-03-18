import pandas as pd
import os

# Define base directory relative to script
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

file_path = os.path.join(project_root, "weekly_vegetable_all_location HARTI - weekly.csv")

# Load the dataset, skipping the first 3 rows which contain header/notes
df = pd.read_csv(file_path, skiprows=3, header=None)

# The columns: 0: Year, 1: LocationID, 2: Location, 3: ItemID, 4: Item
# The rest are weekly prices.
# Count the number of missing values in each row
df['MissingCount'] = df.isnull().sum(axis=1)

# Group by Item (Vegetable Type) which is column 4, then sum the missing counts
result = df.groupby(4)['MissingCount'].sum().reset_index()
result.columns = ['Vegetable_Type', 'Missing_Cells']

# Sort the summary descending
result = result.sort_values(by='Missing_Cells', ascending=False)

# Save to a CSV file
output_path = os.path.join(base_dir, "missing_cells_by_vegetable.csv")
result.to_csv(output_path, index=False)

print(f"Summary successfully saved to {output_path}\n")
print(result.to_string(index=False))
