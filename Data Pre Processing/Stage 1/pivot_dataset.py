import pandas as pd
import os

# Base directory relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(base_dir))

input_path = os.path.join(project_root, "Weekly_vegetable_all_location HARTI - Ranuga - weekly.csv")
output_path = os.path.join(base_dir, "Weekly_vegetable_long_format.csv")

print("Loading dataset...")
df = pd.read_csv(input_path)

# Identify the columns to keep as IDs
id_vars = ['year', 'location', 'code', 'vegetable_type']
value_vars = [f'w{i}' for i in range(1, 53)]

# Melt the wide format into a long format
print("Melting dataset...")
melted_df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='week', value_name='price')

# Create the specific ID format requested: Year-Week (e.g., 2008-w1)
melted_df['Year_Week'] = melted_df['year'].astype(str) + '-' + melted_df['week']

# Extract just the week number for proper sorting
melted_df['week_num'] = melted_df['week'].str.replace('w', '').astype(int)

# Reorder the columns for better readability
final_df = melted_df[['Year_Week', 'year', 'week', 'location', 'code', 'vegetable_type', 'price', 'week_num']]

# Sort the values logically
print("Sorting dataset...")
final_df = final_df.sort_values(by=['year', 'location', 'vegetable_type', 'week_num'])

# Drop the temporary sorting column
final_df = final_df.drop(columns=['week_num'])

# Save to Output CSV
print(f"Saving to {output_path}...")
final_df.to_csv(output_path, index=False)

print("\nDone! Here is a preview of the new dataset:")
print(final_df.head(10).to_string(index=False))
