import pandas as pd
import os

# Base directory relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(base_dir))

# Define the file paths
input_path = os.path.join(project_root, "Data Pre Processing", "Stage 1", "Weekly_vegetable_long_format.csv")
combined_output_path = os.path.join(base_dir, "Combined_data.csv")

# Define the allowed lists
allowed_vegetables = [
    "BEETROOT", "GREEN BEANS", "LEEKS", "TOMATOES", "CABBAGE", "CARROT", 
    "LADIES FINGERS", "PUMPKIN", "GREEN CHILLIES", "BRINJALS", 
    "ASH PLANTAINS", "SNAKE GOURD"
]

allowed_locations = [
    "Badulla", "Puttalam", "Thambuththegama", "Hambanthota", "Anuradhapura", 
    "Kurunegala", "Kaluthara", "Mathara", "Meegoda", "Dambulla", 
    "Kandy", "Nuwaraeliya", "Embilipitiya", "Colombo"
]

allowed_years = ['2013', '2014', '2015', '2016', '2017', '2018', '2019']

print("Loading original dataset...")
df = pd.read_csv(input_path, low_memory=False)

print(f"Original shape: {df.shape}")

# Filter Vegetable Types
df_filtered = df[df['vegetable_type'].isin(allowed_vegetables)].copy()
print(f"Shape after filtering vegetable types: {df_filtered.shape}")

# Filter Locations
df_filtered = df_filtered[df_filtered['location'].isin(allowed_locations)]
print(f"Shape after filtering locations: {df_filtered.shape}")

# Filter Years
# Ensure the year column is treated as string for the isin comparison
df_filtered['year'] = df_filtered['year'].astype(str)
df_combined = df_filtered[df_filtered['year'].isin(allowed_years)]
print(f"Combined Dataset Shape: {df_combined.shape}")

# Save to CSV
print("Saving Combined Data...")
df_combined.to_csv(combined_output_path, index=False)

print("Done successfully!")
