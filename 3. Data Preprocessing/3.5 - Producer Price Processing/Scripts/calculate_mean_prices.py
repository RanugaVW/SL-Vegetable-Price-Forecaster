import pandas as pd
import numpy as np

location_file = r"C:\Users\Ranuga\Data Science Project\Final Complete Dataset\Data Pre Process\Missing Value Handling\Farmer Produce Locaation.csv"
producer_prices_file = r"C:\Users\Ranuga\Data Science Project\Final Complete Dataset\Data Pre Process\weekly_producer_vegetable_long_format.csv"
output_file = r"C:\Users\Ranuga\Data Science Project\Final Complete Dataset\Data Pre Process\Missing Value Handling\Calculated_Retail_Producer_Prices.csv"

# Load data
mappings_df = pd.read_csv(location_file)
prices_df = pd.read_csv(producer_prices_file)

# Clean string columns just in case
mappings_df.rename(columns={'Secondary Producer Market Target (Fallback)': 'Secondary_Target'}, inplace=True)
mappings_df['Vegetable'] = mappings_df['Vegetable'].astype(str).str.strip().str.upper()
mappings_df['Primary Producer Market Target'] = mappings_df['Primary Producer Market Target'].astype(str).str.strip()
mappings_df['Secondary_Target'] = mappings_df['Secondary_Target'].astype(str).str.strip()

prices_df['vegetable_type'] = prices_df['vegetable_type'].astype(str).str.strip().str.upper()
prices_df['location'] = prices_df['location'].astype(str).str.strip()

# Create a deduplicated subset of prices
prices_subset = prices_df[['Year_Week', 'vegetable_type', 'location', 'farmer_price']].drop_duplicates(subset=['Year_Week', 'vegetable_type', 'location'])

# Get a grid of all available weeks and vegetables
weeks_veg_df = prices_df[['Year_Week', 'year', 'week', 'vegetable_type']].drop_duplicates()

# Merge available weeks and vegetables with the mappings to get all combinations for each Retail Market
merged_base = pd.merge(weeks_veg_df, mappings_df, left_on='vegetable_type', right_on='Vegetable', how='inner')

# Join to get Primary Price
final_df = pd.merge(merged_base, prices_subset, 
                    left_on=['Year_Week', 'vegetable_type', 'Primary Producer Market Target'],
                    right_on=['Year_Week', 'vegetable_type', 'location'], 
                    how='left')
final_df.rename(columns={'farmer_price': 'primary_price'}, inplace=True)
final_df.drop(columns=['location'], inplace=True, errors='ignore')

# Join to get Secondary Price
final_df = pd.merge(final_df, prices_subset, 
                    left_on=['Year_Week', 'vegetable_type', 'Secondary_Target'],
                    right_on=['Year_Week', 'vegetable_type', 'location'], 
                    how='left')
final_df.rename(columns={'farmer_price': 'secondary_price'}, inplace=True)
final_df.drop(columns=['location'], inplace=True, errors='ignore')

# Calculate the mean price. mean(axis=1) automatically handles NaNs by using the non-NaN value, or NaN if both are NaN.
final_df['mean_farmer_price'] = final_df[['primary_price', 'secondary_price']].mean(axis=1)

# Format and organize final columns
cols_to_keep = ['Year_Week', 'year', 'week', 'Retail Market', 'Vegetable', 'Primary Producer Market Target', 'Secondary_Target', 'primary_price', 'secondary_price', 'mean_farmer_price']
final_df = final_df[cols_to_keep]

final_df.rename(columns={'Secondary_Target': 'Secondary Producer Market Target (Fallback)'}, inplace=True)

# Sort the dataset for better readability
final_df.sort_values(by=['Retail Market', 'Vegetable', 'year', 'week'], inplace=True)

# Save to output file
final_df.to_csv(output_file, index=False)
print(f"Generated {len(final_df)} rows.")
print(f"Saved dataset to {output_file}")
