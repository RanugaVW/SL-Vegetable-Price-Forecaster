import pandas as pd
import os
import itertools

input_file = r'C:\Users\Ranuga\Data Science Project\Final Complete Dataset\weekly_producer_vegetable_all_location(2008-2024).xlsx'
out_dir = r'C:\Users\Ranuga\Data Science Project\Final Complete Dataset\Data Pre Process'
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, 'weekly_producer_vegetable_long_format.csv')

print('Reading excel...')
df = pd.read_excel(input_file, header=2)

print('Filtering valid rows...')
df = df.dropna(subset=['Year'])

print('Processing...')
value_vars = [c for c in df.columns if str(c).startswith('W') and str(c)[1:].isdigit()]
id_vars = ['Year', 'Loc Cod', 'Location', 'Item Cod', 'Items']

df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='week_orig', value_name='price')

df_melted['year'] = df_melted['Year'].astype(int).astype(str)
df_melted['week'] = df_melted['week_orig'].str.lower()
df_melted['location'] = df_melted['Location']
df_melted['vegetable_type'] = df_melted['Items'].astype(str).str.strip().str.upper()

# Fix typos in the dataset to match requested spelling
df_melted['vegetable_type'] = df_melted['vegetable_type'].str.replace('TOMATOE', 'TOMATOES').str.replace('GREEN CILLIES', 'GREEN CHILLIES')

df_melted['location_clean'] = df_melted['location'].astype(str).str.lower().str.replace(' ', '')
df_melted['price'] = pd.to_numeric(df_melted['price'], errors='coerce')


# DEFINE FILTERS
valid_years = [str(y) for y in range(2013, 2020)]
valid_vegetables = ["BEETROOT", "GREEN BEANS", "LEEKS", "TOMATOES", "CABBAGE", "CARROT", "LADIES FINGERS", "PUMPKIN", "GREEN CHILLIES", "BRINJALS", "ASH PLANTAINS", "SNAKE GOURD"]
valid_locations = ["Badulla", "Puttalam", "Thambuththegama", "Hambanthota", "Anuradhapura", "Kurunegala", "Kaluthara", "Mathara", "Meegoda", "Dambulla", "Kandy", "Nuwaraeliya", "Embilipitiya", "Colombo"]

# Generate Cartesian Product
weeks = [f"w{i}" for i in range(1, 53)]
year_weeks = []
for y in valid_years:
    for w in weeks:
        year_weeks.append((y, w, f"{y}-{w}"))

combinations = list(itertools.product(year_weeks, valid_locations, valid_vegetables))
df_full = pd.DataFrame([ 
    (yw[2], yw[0], yw[1], loc, veg) for yw, loc, veg in combinations 
], columns=['Year_Week', 'year', 'week', 'location', 'vegetable_type'])

df_full['location_clean'] = df_full['location'].str.lower().str.replace(' ', '')

# Merge 
df_to_merge = df_melted[['year', 'week', 'location_clean', 'vegetable_type', 'price']]
df_merged = pd.merge(df_full, df_to_merge, on=['year', 'week', 'location_clean', 'vegetable_type'], how='left')

# Sort
df_merged['week_num'] = df_merged['week'].str.replace('w','').astype(int)
df_merged = df_merged.sort_values(['location', 'vegetable_type', 'year', 'week_num'])

# Format Price correctly exactly 2 decimals
df_merged['price'] = df_merged['price'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")

# Output Col
out_cols = ['Year_Week', 'year', 'week', 'location', 'vegetable_type', 'price']
df_out = df_merged[out_cols]

print('Output shape:', df_out.shape)

print('Saving filtered data...')
df_out.to_csv(out_file, index=False)
print('Done!')
