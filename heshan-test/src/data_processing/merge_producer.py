import pandas as pd
import numpy as np

# 1. Load main.csv
print('Loading main.csv...')
df_main = pd.read_csv('data/main.csv')

# 2. Load producer prices
print('Loading producer prices...')
df_prod = pd.read_csv('data/weekly_producer_vegetable_all_location(2008-2024) - weekly.csv')

# 3. Melt producer prices to long format
weeks = [f'W{i}' for i in range(1, 53)]
# keep only the weeks that exist in columns
valid_weeks = [w for w in weeks if w in df_prod.columns]

df_prod_long = pd.melt(df_prod, 
                      id_vars=['Year', 'Loc Cod', 'Location', 'Item Cod', 'Items'], 
                      value_vars=valid_weeks,
                      var_name='week', 
                      value_name='producer_price')

# Basic cleaning
df_prod_long['week'] = df_prod_long['week'].str.lower() # 'W1' -> 'w1'
df_prod_long['Location'] = df_prod_long['Location'].astype(str).str.strip().str.title()
df_prod_long['Items'] = df_prod_long['Items'].astype(str).str.strip().str.upper()

# Handle spelling inconsistencies
veg_mapping = {
    'GREEN CILLIES': 'GREEN CHILLIES',
    'TOMATOE': 'TOMATOES'
}
df_prod_long['Items'] = df_prod_long['Items'].replace(veg_mapping)

location_mapping = {
    # Check if there are any location mappings needed? We can do a quick check
}

# Ensure main.csv keys match
df_main['location_clean'] = df_main['location'].astype(str).str.strip().str.title()
df_main['veg_clean'] = df_main['vegetable_type'].astype(str).str.strip().str.upper()

# 4. Try to merge
df_prod_long = df_prod_long.rename(columns={
    'Year': 'year',
    'Location': 'location_clean',
    'Items': 'veg_clean'
})

merged = pd.merge(df_main, df_prod_long[['year', 'week', 'location_clean', 'veg_clean', 'producer_price']], 
                 on=['year', 'week', 'location_clean', 'veg_clean'], 
                 how='left')

matched_prop = merged['producer_price'].notna().mean()
print(f'Matched producer prices: {merged["producer_price"].notna().sum()} out of {len(merged)} ({matched_prop:.2%})')

# Inspect mismatched items
if matched_prop < 0.6:
    print('Locations in main:', sorted(df_main['location_clean'].unique()))
    print('Locations in producer:', sorted(df_prod_long['location_clean'].unique()))

merged = merged.drop(columns=['location_clean', 'veg_clean'])
merged.to_csv('data/main_with_producer_prices.csv', index=False)
print('Saved to main_with_producer_prices.csv!')
