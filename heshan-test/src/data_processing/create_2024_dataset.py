import pandas as pd
import numpy as np

# Load main target format for reference
df_main_full = pd.read_csv('data/main.csv')
valid_pairs = df_main_full[['vegetable_type', 'location']].drop_duplicates().copy()
valid_pairs['vegetable_type'] = valid_pairs['vegetable_type'].str.strip().str.upper()
valid_pairs['location'] = valid_pairs['location'].str.strip().str.upper()

df_main = pd.read_csv('data/main_with_producer_prices.csv', nrows=1)
columns_order = df_main.columns.tolist()

# Load downloaded file
raw_csv_path = r'c:\Users\hesha\Downloads\weekly_vegetable_all_location HARTI - weekly.csv'

# Since we know the structure, let's process it carefully
df_raw = pd.read_csv(raw_csv_path, skiprows=1, header=None, low_memory=False)

# Drop any entirely empty rows and noise
df_raw = df_raw.dropna(how='all')

# Let's filter for 2024
df_2024 = df_raw[df_raw[0].astype(str).str.strip() == '2024'].copy()

# Rename the columns
col_names = ['year', 'loc_code', 'location', 'code', 'vegetable_type'] + [f'w{i}' for i in range(1, 53)]
# Only take the first 57 columns
df_2024 = df_2024.iloc[:, :57]
df_2024.columns = col_names

# Melt the dataframe
df_melted = df_2024.melt(id_vars=['year', 'loc_code', 'location', 'code', 'vegetable_type'], 
                         value_vars=[f'w{i}' for i in range(1, 53)], 
                         var_name='week', 
                         value_name='price')

# Clean up Year_Week
df_melted['Year_Week'] = df_melted['year'].astype(str) + '-' + df_melted['week']

# Fill place holders
df_melted['no_of_holidays'] = np.nan
df_melted['vegetable_zone'] = np.nan
df_melted['seasonality'] = np.nan
df_melted['lanka_auto_diesel_price'] = np.nan
df_melted['mean_apparent_temperature'] = np.nan
df_melted['rain_sum'] = np.nan
df_melted['usd_exchange_rate'] = np.nan
df_melted['producer_price'] = np.nan

# Convert types or just leave as is since they are NaNs
df_melted['price'] = pd.to_numeric(df_melted['price'], errors='coerce')

# Strip and uppercase for matching
df_melted['veg_match'] = df_melted['vegetable_type'].str.strip().str.upper()
df_melted['loc_match'] = df_melted['location'].str.strip().str.upper()

# Merge to filter only the chosen vegetables and locations
df_filtered = df_melted.merge(valid_pairs, left_on=['veg_match', 'loc_match'], right_on=['vegetable_type', 'location'], how='inner', suffixes=('', '_drop'))

# Assign correctly matched names from the valid_pairs to avoid lowercase/uppercase discrepancies later
df_filtered['vegetable_type'] = df_filtered['vegetable_type_drop']
df_filtered['location'] = df_filtered['location_drop']

# Drop rows where price is NaN if desired (optional, let's keep all to match existing behavior unless asked)
# We can just output the matching columns
df_out = df_filtered[columns_order]

# Ensure column types similar to main dataset if possible
df_out['year'] = 2024
df_out['code'] = pd.to_numeric(df_out['code'], errors='coerce')

df_out.to_csv('data/2024_dataset.csv', index=False)
print("Saved 2024 dataset to `2024_dataset.csv` with shape:", df_out.shape)
