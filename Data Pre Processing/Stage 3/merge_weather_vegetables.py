import pandas as pd
import numpy as np
import os

# Base directory relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(base_dir))

vegetables_df_path = os.path.join(project_root, "Data Pre Processing", "Stage 2", "Combined_data.csv")
weather_df_path = os.path.join(project_root, "Weather Data", "Weekly Data", "Weekly_Weather_Data_2013_2019.csv")
output_df_path = os.path.join(base_dir, "Final_Merged_Dataset.csv")

# 1. Classify Vegetables
vegetable_zones = {
    "UP": ["BEETROOT", "GREEN BEANS", "LEEKS", "TOMATOES", "CABBAGE", "CARROT"],
    "LOW": ["LADIES FINGERS", "PUMPKIN", "GREEN CHILLIES", "BRINJALS", "ASH PLANTAINS", "SNAKE GOURD"]
}

def get_veg_zone(veg_name):
    if veg_name in vegetable_zones["UP"]: return "UP"
    if veg_name in vegetable_zones["LOW"]: return "LOW"
    return "UNKNOWN"

# 2. Market Mapping Dictionary (Destination -> Origins)
market_mapping = {    
    "Badulla": {"UP": ["Nuwara Eliya", "Welimada", "Bandarawela", "Haputale", "Ella", "Haldummulla"], "LOW": ["Ampara", "Monaragala", "Mahiyanganaya"]},
    "Puttalam": {"UP": ["Kurunegala", "Dambulla", "Matale"], "LOW": ["Puttalam", "Kurunegala", "Chilaw", "Anamaduwa", "Nikaweratiya"]},
    "Thambuththegama": {"UP": ["Matale", "Dambulla", "Kandy"], "LOW": ["Anuradhapura", "Polonnaruwa", "Kekirawa", "Medawachchiya"]},
    "Hambanthota": {"UP": ["Badulla", "Welimada"], "LOW": ["Hambanthota", "Tissamaharama", "Sooriyawewa", "Tangalle", "Ambalantota"]},
    "Anuradhapura": {"UP": ["Kandy", "Matale", "Dambulla"], "LOW": ["Anuradhapura", "Kebithigollewa", "Medawachchiya", "Vavuniya", "Mihintale"]},
    "Kurunegala": {"UP": ["Kandy", "Matale"], "LOW": ["Kurunegala", "Pannala", "Kuliyapitiya", "Mawathagama", "Ibbagamuwa"]},
    "Kaluthara": {"UP": ["Nuwara Eliya", "Badulla"], "LOW": ["Kalutara", "Beruwala", "Matugama", "Horana", "Panadura"]},
    "Mathara": {"UP": ["Badulla", "Haputale"], "LOW": ["Matara", "Weligama", "Akuressa", "Mulatiyana", "Kamburupitiya"]},
    "Meegoda": {"UP": ["Nuwara Eliya", "Badulla", "Kandy", "Matale"], "LOW": ["Kurunegala", "Anuradhapura", "Gampaha", "Kalutara", "Hambanthota"]},
    "Dambulla": {"UP": ["Kandy", "Matale", "Badulla", "Nuwara Eliya"], "LOW": ["Anuradhapura", "Polonnaruwa", "Kurunegala", "Kekirawa"]},
    "Kandy": {"UP": ["Nuwara Eliya", "Matale", "Badulla", "Hatton"], "LOW": ["Matale", "Kurunegala", "Dambulla"]},
    "Nuwaraeliya": {"UP": ["Nuwara Eliya", "Ragala", "Kandapola", "Ambewela", "Pattipola"], "LOW": ["Ampara", "Badulla", "Monaragala"]},
    "Embilipitiya": {"UP": ["Badulla", "Haputale", "Wellawaya"], "LOW": ["Hambanthota", "Ratnapura", "Embilipitiya", "Balangoda"]},
    "Colombo": {"UP": ["Nuwara Eliya", "Badulla", "Kandy", "Hatton", "Matale"], "LOW": ["Kurunegala", "Anuradhapura", "Hambanthota", "Gampaha", "Kalutara", "Ampara"]}
}

print("Loading datasets...")
veg_df = pd.read_csv(vegetables_df_path)
weather_df = pd.read_csv(weather_df_path)

# Optimize weather lookups by indexing Year_Week
weather_df.set_index('Year_Week', inplace=True)

# Arrays to store the new columns
rain_means = []
temp_means = []

print("Merging Data based on Origin Mapping Matrix...")
total_rows = len(veg_df)

for idx, row in veg_df.iterrows():
    if idx % 10000 == 0:
        print(f"Processed {idx}/{total_rows} rows...")
        
    market = row['location']
    veg_type = row['vegetable_type']
    target_week = str(row['Year_Week']) # Ensure string 
    
    zone = get_veg_zone(veg_type)
    
    if zone == "UNKNOWN" or market not in market_mapping:
        rain_means.append(np.nan)
        temp_means.append(np.nan)
        continue
    
    # Identify which origins we need to check this week for the crop 
    origins = market_mapping[market][zone]
    
    # Extract weather for this week, filter by origins
    if target_week in weather_df.index:
        week_weather = weather_df.loc[[target_week]] # Get all locations for this week
        origin_weather = week_weather[week_weather['location'].isin(origins)]
        
        if not origin_weather.empty:
            rain_means.append(origin_weather['rain_sum_mm'].mean())
            temp_means.append(origin_weather['mean_apparent_temp_c'].mean())
        else:
            rain_means.append(np.nan)
            temp_means.append(np.nan)
    else:
        rain_means.append(np.nan)
        temp_means.append(np.nan)

# Attach results
veg_df['origin_rain_sum_mean_mm'] = rain_means
veg_df['origin_temperature_mean_c'] = temp_means

# Format to 2 decimal places
veg_df['origin_rain_sum_mean_mm'] = veg_df['origin_rain_sum_mean_mm'].round(2)
veg_df['origin_temperature_mean_c'] = veg_df['origin_temperature_mean_c'].round(2)

print("Saving final dataset...")
veg_df.to_csv(output_df_path, index=False)

missing_weather = veg_df['origin_rain_sum_mean_mm'].isna().sum()
print(f"Success! Saved to {output_df_path}")
print(f"Final Dataset Total Rows: {len(veg_df)}")
print(f"Total Rows Missing Weather Matches: {missing_weather}")
print(veg_df.head(10).to_string())
