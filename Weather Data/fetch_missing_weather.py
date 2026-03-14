import pandas as pd
import requests
import time
import os

# Define paths relative to script location
base_dir = os.path.dirname(os.path.abspath(__file__))

# --- Configuration ---
start_date = "2013-01-04"
end_date = "2019-12-27"
output_file = os.path.join(base_dir, "Daily Data", "Historical_Weather_Data_2013_2019_Daily.csv")

# Missing locations manually mapped to their exact coordinates
missing_locations = {
    "Hambanthota": (6.1248, 81.1228),
    "Kebithigollewa": (8.6653, 80.8033),
    "Mahiyanganaya": (7.3207, 80.9859),
    "Sooriyawewa": (6.3204, 80.9995),
    "Ambewela": (6.8778, 80.8041)
}

# 2. Function to Get Historical Weather
def get_historical_weather(lat, lon, location_name):
    print(f"Fetching Weather for {location_name} at {lat}, {lon}...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["apparent_temperature_mean", "rain_sum"],
        "timezone": "Asia/Colombo"
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()["daily"]
        df = pd.DataFrame(data)
        df["location"] = location_name
        return df
    
    print(f"Failed to fetch weather for {location_name}: {response.text}")
    return None

# --- Main Script ---
all_weather_data = []

# Fetch for the missing ones
for loc, (lat, lon) in missing_locations.items():
    weather_df = get_historical_weather(lat, lon, loc)
    if weather_df is not None:
        all_weather_data.append(weather_df)
    time.sleep(1)

# Combine and Format
if all_weather_data:
    new_df = pd.concat(all_weather_data, ignore_index=True)
    
    # Rename columns to match requested format exactly
    new_df = new_df.rename(columns={
        "apparent_temperature_mean": "apparent_temperature_mean (°C)",
        "rain_sum": "rain_sum (mm)"
    })
    
    # Order the columns appropriately
    new_df = new_df[["time", "location", "rain_sum (mm)", "apparent_temperature_mean (°C)"]]
    
    # Append to existing file
    print("Appending to existing dataset...")
    existing_df = pd.read_csv(output_file)
    final_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    final_df.to_csv(output_file, index=False)
    print(f"\nSuccess! Missing locations added to {output_file}")
else:
    print("No data was collected.")
