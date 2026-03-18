import pandas as pd
import requests
import time
import os

# Define paths relative to script location
base_dir = os.path.dirname(os.path.abspath(__file__))

start_date = "2013-01-04"
end_date = "2019-12-27"
output_file = os.path.join(base_dir, "Daily Data", "Historical_Weather_Data_2013_2019_Daily.csv")

missing_locations = {
    "Badulla": (6.9847, 81.0568),
    "Kurunegala": (7.4855, 80.3647),
    "Matale": (7.4675, 80.6234),
    "Dambulla": (7.8596, 80.6543)
}

def get_historical_weather(lat, lon, location_name):
    print(f"Fetching Weather for {location_name}...")
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
        data = response.json().get("daily")
        if data:
            df = pd.DataFrame(data)
            df["location"] = location_name
            return df
    return None

all_weather_data = []

for loc, (lat, lon) in missing_locations.items():
    weather_df = get_historical_weather(lat, lon, loc)
    if weather_df is not None:
        all_weather_data.append(weather_df)
    time.sleep(1)

if all_weather_data:
    new_df = pd.concat(all_weather_data, ignore_index=True)
    new_df = new_df.rename(columns={
        "apparent_temperature_mean": "apparent_temperature_mean (°C)",
        "rain_sum": "rain_sum (mm)"
    })
    
    # Check if the columns match the exact format of existing daily file
    existing_df = pd.read_csv(output_file)
    
    # Keep only the matching columns that the existing file has (e.g. time, location, rain_sum (mm), apparent_temperature_mean (°C))
    for col in existing_df.columns:
        if col not in new_df.columns:
            new_df[col] = None
    
    new_df = new_df[existing_df.columns]
    
    final_df = pd.concat([existing_df, new_df], ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print("Missing data correctly appended to Daily dataset.")
else:
    print("Could not fetch missing data.")
