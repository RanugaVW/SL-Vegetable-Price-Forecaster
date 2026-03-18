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

low_country = [
    "Akuressa", "Ambalantota", "Ampara", "Anamaduwa", "Anuradhapura", "Balangoda", 
    "Beruwala", "Chilaw", "Embilipitiya", "Gampaha", "Hambanthota", "Horana", 
    "Ibbagamuwa", "Kalutara", "Kamburupitiya", "Kebithigollewa", "Kekirawa", 
    "Kuliyapitiya", "Mahiyanganaya", "Matara", "Matugama", "Mawathagama", 
    "Medawachchiya", "Mihintale", "Monaragala", "Mulatiyana", "Nikaweratiya", 
    "Panadura", "Pannala", "Polonnaruwa", "Puttalam", "Ratnapura", "Sooriyawewa", 
    "Tangalle", "Tissamaharama", "Vavuniya", "Weligama"
]

up_country = [
    "Ambewela", "Bandarawela", "Ella", "Haldummulla", "Haputale", "Hatton", 
    "Kandapola", "Kandy", "Nuwara Eliya", "Pattipola", "Ragala", "Welimada", "Wellawaya"
]

all_locations = low_country + up_country

# 1. Function to Get Coordinates
def get_coordinates(location_name):
    print(f"Geocoding {location_name}...")
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_name}&count=5&language=en&format=json"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            # Try to find exactly in Sri Lanka
            for res in data["results"]:
                if res.get("country_code") == "LK" or res.get("country") == "Sri Lanka":
                    return res["latitude"], res["longitude"]
            
            # Fallback to the first result if the country is missing but matching name
            return data["results"][0]["latitude"], data["results"][0]["longitude"]
    print(f"Failed to find coords for {location_name}")
    return None, None

# 2. Function to Get Historical Weather
def get_historical_weather(lat, lon, location_name):
    print(f"Fetching Weather for {location_name}...")
    # Open-Meteo Historical Archive API
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

# Map locations to coordinates
for loc in all_locations:
    loc = loc.strip()
    lat, lon = get_coordinates(loc)
    if lat and lon:
        weather_df = get_historical_weather(lat, lon, loc)
        if weather_df is not None:
            all_weather_data.append(weather_df)
    
    # Be polite to the free API
    time.sleep(1)

# Combine and Format
if all_weather_data:
    final_df = pd.concat(all_weather_data, ignore_index=True)
    
    # Rename columns to match requested format exactly
    final_df = final_df.rename(columns={
        "apparent_temperature_mean": "apparent_temperature_mean (°C)",
        "rain_sum": "rain_sum (mm)"
    })
    
    # Order the columns appropriately
    final_df = final_df[["time", "location", "rain_sum (mm)", "apparent_temperature_mean (°C)"]]
    
    # Save the file
    final_df.to_csv(output_file, index=False)
    print(f"\nSuccess! Saved to {output_file}")
    print(final_df.head(10))
else:
    print("No data was collected.")
