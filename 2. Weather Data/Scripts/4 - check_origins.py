import pandas as pd
import os

# Define paths relative to script location
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the weather dataset locations
weather_df = pd.read_csv(os.path.join(base_dir, "Weekly Data", "Weekly_Weather_Data_2013_2019.csv"))
available_locations = set(weather_df['location'].unique())

# Define the dictionary from the user's images
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

required_locations = set()
for market, regions in market_mapping.items():
    for origin in regions["UP"]: required_locations.add(origin)
    for origin in regions["LOW"]: required_locations.add(origin)

print(f"Total required distinct origin locations: {len(required_locations)}")
missing = required_locations - available_locations

print(f"\nMissing weather locations needed for Origins: {missing}")

