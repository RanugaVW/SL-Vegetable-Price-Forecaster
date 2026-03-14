import pandas as pd
import os

# Define paths relative to script location
base_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_dir, "Daily Data", "Historical_Weather_Data_2013_2019_Daily.csv")
output_file = os.path.join(base_dir, "Weekly Data", "Weekly_Weather_Data_2013_2019.csv")

# Load daily data
print(f"Loading {input_file}...")
df = pd.read_csv(input_file)
df['time'] = pd.to_datetime(df['time'])

all_weekly_data = []

# Process year by year to strictly enforce the exactly 52-weeks rule starting on Friday
for year in range(2013, 2020):
    year_data = df[df['time'].dt.year == year].copy()
    
    # 1. Find the first Friday of the year
    # weekday() returns 0 for Monday, 4 for Friday
    first_day_of_year = pd.Timestamp(f"{year}-01-01")
    days_to_add = (4 - first_day_of_year.weekday() + 7) % 7
    first_friday = first_day_of_year + pd.Timedelta(days=days_to_add)
    
    # We only care about up to 52 weeks (52 * 7 = 364 days)
    end_of_52_weeks = first_friday + pd.Timedelta(days=(52 * 7) - 1)
    
    # Filter the exact 364-day span for this year
    valid_year_data = year_data[(year_data['time'] >= first_friday) & (year_data['time'] <= end_of_52_weeks)].copy()
    
    # Calculate the week number mathematically (0-indexed then add 1)
    valid_year_data['week_diff_days'] = (valid_year_data['time'] - first_friday).dt.days
    valid_year_data['week_num'] = (valid_year_data['week_diff_days'] // 7) + 1
    
    # We construct the ID to look exactly like the vegetable dataset
    valid_year_data['Year_Week'] = str(year) + "-w" + valid_year_data['week_num'].astype(str)
    valid_year_data['year'] = year
    
    all_weekly_data.append(valid_year_data)

# Combine all the filtered & tagged data horizontally
combined_daily_valid = pd.concat(all_weekly_data, ignore_index=True)

print("Aggregating into weekly metrics...")
# Aggregate! Group by our custom generated week and location
weekly_df = combined_daily_valid.groupby(['Year_Week', 'year', 'week_num', 'location']).agg(
    rain_sum_mm=('rain_sum (mm)', 'sum'),
    mean_apparent_temp_c=('apparent_temperature_mean (°C)', 'mean')
).reset_index()

# Round to 2 decimal places
weekly_df['rain_sum_mm'] = weekly_df['rain_sum_mm'].round(2)
weekly_df['mean_apparent_temp_c'] = weekly_df['mean_apparent_temp_c'].round(2)

# Sort logically for readability and identical structures with the vegetable dataset
weekly_df = weekly_df.sort_values(by=['year', 'location', 'week_num'])

# Drop the helper week column (the vegetable dataset just uses 'Year_Week' and 'year')
weekly_df = weekly_df.drop(columns=['week_num'])

print(f"Final dataset has {len(weekly_df)} rows")
print("Preview:")
print(weekly_df.head(10))

print(f"Saving to {output_file}...")
weekly_df.to_csv(output_file, index=False)
print("Complete!")
