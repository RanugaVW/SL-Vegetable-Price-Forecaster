import pandas as pd
import os

def final_cleanup():
    s4_path = r'c:\Users\Ranuga\Data Science Project\Data Pre Processing\Stage 4\Combined_data - Main - Combined_data.csv'
    holiday_path = r'c:\Users\Ranuga\Data Science Project\Data Pre Processing\Stage 4\Combined_data - Main Heshan - Heshan_holiday_update.csv'
    
    print(f"Loading merged data from: {s4_path}...")
    df = pd.read_csv(s4_path)
    
    # We want to keep these core columns
    core_cols = ['Year_Week', 'year', 'week', 'location', 'code', 'vegetable_type', 'price', 
                 'lanka_auto_diesel_price', 'usd_exchange_rate']
    
    # We want ONE set of weather columns
    # We'll take them from a reliable source if possible, or just pick the cleanest one.
    # Looking at the previous info, 'rain_sum' and 'mean_apparent_temperature' should be the targets.
    
    final_weather_cols = ['rain_sum', 'mean_apparent_temperature']
    
    # Selection logic: prioritize the columns without suffixes if they have data.
    # In my last run, 'rain_sum' and 'mean_apparent_temperature' were merged last from Stage 3.
    
    # We also need holiday and zone
    # Heshan's update has 'no_of_holidays' and 'zone'
    print("Loading Heshan data again for clean mapping...")
    df_h = pd.read_csv(holiday_path)
    h_subset = df_h[['Year_Week', 'location', 'vegetable_type', 'no_of_holidays', 'zone']].drop_duplicates()
    
    # Re-build the dataframe from scratch to be absolutely sure about columns
    df_clean = df[core_cols].copy()
    
    # Re-merge holiday/zone correctly
    df_clean = pd.merge(df_clean, h_subset, on=['Year_Week', 'location', 'vegetable_type'], how='left')
    df_clean = df_clean.rename(columns={'zone': 'vegetable_zone'})
    
    # Re-merge weather from Stage 3 correctly
    s3_path = r'c:\Users\Ranuga\Data Science Project\Data Pre Processing\Stage 3\Final_Merged_Dataset.csv'
    df3 = pd.read_csv(s3_path)
    weather_subset = df3[['Year_Week', 'location', 'vegetable_type', 'origin_rain_sum_mean_mm', 'origin_temperature_mean_c']].drop_duplicates()
    weather_subset = weather_subset.rename(columns={
        'origin_rain_sum_mean_mm': 'rain_sum',
        'origin_temperature_mean_c': 'mean_apparent_temperature'
    })
    
    df_clean = pd.merge(df_clean, weather_subset, on=['Year_Week', 'location', 'vegetable_type'], how='left')
    
    # Re-add 'seasonality' (even if empty) to maintain structure
    if 'seasonality' in df.columns:
        df_clean['seasonality'] = df['seasonality']
    else:
        df_clean['seasonality'] = None

    # Final Column Order
    final_cols = ['Year_Week', 'year', 'week', 'location', 'code', 'vegetable_type', 'price', 
                  'no_of_holidays', 'vegetable_zone', 'seasonality', 'lanka_auto_diesel_price', 
                  'mean_apparent_temperature', 'rain_sum', 'usd_exchange_rate']
    
    # Ensure all columns exist
    for col in final_cols:
        if col not in df_clean.columns:
            df_clean[col] = None
            
    df_clean = df_clean[final_cols]
    
    print("\n--- Final Integrity Check ---")
    missing_report = []
    fully_empty_cols = []
    for col in df_clean.columns:
        total_na = df_clean[col].isna().sum()
        if total_na == len(df_clean):
            fully_empty_cols.append(col)
        elif total_na > 0 and col != 'price':
            missing_report.append(f"- {col}: {total_na} ({total_na/len(df_clean)*100:.2f}%)")
            
    if missing_report:
        print("Missing values (excluding price/empty):")
        for line in missing_report: print(line)
    else:
        print("Done! No missing values in non-empty columns.")
    
    print(f"Empty columns: {fully_empty_cols}")
    print(f"Total Rows: {len(df_clean)}")
    
    # Save
    df_clean.to_csv(s4_path, index=False)
    print(f"\nFinal cleaned dataset saved to: {s4_path}")

if __name__ == "__main__":
    final_cleanup()
