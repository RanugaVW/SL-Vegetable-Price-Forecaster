import pandas as pd
import os

def merge_and_check():
    # Paths
    s4_path = r'c:\Users\Ranuga\Data Science Project\Data Pre Processing\Stage 4\Combined_data - Main - Combined_data.csv'
    s3_path = r'c:\Users\Ranuga\Data Science Project\Data Pre Processing\Stage 3\Final_Merged_Dataset.csv'
    holiday_path = r'c:\Users\Ranuga\Data Science Project\Data Pre Processing\Stage 4\Combined_data - Main Heshan - Heshan_holiday_update.csv'
    
    print(f"Loading Stage 4 data from: {s4_path}...")
    df4 = pd.read_csv(s4_path)
    
    print(f"Loading Stage 3 weather data from: {s3_path}...")
    df3 = pd.read_csv(s3_path)

    print(f"Loading Holiday update data from: {holiday_path}...")
    df_h = pd.read_csv(holiday_path)
    
    # Key columns for merging
    merge_keys = ['Year_Week', 'location', 'vegetable_type']
    
    # Pre-clean Stage 4: Remove columns we are about to re-merge if they are empty
    # Also handle the vegetable_zone vs zone mismatch
    targets_to_remerge = ['mean_apparent_temperature', 'rain_sum', 'no_of_holidays', 'vegetable_zone', 'zone']
    for col in targets_to_remerge:
        if col in df4.columns:
            if df4[col].isna().all():
                print(f"Dropping empty column: {col}")
                df4 = df4.drop(columns=[col])
            else:
                print(f"Note: Column {col} has data, will merge and Resolve later.")

    # 1. Merge Weather Data from Stage 3
    weather_cols_s3 = ['origin_rain_sum_mean_mm', 'origin_temperature_mean_c']
    df3_weather = df3[merge_keys + weather_cols_s3].copy()
    df3_weather = df3_weather.rename(columns={
        'origin_rain_sum_mean_mm': 'rain_sum',
        'origin_temperature_mean_c': 'mean_apparent_temperature'
    })
    df3_weather = df3_weather.drop_duplicates(subset=merge_keys)

    df_combined = pd.merge(df4, df3_weather, on=merge_keys, how='left', suffixes=('', '_s3'))
    print(f"Merged weather data. Row count: {len(df_combined)}")

    # 2. Merge Holiday and Zone Data from Heshan's update
    h_cols = ['no_of_holidays', 'zone']
    df_h_subset = df_h[merge_keys + h_cols].copy()
    df_h_subset = df_h_subset.drop_duplicates(subset=merge_keys)

    df_final = pd.merge(df_combined, df_h_subset, on=merge_keys, how='left', suffixes=('', '_h'))
    print(f"Merged holiday and zone data. Row count: {len(df_final)}")

    # Rename 'zone' to 'vegetable_zone' if it exists and 'vegetable_zone' is gone
    if 'zone' in df_final.columns and 'vegetable_zone' not in df_final.columns:
        df_final = df_final.rename(columns={'zone': 'vegetable_zone'})
    elif 'zone' in df_final.columns and 'vegetable_zone' in df_final.columns:
        # If both exist, fillna vegetable_zone with zone
        df_final['vegetable_zone'] = df_final['vegetable_zone'].fillna(df_final['zone'])
        df_final = df_final.drop(columns=['zone'])

    # Final Check for missing values excluding price and fully empty columns
    print("\n--- Missing Values Report ---")
    missing_report = []
    fully_empty_cols = []
    
    for col in df_final.columns:
        total_na = df_final[col].isna().sum()
        if total_na == len(df_final):
            fully_empty_cols.append(col)
        elif total_na > 0:
            if col != 'price':
                missing_report.append(f"- {col}: {total_na} ({total_na/len(df_final)*100:.2f}%)")
    
    if missing_report:
        print("Columns with missing values (excluding price and fully empty):")
        for line in missing_report:
            print(line)
    else:
        print("No columns (other than price and fully empty ones) have missing values.")
        
    print(f"\nFully empty columns (ignored): {fully_empty_cols}")
    
    # Save the result
    df_final.to_csv(s4_path, index=False)
    print(f"\nSuccessfully saved final combined dataset to: {s4_path}")

if __name__ == "__main__":
    merge_and_check()
