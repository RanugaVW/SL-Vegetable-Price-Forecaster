import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_week(w):
    try:
        return int(str(w).lower().replace('w', '').strip())
    except:
        return np.nan

def analyze_missingness():
    print("Loading data...")
    df = pd.read_csv('data/main_filled.csv')
    
    # Preprocessing
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['week_num'] = df['week'].apply(parse_week)
    df['is_missing'] = df['price'].isna()
    
    total_rows = len(df)
    total_missing = df['is_missing'].sum()
    print(f"Total Rows: {total_rows}")
    print(f"Total Missing Prices: {total_missing} ({total_missing/total_rows*100:.2f}%)")
    
    # 1. By Vegetable Type and Week (Seasonal Analysis)
    # Average missingness rate across all years and locations for a specific week and vegetable
    veg_week_missing = df.groupby(['vegetable_type', 'week_num'])['is_missing'].mean().reset_index()
    
    # Detect Seasonal Absence (Missing > 60% of the time combined across all years/locations for that week)
    # Detect Regular Missing (Missing > 0% but < 60% of the time)
    
    plt.figure(figsize=(15, 8))
    pivot_cw = veg_week_missing.pivot(index='vegetable_type', columns='week_num', values='is_missing')
    sns.heatmap(pivot_cw, cmap='Reds', cbar_kws={'label': '% Missing (0 to 1)'})
    plt.title('Heatmap of Missing Prices by Vegetable Type & Week (Averaged across all years & locations)')
    plt.xlabel('Week of the Year (1-52)')
    plt.ylabel('Vegetable Type')
    plt.tight_layout()
    plt.savefig('outputs/missingness_heatmap.png')
    print("\nSaved missingness_heatmap.png")
    
    # Define thresholds
    SEASONAL_THRESHOLD = 0.70 # If missing >=70% of the time for that week across the 7 years
    
    print("\n--- Identifying Seasonal Absence vs True Missing Data ---")
    
    seasonal_weeks = {}
    
    for veg in df['vegetable_type'].unique():
        sub = veg_week_missing[veg_week_missing['vegetable_type'] == veg]
        high_miss_weeks = sub[sub['is_missing'] >= SEASONAL_THRESHOLD]['week_num'].tolist()
        
        if high_miss_weeks:
            seasonal_weeks[veg] = high_miss_weeks
            
    for veg, weeks in seasonal_weeks.items():
        print(f"SEASONAL ABSENCE DETECTED -> {veg}: Highly likely absent in weeks {weeks}")
        
    print("\n--- Summary of Null Classifications ---")
    
    # Classify every missing row
    # Merge the 'mean missingness for that (veg, week)' back to the main df to classify
    df = df.merge(veg_week_missing.rename(columns={'is_missing': 'week_veg_missing_rate'}), 
                  on=['vegetable_type', 'week_num'], how='left')
    
    def classify_null(row):
        if not row['is_missing']:
            return 'Present'
        if row['week_veg_missing_rate'] >= SEASONAL_THRESHOLD:
            return 'Seasonal Absence'
        else:
            return 'True Missing Data'
            
    df['missing_classification'] = df.apply(classify_null, axis=1)
    
    class_counts = df['missing_classification'].value_counts()
    for k, v in class_counts.items():
        print(f"{k}: {v}")
        
    # Validation by Location
    loc_missing = df[df['is_missing']].groupby('location')['missing_classification'].value_counts().unstack().fillna(0)
    print("\n--- Missing Breakdown By Location ---")
    print(loc_missing)
    
    # Save a stacked bar chart for locations
    loc_missing.plot(kind='bar', stacked=True, figsize=(12, 6), color=['orange', 'darkred'])
    plt.title('Missing Data Types by Location')
    plt.ylabel('Number of Missing Rows')
    plt.tight_layout()
    plt.savefig('outputs/missingness_by_location.png')
    print("Saved missingness_by_location.png")
    
    # Validation by Year
    yr_missing = df[df['is_missing']].groupby('year')['missing_classification'].value_counts().unstack().fillna(0)
    print("\n--- Missing Breakdown By Year ---")
    print(yr_missing)

    yr_missing.plot(kind='bar', stacked=True, figsize=(10, 5), color=['orange', 'darkred'])
    plt.title('Missing Data Types by Year')
    plt.ylabel('Number of Missing Rows')
    plt.tight_layout()
    plt.savefig('outputs/missingness_by_year.png')
    print("Saved missingness_by_year.png")

    df.to_csv('data/main_analyzed.csv', index=False)
    print("Saved detailed classifications to main_analyzed.csv")

if __name__ == "__main__":
    analyze_missingness()
