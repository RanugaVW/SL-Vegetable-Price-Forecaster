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

def generate_yearly_heatmaps():
    print("Loading data...")
    df = pd.read_csv('data/main_filled.csv')
    
    # Preprocessing
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['week_num'] = df['week'].apply(parse_week)
    df['is_missing'] = df['price'].isna()
    
    years = sorted(df['year'].dropna().unique())
    output_dir = 'outputs/yearly_heatmaps'
    
    for year in years:
        year_df = df[df['year'] == year]
        
        # Calculate missing rate by vegetable and week for THIS year (average across locations)
        veg_week_missing = year_df.groupby(['vegetable_type', 'week_num'])['is_missing'].mean().reset_index()
        
        plt.figure(figsize=(15, 8))
        pivot_cw = veg_week_missing.pivot(index='vegetable_type', columns='week_num', values='is_missing')
        
        # Use vmax=1.0 so color scale is consistent across all years
        sns.heatmap(pivot_cw, cmap='Reds', vmin=0, vmax=1.0, cbar_kws={'label': '% Missing across locations (0 to 1)'})
        
        plt.title(f'Heatmap of Missing Prices by Vegetable Type & Week - {int(year)}')
        plt.xlabel('Week of the Year (1-52)')
        plt.ylabel('Vegetable Type')
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, f'heatmap_{int(year)}.png')
        plt.savefig(out_path)
        plt.close() # Free up memory
        print(f"Saved {out_path}")

if __name__ == "__main__":
    generate_yearly_heatmaps()
