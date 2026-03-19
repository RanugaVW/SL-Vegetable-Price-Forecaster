import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Create a separate directory for the analysis
out_dir = 'outputs/producer_vs_retail_analysis'
os.makedirs(out_dir, exist_ok=True)

print('Loading dataset...')
df = pd.read_csv('data/main_with_producer_prices.csv')

# Drop rows missing either price
df_valid = df.dropna(subset=['price', 'producer_price']).copy()

# Extract numeric week for sorting (w1 -> 1)
df_valid['week_num'] = df_valid['week'].str.extract(r'(\d+)').astype(int)

# Calculate Margin and Markup %
df_valid['margin'] = df_valid['price'] - df_valid['producer_price']
df_valid['markup_pct'] = (df_valid['margin'] / df_valid['producer_price']) * 100

# Let's average across all locations and years to see the general weekly seasonal trend per vegetable
weekly_avg = df_valid.groupby(['vegetable_type', 'week_num'])[['price', 'producer_price', 'margin']].mean().reset_index()

# Save the detailed summary data
weekly_avg.to_csv(f'{out_dir}/weekly_margins_all_vegetables.csv', index=False)
print(f'Saved weekly aggregated data to {out_dir}/weekly_margins_all_vegetables.csv')

# Generate plots for each vegetable
vegetables = weekly_avg['vegetable_type'].unique()

print(f'Generating plots for {len(vegetables)} vegetables...')
for veg in vegetables:
    v_data = weekly_avg[weekly_avg['vegetable_type'] == veg].sort_values('week_num')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Retail vs Producer Prices
    ax1.plot(v_data['week_num'], v_data['price'], label='Retail Price', color='#E74C3C', linewidth=2, marker='o', markersize=4)
    ax1.plot(v_data['week_num'], v_data['producer_price'], label='Producer Price', color='#2ECC71', linewidth=2, marker='o', markersize=4)
    
    ax1.set_xlabel('Week of the Year')
    ax1.set_ylabel('Price (Rs/kg)')
    ax1.set_title(f'{veg.title()} - Retail vs Producer Prices by Week', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Create a twin axis for the margin bar chart
    ax2 = ax1.twinx()
    ax2.bar(v_data['week_num'], v_data['margin'], alpha=0.3, color='#3498DB', label='Retailer Margin (Rs)')
    ax2.set_ylabel('Margin (Rs/kg)')
    
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{veg.replace(" ", "_")}_weekly_margin.png', dpi=150)
    plt.close()

# Also do a specific year breakdown for 2019 as an example
df_2019 = df_valid[df_valid['year'] == 2019]
if not df_2019.empty:
    weekly_avg_2019 = df_2019.groupby(['vegetable_type', 'week_num'])[['price', 'producer_price', 'margin']].mean().reset_index()
    weekly_avg_2019.to_csv(f'{out_dir}/weekly_margins_2019.csv', index=False)
    print(f'Saved 2019 specific data to {out_dir}/weekly_margins_2019.csv')

print('Analysis complete! Check the producer_vs_retail_analysis folder.')
