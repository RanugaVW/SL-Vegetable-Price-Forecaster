import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

out_dir = 'market_specific_analysis'
os.makedirs(out_dir, exist_ok=True)

print('Loading dataset...')
df = pd.read_csv('data/main_with_producer_prices.csv')

# Extract numeric week
df['week_num'] = df['week'].str.extract(r'(\d+)').astype(int)

# Filter out extreme outliers for visualization
df.loc[df['price'] > 1500, 'price'] = np.nan

def analyze_market_behavior(location, vegetable):
    data = df[(df['location'] == location) & (df['vegetable_type'] == vegetable)].copy()
    
    if data.empty:
        print(f"Skipping {location} - {vegetable} (No data)")
        return
        
    print(f"Analyzing {vegetable} in {location}...")
    
    # Sort chronologically
    data = data.sort_values(['year', 'week_num'])
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'Market Analysis: {vegetable.title()} in {location.title()}', fontsize=16, fontweight='bold')
    
    # 1. Seasonality: Average Price by Week
    ax1 = plt.subplot(2, 2, 1)
    weekly_avg = data.groupby('week_num')['price'].mean().reset_index()
    sns.lineplot(data=weekly_avg, x='week_num', y='price', ax=ax1, color='red', marker='o')
    ax1.set_title('Average Seasonal Trend (Week 1 to 52)')
    ax1.set_xlabel('Week of Year')
    ax1.set_ylabel('Avg Retail Price (Rs/kg)')
    
    # 2. Year-over-Year Trend
    ax2 = plt.subplot(2, 2, 2)
    yearly_avg = data.groupby('year')['price'].mean().reset_index()
    sns.barplot(data=yearly_avg, x='year', y='price', ax=ax2, palette='viridis')
    ax2.set_title('Year-over-Year Average Price')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Price (Rs/kg)')
    
    # 3. Retail vs Producer Price Gap (if producer data exists for this market)
    ax3 = plt.subplot(2, 2, 3)
    if not data['producer_price'].isna().all():
        weekly_prod_avg = data.groupby('week_num')[['price', 'producer_price']].mean().reset_index()
        ax3.plot(weekly_prod_avg['week_num'], weekly_prod_avg['price'], label='Retail Price', color='red')
        ax3.plot(weekly_prod_avg['week_num'], weekly_prod_avg['producer_price'], label='Producer Price', color='green')
        ax3.fill_between(weekly_prod_avg['week_num'], weekly_prod_avg['producer_price'], weekly_prod_avg['price'], color='blue', alpha=0.1, label='Markup Margin')
        ax3.set_title('Retail vs Producer Price (Seasonal)')
        ax3.set_xlabel('Week of Year')
        ax3.set_ylabel('Price (Rs/kg)')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No Producer Price Data\nAvilable for this Market', 
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        ax3.set_title('Retail vs Producer Price')
        ax3.axis('off')
        
    # 4. Price Distribution (Volatility)
    ax4 = plt.subplot(2, 2, 4)
    sns.boxplot(data=data, x='year', y='price', ax=ax4, palette='Set2')
    ax4.set_title('Price Volatility per Year')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Price (Rs/kg)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    safe_loc = location.replace('/', '_').replace(' ', '_')
    safe_veg = vegetable.replace('/', '_').replace(' ', '_')
    plt.savefig(f'{out_dir}/{safe_loc}_{safe_veg}_analysis.png', dpi=150)
    plt.close()

# Let's run this for a few key markets and vegetables as examples
markets_to_test = ['Dambulla', 'Colombo', 'Kandy', 'Nuwaraeliya']
vegetables_to_test = ['CARROT', 'PUMPKIN', 'TOMATOES', 'GREEN CHILLIES']

for m in markets_to_test:
    for v in vegetables_to_test:
        analyze_market_behavior(m, v)

print(f'\nDone! Check the {out_dir} folder for the generated market analysis dashboards.')
