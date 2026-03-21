import pandas as pd

df = pd.read_csv(r'C:\Users\Ranuga\Data Science Project\Final Complete Dataset\Model Building\2024_Predictions_vs_Actuals.csv')
df = df.sort_values(by=['retail_market', 'vegetable_type', 'year', 'week_num']).reset_index(drop=True)

df['prev_actual'] = df.groupby(['retail_market', 'vegetable_type'])['retail_price'].shift(1)
df['actual_diff'] = df['retail_price'] - df['prev_actual']

# Find massive actual spikes (price went up by >= 100)
spikes = df[df['actual_diff'] >= 100].copy()

if not spikes.empty:
    # Filter for reasonable error, then sort by smallest error percentage to find perfectly predicted spikes
    best_spike = spikes.sort_values(by='Error_Percentage').iloc[0]
    
    market = best_spike['retail_market']
    veg = best_spike['vegetable_type']
    target_idx = best_spike.name
    
    group_df = df[(df['retail_market'] == market) & (df['vegetable_type'] == veg)]
    idx_loc = group_df.index.get_loc(target_idx)
    
    start_loc = max(0, idx_loc - 2)
    end_loc = min(len(group_df), idx_loc + 3)
    
    subset = group_df.iloc[start_loc:end_loc]
    
    print(f"### Best Predicted Spike: {veg} in {market}")
    print("| Year | Week | Market | Vegetable | Actual Price | Actual Diff vs Prev Week | Predicted Price | Error % |")
    print("|---|---|---|---|---|---|---|---|")
    for _, row in subset.iterrows():
        diff_val = "N/A" if pd.isna(row['actual_diff']) else f"+{row['actual_diff']:.2f}" if row['actual_diff'] > 0 else f"{row['actual_diff']:.2f}"
        print(f"| {row['year']} | {row['week_num']} | {row['retail_market']} | {row['vegetable_type']} | {row['retail_price']:.2f} | {diff_val} | {row['Ensemble_Price_Predict']:.2f} | {row['Error_Percentage']:.2f}% |")
else:
    print('No spikes >= 100 found.')
