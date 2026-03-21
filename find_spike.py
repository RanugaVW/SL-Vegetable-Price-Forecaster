import pandas as pd

df = pd.read_csv(r'C:\Users\Ranuga\Data Science Project\Final Complete Dataset\Model Building\2024_Predictions_vs_Actuals.csv')
df = df.sort_values(by=['retail_market', 'vegetable_type', 'year', 'week_num']).reset_index(drop=True)
df['prev_pred'] = df.groupby(['retail_market', 'vegetable_type'])['Ensemble_Price_Predict'].shift(1)
df['pred_diff'] = df['Ensemble_Price_Predict'] - df['prev_pred']

spikes = df[df['pred_diff'] >= 100]

if not spikes.empty:
    first_spike = spikes.iloc[0]
    market = first_spike['retail_market']
    veg = first_spike['vegetable_type']
    target_idx = first_spike.name
    
    group_df = df[(df['retail_market'] == market) & (df['vegetable_type'] == veg)]
    idx_loc = group_df.index.get_loc(target_idx)
    
    start_loc = max(0, idx_loc - 2)
    end_loc = min(len(group_df), idx_loc + 3)
    
    subset = group_df.iloc[start_loc:end_loc]
    
    print(f"### Found Spike Scenario: {veg} in {market}")
    print("| Year | Week | Market | Vegetable | Actual Price | Predicted Price | Predicted diff vs previous week |")
    print("|---|---|---|---|---|---|---|")
    for _, row in subset.iterrows():
        diff_val = "N/A" if pd.isna(row['pred_diff']) else f"{row['pred_diff']:.2f}"
        print(f"| {row['year']} | {row['week_num']} | {row['retail_market']} | {row['vegetable_type']} | {row['retail_price']:.2f} | {row['Ensemble_Price_Predict']:.2f} | {diff_val} |")
else:
    print('No spikes >= 100 found.')
