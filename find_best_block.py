import pandas as pd

df = pd.read_csv(r'C:\Users\Ranuga\Data Science Project\Final Complete Dataset\Model Building\2024_Predictions_vs_Actuals.csv')
df = df.sort_values(by=['retail_market', 'vegetable_type', 'year', 'week_num']).reset_index(drop=True)

df['prev_actual'] = df.groupby(['retail_market', 'vegetable_type'])['retail_price'].shift(1)
df['actual_diff'] = df['retail_price'] - df['prev_actual']

spikes = df[df['actual_diff'] >= 100]

best_block = None
best_avg_err = float('inf')

for target_idx in spikes.index:
    market = df.loc[target_idx, 'retail_market']
    veg = df.loc[target_idx, 'vegetable_type']
    
    group_df = df[(df['retail_market'] == market) & (df['vegetable_type'] == veg)]
    idx_loc = group_df.index.get_loc(target_idx)
    
    start_loc = max(0, idx_loc - 2)
    end_loc = min(len(group_df), idx_loc + 3)
    
    subset = group_df.iloc[start_loc:end_loc]
    
    if len(subset) < 4:
        continue
        
    avg_error = subset['Error_Percentage'].mean()
    
    if avg_error < best_avg_err:
        best_avg_err = avg_error
        best_block = subset

if best_block is not None:
    res = best_block[['year', 'week_num', 'retail_market', 'vegetable_type', 'retail_price', 'actual_diff', 'Ensemble_Price_Predict', 'Error_Percentage']].copy()
    res['actual_diff'] = res['actual_diff'].apply(lambda x: f"+{x:.2f}" if x>0 else f"{x:.2f}")
    res['Error_Percentage'] = res['Error_Percentage'].apply(lambda x: f"{x:.2f}%")
    res['retail_price'] = res['retail_price'].apply(lambda x: f"{x:.2f}")
    res['Ensemble_Price_Predict'] = res['Ensemble_Price_Predict'].apply(lambda x: f"{x:.2f}")
    with open('best_block_res.md', 'w') as f:
        f.write(f"### Most Consistently Accurate Spike Block: {res.iloc[0]['vegetable_type']} in {res.iloc[0]['retail_market']}\n")
        f.write(res.to_markdown(index=False))
        f.write(f"\n\nAverage Error for this block: {best_avg_err:.2f}%")
