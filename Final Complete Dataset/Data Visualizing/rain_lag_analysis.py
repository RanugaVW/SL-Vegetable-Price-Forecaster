import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

FILE     = r'C:\Users\Ranuga\Data Science Project\Final Complete Dataset\Data Pre Process\Combining DataSets\Final_Combined_data.csv'
SAVE_DIR = r'C:\Users\Ranuga\Data Science Project\Final Complete Dataset\Data Visualizing'

df = pd.read_csv(FILE, na_values=['NaN'])
df['date'] = pd.to_datetime(
    df['year'].astype(str) + df['week'].str.replace('w', '-W') + '-1',
    format='%Y-W%W-%w'
)

# Aggregate weekly mean across all markets
weekly = (
    df.groupby(['date', 'vegetable_type'])[['rain_sum', 'retail_price', 'mean_farmer_price']]
    .mean().reset_index().sort_values('date')
)

vegetables = sorted(weekly['vegetable_type'].unique())
MAX_LAG = 12  # check up to 12 weeks

# ── Cross-correlation per vegetable ────────────────────────────────────────
BG       = '#0f0f1a'
PANEL_BG = '#16213e'
GRID_COL = '#2a2a4a'

fig = plt.figure(figsize=(20, 18), facecolor=BG)
fig.suptitle('Cross-Correlation: rain_sum vs retail_price  (lag 0 – 12 weeks)',
             color='white', fontsize=14, fontweight='bold', y=1.01)

best_lags = {}

for i, veg in enumerate(vegetables):
    sub = weekly[weekly['vegetable_type'] == veg].dropna(
        subset=['rain_sum', 'retail_price']
    ).sort_values('date').reset_index(drop=True)

    rain   = sub['rain_sum'].values
    price  = sub['retail_price'].values

    # Standardise
    rain_z  = (rain  - rain.mean())  / (rain.std()  + 1e-9)
    price_z = (price - price.mean()) / (price.std() + 1e-9)

    corrs = []
    for lag in range(MAX_LAG + 1):
        if lag == 0:
            c = np.corrcoef(rain_z, price_z)[0, 1]
        else:
            c = np.corrcoef(rain_z[:-lag], price_z[lag:])[0, 1]
        corrs.append(c)

    best_lag = int(np.argmax(np.abs(corrs)))
    best_lags[veg] = {'best_lag': best_lag, 'corr': corrs[best_lag]}

    ax = fig.add_subplot(4, 3, i + 1)
    ax.set_facecolor(PANEL_BG)
    colors = ['#ff6b6b' if abs(c) == max(abs(x) for x in corrs) else '#a8edea' for c in corrs]
    ax.bar(range(MAX_LAG + 1), corrs, color=colors, edgecolor=BG, linewidth=0.5)
    ax.axhline(0, color='#555577', lw=0.8)
    ax.axvline(best_lag, color='#f8b500', lw=1.5, linestyle='--', alpha=0.8,
               label=f'Best lag={best_lag}w (r={corrs[best_lag]:.2f})')
    ax.axvline(4, color='#88ff88', lw=1.0, linestyle=':', alpha=0.7, label='Lag 4')
    ax.set_title(veg, color='white', fontsize=9, fontweight='bold', pad=5)
    ax.set_xlabel('Lag (weeks)', color='#888', fontsize=7)
    ax.set_ylabel('Correlation', color='#888', fontsize=7)
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    ax.set_xticks(range(0, MAX_LAG + 1))
    ax.grid(color=GRID_COL, lw=0.4, linestyle='--')
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=6.5,
              loc='lower right', edgecolor='#444466')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

plt.tight_layout()
path = rf'{SAVE_DIR}\rain_sum_lag_analysis.png'
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────
print("\n── Rain Lag Analysis Summary ─────────────────────")
print(f"{'Vegetable':<22} {'Best Lag (weeks)':>18} {'Correlation':>13}")
print("─" * 55)
for veg, info in sorted(best_lags.items()):
    marker = " ◄ lag=4" if info['best_lag'] == 4 else ""
    print(f"{veg:<22} {info['best_lag']:>18}     {info['corr']:>+.3f}{marker}")

at_4 = sum(1 for v in best_lags.values() if v['best_lag'] == 4)
print(f"\n{at_4}/{len(vegetables)} vegetables have peak correlation at lag=4 weeks")
print(f"Chart saved → {path}")
