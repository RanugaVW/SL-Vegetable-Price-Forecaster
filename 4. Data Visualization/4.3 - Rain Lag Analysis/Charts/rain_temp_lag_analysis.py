"""
Complete Rain & Temperature Lag Analysis
─────────────────────────────────────────
Generates 6 comprehensive charts analyzing how lagged rain and temperature
affect retail_price and farmer_price for each vegetable type across all years.

Charts produced:
  1. Rain → Retail Price    (cross-correlation per vegetable, all years combined)
  2. Rain → Farmer Price    (cross-correlation per vegetable, all years combined)
  3. Temperature → Retail Price   (cross-correlation per vegetable, all years combined)
  4. Temperature → Farmer Price   (cross-correlation per vegetable, all years combined)
  5. Best Lag Heatmap — Rain  (vegetable × year)  for retail & farmer price
  6. Best Lag Heatmap — Temp  (vegetable × year)  for retail & farmer price
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings, os
warnings.filterwarnings('ignore')

# ─── Config ──────────────────────────────────────────────────────────────────
FILE     = r'C:\Users\Ranuga\Data Science Project\3. Data Preprocessing\3.7 - Combining Datasets\Outputs\Final_Combined_data.csv'
SAVE_DIR = r'C:\Users\Ranuga\Data Science Project\4. Data Visualization\4.3 - Rain Lag Analysis\Charts'
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_LAG = 12   # 0 to 12 weeks

# ─── Styling ─────────────────────────────────────────────────────────────────
BG       = '#0d1117'
PANEL_BG = '#161b22'
GRID_COL = '#21262d'
TEXT_COL = '#c9d1d9'
ACCENT_1 = '#58a6ff'    # blue highlight
ACCENT_2 = '#f78166'    # coral highlight
ACCENT_3 = '#3fb950'    # green
ACCENT_4 = '#d2a8ff'    # purple
BEST_COL = '#ffa657'    # orange for best lag line

plt.rcParams.update({
    'font.family': 'Segoe UI',
    'axes.facecolor': PANEL_BG,
    'figure.facecolor': BG,
    'text.color': TEXT_COL,
    'axes.labelcolor': TEXT_COL,
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
})

# ─── Load Data ───────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(FILE, na_values=['NaN'])
df['week_num'] = pd.to_numeric(df['week'].str.extract(r'(\d+)')[0])

# Create a proper date column for time-ordering
df['date'] = pd.to_datetime(
    df['year'].astype(str) + df['week'].str.replace('w', '-W') + '-1',
    format='%Y-W%W-%w'
)

vegetables = sorted(df['vegetable_type'].unique())
years = sorted(df['year'].unique())
print(f"  {len(vegetables)} vegetables × {len(years)} years = {len(vegetables)*len(years)} groups")
print(f"  Rows: {len(df):,}")


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY: Compute cross-correlation for a given weather col vs price col
# ═════════════════════════════════════════════════════════════════════════════
def compute_lag_correlations(sub_df, weather_col, price_col, max_lag=MAX_LAG):
    """Return list of correlations for lags 0..max_lag."""
    sub = sub_df.dropna(subset=[weather_col, price_col]).sort_values('date').reset_index(drop=True)
    if len(sub) < max_lag + 5:
        return [np.nan] * (max_lag + 1)

    w = sub[weather_col].values
    p = sub[price_col].values

    # Standardise
    w_z = (w - w.mean()) / (w.std() + 1e-9)
    p_z = (p - p.mean()) / (p.std() + 1e-9)

    corrs = []
    for lag in range(max_lag + 1):
        if lag == 0:
            c = np.corrcoef(w_z, p_z)[0, 1]
        else:
            c = np.corrcoef(w_z[:-lag], p_z[lag:])[0, 1]
        corrs.append(c if np.isfinite(c) else 0.0)
    return corrs


# ═════════════════════════════════════════════════════════════════════════════
# CHART FUNCTION: 4×3 cross-correlation bar charts (one per vegetable)
# ═════════════════════════════════════════════════════════════════════════════
def plot_cross_correlation_grid(df, weather_col, price_col, title, accent_bar, accent_best, filename):
    """Create a 4×3 grid of cross-correlation bar charts."""
    # Aggregate weekly mean across all markets
    weekly = (
        df.groupby(['date', 'vegetable_type'])[[weather_col, price_col]]
        .mean().reset_index().sort_values('date')
    )

    fig = plt.figure(figsize=(22, 20), facecolor=BG)
    fig.suptitle(title, color='white', fontsize=16, fontweight='bold', y=0.98)

    best_lags = {}

    for i, veg in enumerate(vegetables):
        sub = weekly[weekly['vegetable_type'] == veg]
        corrs = compute_lag_correlations(sub, weather_col, price_col)

        best_lag = int(np.nanargmax(np.abs(corrs)))
        best_lags[veg] = {'best_lag': best_lag, 'corr': corrs[best_lag]}

        ax = fig.add_subplot(4, 3, i + 1)

        # Color: highlight the best lag
        colors = []
        for j, c in enumerate(corrs):
            if j == best_lag:
                colors.append(ACCENT_2)
            elif c >= 0:
                colors.append(accent_bar)
            else:
                colors.append('#f8514966')

        ax.bar(range(MAX_LAG + 1), corrs, color=colors, edgecolor=BG, linewidth=0.5, width=0.75)
        ax.axhline(0, color='#30363d', lw=0.8)
        ax.axvline(best_lag, color=BEST_COL, lw=1.8, linestyle='--', alpha=0.85,
                   label=f'Best lag = {best_lag}w  (r = {corrs[best_lag]:+.3f})')

        # Mark lag 4 if it's not the best
        if best_lag != 4:
            ax.axvline(4, color=ACCENT_3, lw=1.0, linestyle=':', alpha=0.6,
                       label=f'Lag 4  (r = {corrs[4]:+.3f})')

        ax.set_title(veg, color='white', fontsize=10, fontweight='bold', pad=6)
        ax.set_xlabel('Lag (weeks)', fontsize=8)
        ax.set_ylabel('Correlation (r)', fontsize=8)
        ax.set_xticks(range(0, MAX_LAG + 1))
        ax.set_ylim(-0.5, 0.5)
        ax.grid(color=GRID_COL, lw=0.4, linestyle='--', alpha=0.5)
        ax.legend(facecolor='#1c2128', labelcolor='white', fontsize=7,
                  loc='lower right', edgecolor='#30363d', framealpha=0.9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(SAVE_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  ✓ Saved: {filename}")

    # Print summary table
    print(f"\n  {'Vegetable':<22} {'Best Lag (wk)':>14} {'Correlation':>13}")
    print(f"  {'─'*50}")
    for veg, info in sorted(best_lags.items()):
        print(f"  {veg:<22} {info['best_lag']:>14}     {info['corr']:>+.3f}")

    return best_lags


# ═════════════════════════════════════════════════════════════════════════════
# CHART FUNCTION: Heatmap (vegetable × year) showing best lag per cell
# ═════════════════════════════════════════════════════════════════════════════
def plot_best_lag_heatmap(df, weather_col, price_cols, title, filename):
    """Create side-by-side heatmaps of best lag (vegetable × year) for two price targets."""

    fig, axes = plt.subplots(1, 2, figsize=(24, 12), facecolor=BG)
    fig.suptitle(title, color='white', fontsize=16, fontweight='bold', y=0.99)

    for ax_idx, (price_col, price_label) in enumerate(price_cols):
        # Build the best-lag matrix: veg × year
        lag_matrix = np.full((len(vegetables), len(years)), np.nan)
        corr_matrix = np.full((len(vegetables), len(years)), np.nan)

        for vi, veg in enumerate(vegetables):
            for yi, yr in enumerate(years):
                sub = df[(df['vegetable_type'] == veg) & (df['year'] == yr)]
                # Aggregate across markets for this year
                weekly_sub = (
                    sub.groupby('date')[[weather_col, price_col]]
                    .mean().reset_index().sort_values('date')
                )
                corrs = compute_lag_correlations(weekly_sub, weather_col, price_col)
                if not all(np.isnan(c) for c in corrs):
                    best_lag = int(np.nanargmax(np.abs(corrs)))
                    lag_matrix[vi, yi] = best_lag
                    corr_matrix[vi, yi] = corrs[best_lag]

        ax = axes[ax_idx]

        # Custom colormap: blue (lag 0) → yellow (lag 6) → red (lag 12)
        cmap = LinearSegmentedColormap.from_list('lag_cmap',
            ['#58a6ff', '#3fb950', '#f0e68c', '#ffa657', '#f85149'], N=13)

        im = ax.imshow(lag_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=12)

        # Annotate each cell with lag & correlation
        for vi in range(len(vegetables)):
            for yi in range(len(years)):
                lag_val = lag_matrix[vi, yi]
                corr_val = corr_matrix[vi, yi]
                if not np.isnan(lag_val):
                    text_color = 'black' if lag_val < 8 else 'white'
                    ax.text(yi, vi, f'{int(lag_val)}w\n{corr_val:+.2f}',
                            ha='center', va='center', fontsize=7.5,
                            fontweight='bold', color=text_color)

        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, fontsize=9)
        ax.set_yticks(range(len(vegetables)))
        ax.set_yticklabels(vegetables, fontsize=9)
        ax.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax.set_ylabel('Vegetable Type', fontsize=11, fontweight='bold')
        ax.set_title(f'→ {price_label}', color='white', fontsize=13, fontweight='bold', pad=10)

        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Best Lag (weeks)', color=TEXT_COL, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    plt.subplots_adjust(wspace=0.25, right=0.90)
    path = os.path.join(SAVE_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  ✓ Saved: {filename}")


# ═════════════════════════════════════════════════════════════════════════════
# CHART FUNCTION: Yearly breakdown per vegetable (small multiples)
# ═════════════════════════════════════════════════════════════════════════════
def plot_yearly_cross_correlation(df, weather_col, price_col, weather_label, price_label, accent_color, filename):
    """Create a large grid: rows = vegetables, columns = years."""

    fig, axes = plt.subplots(len(vegetables), len(years), figsize=(28, 36), facecolor=BG)
    fig.suptitle(
        f'Cross-Correlation: {weather_label} → {price_label}  (by Vegetable × Year, Lag 0–{MAX_LAG} weeks)',
        color='white', fontsize=16, fontweight='bold', y=0.995
    )

    for vi, veg in enumerate(vegetables):
        for yi, yr in enumerate(years):
            ax = axes[vi, yi]

            sub = df[(df['vegetable_type'] == veg) & (df['year'] == yr)]
            weekly_sub = (
                sub.groupby('date')[[weather_col, price_col]]
                .mean().reset_index().sort_values('date')
            )
            corrs = compute_lag_correlations(weekly_sub, weather_col, price_col)

            best_lag = int(np.nanargmax(np.abs(corrs)))

            colors = []
            for j, c in enumerate(corrs):
                if j == best_lag:
                    colors.append(ACCENT_2)
                elif c >= 0:
                    colors.append(accent_color)
                else:
                    colors.append('#f8514944')

            ax.bar(range(MAX_LAG + 1), corrs, color=colors, edgecolor=BG, linewidth=0.3, width=0.7)
            ax.axhline(0, color='#30363d', lw=0.6)
            ax.axvline(best_lag, color=BEST_COL, lw=1.2, linestyle='--', alpha=0.8)
            ax.set_ylim(-0.65, 0.65)
            ax.set_xticks([0, 4, 8, 12])
            ax.tick_params(labelsize=5.5)
            ax.grid(color=GRID_COL, lw=0.3, linestyle='--', alpha=0.4)

            # Annotate best lag
            ax.text(0.97, 0.93, f'best={best_lag}w\nr={corrs[best_lag]:+.2f}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=5.5, color=BEST_COL, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#1c2128', edgecolor='#30363d', alpha=0.8))

            for spine in ax.spines.values():
                spine.set_edgecolor('#21262d')

            # Row and column labels
            if yi == 0:
                ax.set_ylabel(veg, fontsize=7, fontweight='bold', color='white', rotation=90)
            if vi == 0:
                ax.set_title(str(yr), fontsize=9, fontweight='bold', color='white', pad=4)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    path = os.path.join(SAVE_DIR, filename)
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  ✓ Saved: {filename}")


# ═════════════════════════════════════════════════════════════════════════════
# GENERATE ALL CHARTS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("CHART 1/6: Rain → Retail Price (all years combined)")
print("="*65)
plot_cross_correlation_grid(
    df, 'rain_sum', 'retail_price',
    'Cross-Correlation: Rain Sum → Retail Price  (Lag 0–12 weeks, All Years)',
    ACCENT_1, ACCENT_2,
    '1_rain_vs_retail_price.png'
)

print("\n" + "="*65)
print("CHART 2/6: Rain → Farmer Price (all years combined)")
print("="*65)
plot_cross_correlation_grid(
    df, 'rain_sum', 'mean_farmer_price',
    'Cross-Correlation: Rain Sum → Farmer Price  (Lag 0–12 weeks, All Years)',
    ACCENT_4, ACCENT_2,
    '2_rain_vs_farmer_price.png'
)

print("\n" + "="*65)
print("CHART 3/6: Temperature → Retail Price (all years combined)")
print("="*65)
plot_cross_correlation_grid(
    df, 'mean_apparent_temperature', 'retail_price',
    'Cross-Correlation: Temperature → Retail Price  (Lag 0–12 weeks, All Years)',
    ACCENT_3, ACCENT_2,
    '3_temp_vs_retail_price.png'
)

print("\n" + "="*65)
print("CHART 4/6: Temperature → Farmer Price (all years combined)")
print("="*65)
plot_cross_correlation_grid(
    df, 'mean_apparent_temperature', 'mean_farmer_price',
    'Cross-Correlation: Temperature → Farmer Price  (Lag 0–12 weeks, All Years)',
    '#79c0ff', ACCENT_2,
    '4_temp_vs_farmer_price.png'
)

print("\n" + "="*65)
print("CHART 5/6: Best Lag Heatmap — Rain (Vegetable × Year)")
print("="*65)
plot_best_lag_heatmap(
    df, 'rain_sum',
    [('retail_price', 'Retail Price'), ('mean_farmer_price', 'Farmer Price')],
    'Best Rain Lag per Vegetable per Year  (annotated: lag + correlation)',
    '5_rain_best_lag_heatmap.png'
)

print("\n" + "="*65)
print("CHART 6/6: Best Lag Heatmap — Temperature (Vegetable × Year)")
print("="*65)
plot_best_lag_heatmap(
    df, 'mean_apparent_temperature',
    [('retail_price', 'Retail Price'), ('mean_farmer_price', 'Farmer Price')],
    'Best Temperature Lag per Vegetable per Year  (annotated: lag + correlation)',
    '6_temp_best_lag_heatmap.png'
)

# ═════════════════════════════════════════════════════════════════════════════
# BONUS: Full yearly breakdown grids (12 veg × 7 years)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("CHART 7/8: Rain → Retail Price (Vegetable × Year grid)")
print("="*65)
plot_yearly_cross_correlation(
    df, 'rain_sum', 'retail_price', 'Rain Sum', 'Retail Price',
    ACCENT_1, '7_rain_retail_yearly_grid.png'
)

print("\n" + "="*65)
print("CHART 8/8: Temperature → Retail Price (Vegetable × Year grid)")
print("="*65)
plot_yearly_cross_correlation(
    df, 'mean_apparent_temperature', 'retail_price', 'Temperature', 'Retail Price',
    ACCENT_3, '8_temp_retail_yearly_grid.png'
)

print("\n\n✅ All 8 charts saved to:")
print(f"   {SAVE_DIR}")
print("\nFiles:")
for f in sorted(os.listdir(SAVE_DIR)):
    if f.endswith('.png'):
        size = os.path.getsize(os.path.join(SAVE_DIR, f)) / 1024
        print(f"   {f}  ({size:.0f} KB)")
