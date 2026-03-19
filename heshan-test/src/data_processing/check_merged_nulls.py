import pandas as pd

# Load the merged dataset we created earlier
df = pd.read_csv('data/main_with_producer_prices.csv')

total_rows = len(df)
missing_producer = df['producer_price'].isna().sum()
missing_retail = df['price'].isna().sum()

print("=" * 60)
print("MAIN DATASET - PRODUCER PRICE INTEGRATION ANALYSIS")
print("=" * 60)
print(f"Total Rows in Main Dataset: {total_rows:,}")
print(f"Missing Producer Prices: {missing_producer:,} ({(missing_producer/total_rows)*100:.2f}%)")
print(f"Missing Retail Prices: {missing_retail:,} ({(missing_retail/total_rows)*100:.2f}%)")

# Let's break down WHY they are missing when merged
print("\n" + "-" * 60)
print("MISSING PRODUCER PRICES BY LOCATION:")
print("-" * 60)
loc_breakdown = df.groupby('location')['producer_price'].apply(lambda x: sorted([
    f"Missing: {x.isna().sum():,} ({(x.isna().sum()/len(x))*100:.1f}%)"
])[0]).reset_index()

# Actually let's do a better dataframe summary
loc_stats = df.groupby('location').agg(
    total_rows=('producer_price', 'size'),
    missing=('producer_price', lambda x: x.isna().sum())
)
loc_stats['missing_percent'] = (loc_stats['missing'] / loc_stats['total_rows'] * 100).round(1)
print(loc_stats.sort_values('missing_percent', ascending=False))

print("\n" + "-" * 60)
print("MISSING PRODUCER PRICES BY VEGETABLE:")
print("-" * 60)
veg_stats = df.groupby('vegetable_type').agg(
    total_rows=('producer_price', 'size'),
    missing=('producer_price', lambda x: x.isna().sum())
)
veg_stats['missing_percent'] = (veg_stats['missing'] / veg_stats['total_rows'] * 100).round(1)
print(veg_stats.sort_values('missing_percent', ascending=False))

