import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_and_plot_outliers():
    # Setup Paths
    data_path = r'C:\Users\Ranuga\Data Science Project\3. Data Preprocessing\3.7 - Combining Datasets\Outputs\Final_Combined_data.csv'
    base_dir = r'C:\Users\Ranuga\Data Science Project\4. Data Visualization\4.4 - Outlier Analysis'
    
    charts_dir = os.path.join(base_dir, 'Charts')
    reports_dir = os.path.join(base_dir, 'Reports')
    ind_charts_dir = os.path.join(charts_dir, 'Individual_Vegetables')

    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(ind_charts_dir, exist_ok=True)

    print("Loading datasets...")
    df = pd.read_csv(data_path)

    # 1. IQR Outlier Calculation & Report Generation
    print("Calculating outlier statistics (IQR Method)...")
    outlier_stats = []
    
    for veg in df['vegetable_type'].unique():
        veg_data = df[df['vegetable_type'] == veg]['retail_price'].dropna()
        if len(veg_data) == 0:
            continue
            
        Q1 = veg_data.quantile(0.25)
        Q3 = veg_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = veg_data[(veg_data < lower_bound) | (veg_data > upper_bound)]
        
        outlier_stats.append({
            'Vegetable_Type': veg,
            'Total_Records': len(veg_data),
            'Outlier_Count': len(outliers),
            'Outlier_Percentage': round((len(outliers) / len(veg_data)) * 100, 2),
            'Lower_Bound': round(lower_bound, 2),
            'Upper_Bound': round(upper_bound, 2),
            'Max_Price_Found': round(veg_data.max(), 2)
        })

    stats_df = pd.DataFrame(outlier_stats).sort_values(by='Outlier_Percentage', ascending=False)
    stats_df.to_csv(os.path.join(reports_dir, 'outlier_summary.csv'), index=False)
    print("Outlier summary report saved!")

    # 2. Master Boxplot: Overlaid view of all vegetables
    print("Generating Master Boxplot...")
    plt.figure(figsize=(18, 12))
    sns.boxplot(data=df, y='vegetable_type', x='retail_price', flierprops={"marker": "x", "markerfacecolor": "red", "markersize": 5})
    plt.title('Retail Price Distributions & Outliers for All Vegetables (Complete Year Period)', fontsize=18, fontweight='bold')
    plt.xlabel('Retail Price (LKR)', fontsize=14)
    plt.ylabel('Vegetable Type', fontsize=14)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'All_Vegetables_Boxplot_Master.png'), dpi=300)
    plt.close()

    # 3. Individual Boxplots (Farmer Price vs Retail Price)
    print("Generating individual vegetable boxplots...")
    for veg in df['vegetable_type'].unique():
        veg_df = df[df['vegetable_type'] == veg]
        if veg_df.empty: 
            continue
        
        # Melt dataframe to plot both Farmer and Retail prices side-by-side
        melted = veg_df.melt(id_vars=['year', 'week'], 
                             value_vars=['mean_farmer_price', 'retail_price'], 
                             var_name='Price_Type', 
                             value_name='Price')
                             
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=melted, x='Price_Type', y='Price', 
                    palette=['#88CCEE', '#FF8888'], 
                    flierprops={"marker": "o", "color": "red", "alpha": 0.5})
                    
        plt.title(f'Outlier Detection over Complete Period: {veg}', fontsize=16)
        plt.ylabel('Price (LKR)', fontsize=12)
        plt.xlabel('Price Origin Category', fontsize=12)
        plt.xticks(ticks=[0, 1], labels=['Farmer Price', 'Retail Price'])
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Save securely by parsing unsafe string characters
        safe_name = str(veg).replace('/', '_').replace('\\', '_')
        plt.savefig(os.path.join(ind_charts_dir, f'{safe_name}_Outlier_Boxplot.png'), dpi=200)
        plt.close()

    print("All tasks finished successfully. Check the 4.4 - Outlier Analysis directory!")

if __name__ == "__main__":
    analyze_and_plot_outliers()