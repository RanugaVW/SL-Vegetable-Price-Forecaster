import pandas as pd
import os

def generate_seasonal_datasets():
    print("Loading Master Dataset...")
    # Load Master Dataset
    data_path = r'C:\Users\Ranuga\Data Science Project\3. Data Preprocessing\3.7 - Combining Datasets\Outputs\Final_Combined_data.csv'
    df = pd.read_csv(data_path)
    
    # New Output Directory Setup
    base_dir = r'C:\Users\Ranuga\Data Science Project\5. Model Building\5.10 - Yala and Maha Seperation'
    data_dir = os.path.join(base_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Filter Yala Season
    df_yala = df[df['seasonality'].str.contains('Yala', case=False, na=False)]
    yala_path = os.path.join(data_dir, 'Yala_data.csv')
    df_yala.to_csv(yala_path, index=False)
    print(f"Yala Season generated: {len(df_yala)} rows -> Saved to: {yala_path}")

    # Filter Maha Season
    df_maha = df[df['seasonality'].str.contains('Maha', case=False, na=False)]
    maha_path = os.path.join(data_dir, 'Maha_data.csv')
    df_maha.to_csv(maha_path, index=False)
    print(f"Maha Season generated: {len(df_maha)} rows -> Saved to: {maha_path}")
    
    print("\nDataset split complete!")

if __name__ == "__main__":
    generate_seasonal_datasets()
