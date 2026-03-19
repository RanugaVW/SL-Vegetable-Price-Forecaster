import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title="Sri Lankan Veggie Prices Analysis", layout="wide")

st.title("🥬 Sri Lankan Vegetable Prices Analysis App")
st.markdown("A full-stack interactive data analysis application for exploring time-series data, feature engineering, statistical testing, and predictive modeling for vegetable prices.")

# --- 1. DATA LOADING ---
@st.cache_data
def load_data(file_buffer):
    if file_buffer is not None:
        df = pd.read_csv(file_buffer)
    else:
        # Fallback to local file if available
        try:
            df = pd.read_csv('data/main.csv')
        except FileNotFoundError:
            return None
    
    # Preprocessing
    if 'Year_Week' in df.columns:
        # Create a proxy date for time series plotting based on Year and Week if not actual dates
        if 'year' in df.columns and 'week' in df.columns:
            # Monday of the given week
            df['Proxy_Date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['week'].astype(str) + '-1', format='%Y-%W-%w', errors='coerce')
        else:
            df['Proxy_Date'] = df['Year_Week']
    return df

st.sidebar.header("📁 1. Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")
data = load_data(uploaded_file)

if data is None:
    st.warning("Please upload a CSV file or ensure 'data/main.csv' is in the directory.")
    st.stop()

# Keep original data safe
df = data.copy()

# --- 2. SIDEBAR CONTROLS (LEFT PANEL) ---
st.sidebar.header("🎛️ 2. Filters & Selectors")

# Categorical filters
col_veg = 'vegetable_type' if 'vegetable_type' in df.columns else None
col_loc = 'location' if 'location' in df.columns else None
col_season = 'seasonality' if 'seasonality' in df.columns else None

if col_veg:
    veg_options = df[col_veg].dropna().unique()
    selected_vegs = st.sidebar.multiselect("Select Vegetable", options=veg_options, default=veg_options[:2])
    if selected_vegs: df = df[df[col_veg].isin(selected_vegs)]

if col_loc:
    loc_options = df[col_loc].dropna().unique()
    selected_locs = st.sidebar.multiselect("Select Location", options=loc_options, default=loc_options[:2])
    if selected_locs: df = df[df[col_loc].isin(selected_locs)]

if col_season:
    season_options = df[col_season].dropna().unique()
    selected_season = st.sidebar.multiselect("Select Season", options=season_options, default=season_options)
    if selected_season: df = df[df[col_season].isin(selected_season)]

# Basic Variable Selection
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
st.sidebar.subheader("Variable Selection")
x_var = st.sidebar.selectbox("X-axis Variable", options=df.columns.tolist(), index=list(df.columns).index("Proxy_Date") if "Proxy_Date" in df.columns else 0)
y_var = st.sidebar.selectbox("Y-axis Variable (Target)", options=numeric_cols, index=numeric_cols.index("price") if "price" in numeric_cols else 0)

# Target enforcement for modeling and correlation
target_col = y_var

# --- 3. FEATURE ENGINEERING PANEL ---
st.sidebar.header("🛠️ 3. Feature Engineering")
apply_log = st.sidebar.checkbox(f"Log Transform ({target_col})")
scaler_choice = st.sidebar.selectbox("Scaling", ["None", "Standardization (Z-score)", "Min-Max Scaling"])
diff_lag = st.sidebar.selectbox("Differencing", [0, 1, 4, 52], format_func=lambda x: f"Lag {x}" if x > 0 else "None")
rolling_window = st.sidebar.slider("Rolling Mean/Std Window", min_value=1, max_value=52, value=4, step=1)

# Apply Transformations
if apply_log:
    df[f"log_{target_col}"] = np.log1p(df[target_col])
    target_col = f"log_{target_col}"

if diff_lag > 0:
    df[f"{target_col}_diff_{diff_lag}"] = df[target_col].diff(diff_lag)
    target_col = f"{target_col}_diff_{diff_lag}"

df[f"{target_col}_rolling_mean"] = df[target_col].rolling(window=rolling_window).mean()
df[f"{target_col}_rolling_std"] = df[target_col].rolling(window=rolling_window).std()

if scaler_choice == "Standardization (Z-score)":
    scaler = StandardScaler()
    num_cols_to_scale = df.select_dtypes(include=np.number).columns
    df[num_cols_to_scale] = scaler.fit_transform(df[num_cols_to_scale])
elif scaler_choice == "Min-Max Scaling":
    scaler = MinMaxScaler()
    num_cols_to_scale = df.select_dtypes(include=np.number).columns
    df[num_cols_to_scale] = scaler.fit_transform(df[num_cols_to_scale])

# Sort and clean data for Time Series
if 'Proxy_Date' in df.columns:
    df = df.sort_values(by='Proxy_Date')

# --- MAIN AREA (RIGHT PANEL) ---
tab_viz, tab_corr, tab_stats, tab_ts, tab_ml, tab_data = st.tabs([
    "📊 Visualizations", 
    "🔗 Correlation & Features", 
    "📈 Statistical Analysis",
    "⏳ Time Series",
    "🤖 ML & Models",
    "💾 Data View"
])

# ------------- TAB 1: VISUALIZATIONS -------------
with tab_viz:
    st.header("Visualizations")
    graph_type = st.selectbox("Select Graph Type", [
        "Line Plot (Time Series)", "Scatter Plot", "Boxplot", "Histogram + KDE", "Rolling Averages"
    ])
    
    if graph_type == "Line Plot (Time Series)":
        if 'Proxy_Date' in df.columns:
            fig = px.line(df, x='Proxy_Date', y=target_col, color=col_veg, title=f"Time Series: {target_col} over Time")
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("No date proxy found for Line Plot.")
            
    elif graph_type == "Scatter Plot":
        fig = px.scatter(df, x=x_var, y=target_col, color=col_veg, hover_data=df.columns, title=f"Scatter: {x_var} vs {target_col}")
        st.plotly_chart(fig, width='stretch')
        
    elif graph_type == "Boxplot":
        box_x = st.selectbox("Boxplot Category", [col_veg, col_loc, col_season] if col_veg else df.columns)
        fig = px.box(df, x=box_x, y=target_col, color=box_x, title=f"Boxplot: {target_col} by {box_x}")
        st.plotly_chart(fig, width='stretch')
        
    elif graph_type == "Histogram + KDE":
        fig = plt.figure(figsize=(10,4))
        sns.histplot(df[target_col].dropna(), kde=True, bins=30)
        st.pyplot(fig)
        
    elif graph_type == "Rolling Averages":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Proxy_Date'] if 'Proxy_Date' in df.columns else df.index, y=df[target_col], name='Original Target', opacity=0.3))
        fig.add_trace(go.Scatter(x=df['Proxy_Date'] if 'Proxy_Date' in df.columns else df.index, y=df[f"{target_col}_rolling_mean"], name=f'Rolling Mean ({rolling_window})', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df['Proxy_Date'] if 'Proxy_Date' in df.columns else df.index, y=df[f"{target_col}_rolling_std"], name=f'Volatility (Rolling Std)', line=dict(color='orange')))
        st.plotly_chart(fig, width='stretch')

# ------------- TAB 2: CORRELATION & FEATURES -------------
with tab_corr:
    st.header("Correlation Optimization")
    st.markdown("Recompute correlations after feature engineering to see what impacts the target most.")
    
    numeric_df = df.select_dtypes(include=np.number).dropna()
    
    if len(numeric_df) > 10:
        corr_matrix = numeric_df.corr(method='pearson')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"Top Correlated with {target_col}")
            corrs = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
            st.dataframe(corrs.to_frame(name="Pearson r"))
            
            top_feature = corrs.index[0] if len(corrs) > 0 else None
            if top_feature:
                st.success(f"Insight: **{top_feature}** has the strongest correlation ({corrs.iloc[0]:.2f}) with {target_col}.")
                
        with col2:
            st.subheader("Correlation Heatmap")
            fig = plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
            st.pyplot(fig)
    else:
        st.warning("Not enough numeric data for correlation.")

# ------------- TAB 3: STATISTICAL ANALYSIS -------------
with tab_stats:
    st.header("Summary & Statistical Tests")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Summary Statistics")
        st.dataframe(df[target_col].describe())
        
    with col2:
        st.subheader("Stationarity Test (ADF)")
        st.markdown("Tests if the time series is stationary (p-value < 0.05).")
        ts_data = df[target_col].dropna()
        if len(ts_data) > 10:
            result = adfuller(ts_data)
            st.write(f"ADF Statistic: {result[0]:.4f}")
            st.write(f"p-value: {result[1]:.4f}")
            if result[1] <= 0.05:
                st.success("Series is Stationary (Reject Null Hypothesis)")
            else:
                st.error("Series is Non-Stationary (Fail to Reject Null Hypothesis). Consider differencing.")

# ------------- TAB 4: TIME SERIES -------------
with tab_ts:
    st.header("Time Series Analysis")
    if 'Proxy_Date' in df.columns and len(df) > 52:
        ts_df = df.set_index('Proxy_Date')[target_col].dropna()
        # Resample or just use numeric index for statsmodels if indices are not strictly uniform
        try:
            decomposition = seasonal_decompose(ts_df, model='additive', period=52)
            fig = decomposition.plot()
            fig.set_size_inches(12, 8)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not perform seasonal decomposition. Ensure enough data points and standard frequency. Error: {e}")
    else:
        st.info("Ensure the dataset has valid dates/frequencies and sufficient length for Time Series Decomposition.")

# ------------- TAB 5: ML & MODELS -------------
with tab_ml:
    st.header("Predictive Modeling (Baseline)")
    
    available_features = [c for c in numeric_df.columns if c != target_col]
    default_features = available_features[:3] if len(available_features) >= 3 else available_features
    ml_features = st.multiselect("Select Features for Modeling", options=available_features, default=default_features)
    model_type = st.radio("Model", ["Linear Regression", "Random Forest"])
    
    if st.button("Train Model"):
        if len(ml_features) > 0:
            X = numeric_df[ml_features]
            y = numeric_df[target_col]
            
            # Simple train-test split (chronological if possible)
            train_size = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            if model_type == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Score", f"{r2_score(y_test, preds):.4f}")
            with col2:
                st.metric("MAE", f"{mean_absolute_error(y_test, preds):.4f}")
                
            if model_type == "Random Forest":
                st.subheader("Feature Importance")
                importance = pd.DataFrame({'Feature': ml_features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
                fig = px.bar(importance, x='Importance', y='Feature', orientation='h', title="RF Feature Importance")
                st.plotly_chart(fig)
                
        else:
            st.warning("Select at least one feature.")

# ------------- TAB 6: DATA EXPORT -------------
with tab_data:
    st.header("Processed Dataset Preview")
    st.dataframe(df.head(50))
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Processed Data as CSV",
        data=csv,
        file_name='processed_veggie_data.csv',
        mime='text/csv',
    )
