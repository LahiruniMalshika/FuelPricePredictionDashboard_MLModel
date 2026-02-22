import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import matplotlib.pyplot as plt
import shap
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Sri Lanka Fuel Price Predictor",
    page_icon="⛽",
    layout="wide"
)

st.title("⛽ Sri Lanka Fuel Price Prediction Dashboard")
st.markdown("---")

# ========== LOAD MODELS & DATA ==========
@st.cache_resource
def load_models():
    """Load trained models, feature names, and scaler."""
    try:
        petrol_model = joblib.load('models/xgb_petrol92_best.pkl')
        diesel_model = joblib.load('models/xgb_diesel_best.pkl')
    except FileNotFoundError:
        petrol_model = joblib.load('models/xgb_petrol92.pkl')
        diesel_model = joblib.load('models/xgb_diesel.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    scaler = joblib.load('models/feature_scaler.pkl')
    return petrol_model, diesel_model, feature_names, scaler

@st.cache_data
def load_historical_data():
    """Load raw historical data for plotting."""
    df = pd.read_csv('data/merged_fuel_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date')

@st.cache_data
def load_background_data():
    """Load a sample of training features for SHAP background."""
    X_train = pd.read_csv('feature_engineered_data/X_train.csv')
    # Use a small random sample for speed
    return X_train.sample(n=min(100, len(X_train)), random_state=42)

# Load everything
try:
    petrol_model, diesel_model, feature_names, scaler = load_models()
    historical_df = load_historical_data()
    background_df = load_background_data()          # this contains all engineered features
    data_loaded = True
    st.sidebar.success("✅ Models loaded")
except Exception as e:
    st.error(f"Error loading models/data: {e}")
    st.info("Run the training notebooks first.")
    data_loaded = False

# ========== SIDEBAR INPUTS ==========
st.sidebar.header("📊 Input Parameters")
today = datetime.now().date()
end_of_2030 = datetime(2030, 12, 31).date()

prediction_date = st.sidebar.date_input(
    "📅 Prediction Date",
    min_value=today,
    max_value=end_of_2030,
    value=today
)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Economic Indicators")

if data_loaded:
    latest = historical_df.iloc[-1]
    crude_oil = st.sidebar.number_input("🛢️ Crude Oil (USD/barrel)", 0.0, 200.0, 75.0, 0.1)
    exchange_rate = st.sidebar.number_input("💱 USD/LKR Exchange Rate", 0.0, 500.0, float(latest['Exchange Rate']), 0.1)
    gdp_growth = st.sidebar.number_input("📊 GDP Growth (%)", -20.0, 20.0, 3.0, 0.1)
    inflation = st.sidebar.number_input("📈 Inflation (%)", -10.0, 100.0, 5.0, 0.1)
else:
    crude_oil = st.sidebar.number_input("Crude Oil", 0.0, 200.0, 75.0)
    exchange_rate = st.sidebar.number_input("Exchange Rate", 0.0, 500.0, 300.0)
    gdp_growth = st.sidebar.number_input("GDP Growth", -20.0, 20.0, 3.0)
    inflation = st.sidebar.number_input("Inflation", -10.0, 100.0, 5.0)

predict_button = st.sidebar.button("Predict", type="primary", use_container_width=True)

# ========== FEATURE PREPARATION ==========
def prepare_features(target_date, df, crude, exch, gdp, inf):
    """Create feature vector using historical data up to day before target."""
    target_date = pd.to_datetime(target_date)
    hist = df[df['Date'] < target_date].tail(31).copy()
    if len(hist) < 1:
        return None, None, None

    last_petrol = hist['LP_92'].iloc[-1]
    last_diesel = hist['LAD'].iloc[-1]
    petrol_30 = hist['LP_92'].tail(30).tolist()
    diesel_30 = hist['LAD'].tail(30).tolist()
    while len(petrol_30) < 30:
        petrol_30.insert(0, petrol_30[0])
    while len(diesel_30) < 30:
        diesel_30.insert(0, diesel_30[0])

    features = {
        'Month': target_date.month,
        'Month_Sin': np.sin(2*np.pi*target_date.month/12),
        'Month_Cos': np.cos(2*np.pi*target_date.month/12),
        'Quarter': (target_date.month-1)//3 + 1,
        'Is_Weekend': 1 if target_date.weekday() >= 5 else 0,
        'LP_92_lag_1': petrol_30[-1],
        'LP_92_lag_7': petrol_30[-7] if len(petrol_30)>=7 else petrol_30[-1],
        'LP_92_lag_30': petrol_30[0],
        'LAD_lag_1': diesel_30[-1],
        'LAD_lag_7': diesel_30[-7] if len(diesel_30)>=7 else diesel_30[-1],
        'LAD_lag_30': diesel_30[0],
        'LP_92_rolling_mean_7': np.mean(petrol_30[-7:]),
        'LP_92_rolling_std_7': np.std(petrol_30[-7:]),
        'LP_92_rolling_mean_30': np.mean(petrol_30),
        'LP_92_rolling_std_30': np.std(petrol_30),
        'LAD_rolling_mean_7': np.mean(diesel_30[-7:]),
        'LAD_rolling_std_7': np.std(diesel_30[-7:]),
        'LAD_rolling_mean_30': np.mean(diesel_30),
        'LAD_rolling_std_30': np.std(diesel_30),
        'Crude_Oil_USD': crude,
        'Crude_Oil_lag_7': crude,
        'Crude_Oil_lag_30': crude,
        'Exchange Rate': exch,
        'Exchange_Rate_pct_change': 0.0,
        'Crude_Oil_LKR': crude * exch,
        'GDP_Growth_Pct': gdp,
        'Inflation_Rate': inf,
        'Inflation_adj_factor': (inf/100) * exch,
        'Is_Sinhala_Tamil_New_Year': 1 if (target_date.month==4 and target_date.day in [13,14]) else 0,
        'Is_Vesak': 1 if (target_date.month==5 and 15<=target_date.day<=25) else 0,
        'Is_Christmas': 1 if (target_date.month==12 and target_date.day==25) else 0,
        'Is_New_Year': 1 if (target_date.month==1 and target_date.day==1) else 0,
        'Is_COVID_Period': 1 if target_date.year in [2020,2021] else 0,
        'Is_Crisis_2022': 1 if target_date.year==2022 else 0,
        'Is_Recovery_Period': 1 if target_date.year>=2023 else 0
    }
    return features, last_petrol, last_diesel

# ========== MAIN LOGIC ==========
if data_loaded and predict_button:
    with st.spinner("Calculating predictions..."):
        feat_dict, last_p92, last_d = prepare_features(
            prediction_date, historical_df,
            crude_oil, exchange_rate, gdp_growth, inflation
        )
        if feat_dict is None:
            st.error("Not enough historical data before the selected date.")
        else:
            # Create DataFrame with correct column order
            feat_df = pd.DataFrame([feat_dict])[feature_names]

            # Scale the features (RobustScaler)
            binary_cols = ['Is_Weekend', 'Is_Sinhala_Tamil_New_Year', 'Is_Vesak',
                           'Is_Christmas', 'Is_New_Year', 'Is_COVID_Period',
                           'Is_Crisis_2022', 'Is_Recovery_Period']
            cyclic_cols = ['Month_Sin', 'Month_Cos']
            cols_to_scale = [c for c in feature_names if c not in binary_cols and c not in cyclic_cols]
            feat_df_scaled = feat_df.copy()
            feat_df_scaled[cols_to_scale] = scaler.transform(feat_df[cols_to_scale])

            # Predict
            pred_p92 = petrol_model.predict(feat_df_scaled)[0]
            pred_d = diesel_model.predict(feat_df_scaled)[0]

            # ---- Display results ----
            st.success(f"✅ Prediction for {prediction_date.strftime('%B %d, %Y')}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⛽ Petrol 92", f"LKR {pred_p92:.2f}",
                         delta=f"{pred_p92 - last_p92:.2f}")
            with col2:
                st.metric("🚚 Auto Diesel", f"LKR {pred_d:.2f}",
                         delta=f"{pred_d - last_d:.2f}")
            with col3:
                days_ahead = (pd.to_datetime(prediction_date) - datetime.now()).days
                conf = max(0, 100 - days_ahead*3)
                st.metric("📊 Confidence", f"{conf:.0f}%", f"{days_ahead} days ahead")

            # ---- Debug section ----
            with st.expander("🔍 Debug: Feature values used for prediction"):
                st.write("**Feature vector (scaled)**")
                st.dataframe(feat_df_scaled.T.rename(columns={0:'value'}))
                st.write("**Feature vector (original)**")
                st.dataframe(feat_df.T.rename(columns={0:'value'}))

            # ---- Historical chart ----
            st.subheader("📈 Historical Trends")
            two_years_ago = datetime.now() - timedelta(days=1460)
            recent = historical_df[historical_df['Date'] >= two_years_ago]

            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=('Petrol 92', 'Auto Diesel'),
                                shared_xaxes=True, vertical_spacing=0.1)
            # Petrol
            fig.add_trace(go.Scatter(x=recent['Date'], y=recent['LP_92'],
                                      mode='lines', name='Petrol History',
                                      line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=[prediction_date], y=[pred_p92],
                                      mode='markers+text', name='Prediction',
                                      marker=dict(color='red', size=12, symbol='star'),
                                      text=[f'{pred_p92:.0f}'], textposition='top center'),
                          row=1, col=1)
            # Diesel
            fig.add_trace(go.Scatter(x=recent['Date'], y=recent['LAD'],
                                      mode='lines', name='Diesel History',
                                      line=dict(color='green'), showlegend=False),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=[prediction_date], y=[pred_d],
                                      mode='markers+text', name='Prediction',
                                      marker=dict(color='red', size=12, symbol='star'),
                                      text=[f'{pred_d:.0f}'], textposition='top center',
                                      showlegend=False), row=2, col=1)

            fig.update_layout(height=600, hovermode='x unified')
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="LKR", row=1, col=1)
            fig.update_yaxes(title_text="LKR", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

            # ---- SHAP explanation (FIXED) ----
            st.subheader("🔍 Why this prediction?")
            
                # 1. Use background data to establish a baseline
            background = background_df[feature_names]
                
                # 2. THE FIX: Pass petrol_model.predict instead of petrol_model
                # This bypasses the XGBoost internal JSON bug and uses a model-agnostic explainer
            explainer = shap.Explainer(petrol_model.predict, background)
                
                # 3. Calculate SHAP values
                # This directly returns a rich SHAP Explanation object
            shap_values = explainer(feat_df_scaled)

                # 4. Create the Waterfall plot
            fig, ax = plt.subplots(figsize=(10,6))
            shap.waterfall_plot(shap_values[0], show=False)
                
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("The waterfall plot shows how each feature shifts the prediction from the base average.")


# ---- Global Feature Importance ----
            st.subheader("🌍 Overall Feature Importance (Global)")
                # Get XGBoost built-in feature importance
            importances = petrol_model.feature_importances_
                
                # Create a DataFrame and sort for top 10
            imp_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=True).tail(10)
                
                # Plotly Horizontal Bar Chart
            fig_imp = go.Figure(go.Bar(
                    x=imp_df['Importance'],
                    y=imp_df['Feature'],
                    orientation='h',
                    marker_color='#1f77b4'
                ))
            fig_imp.update_layout(
                    title="Top 10 Most Influential Features (All-Time)",
                    xaxis_title="Relative Importance",
                    yaxis_title="Feature",
                    height=400
                )
            st.plotly_chart(fig_imp, use_container_width=True)
                
            st.markdown("""
                **What the model has learned:** The chart above shows the *global* behavior of the model. 
                It proves that historical lagged prices heavily dominate the prediction mechanism, while macroeconomic 
                factors like GDP and Inflation act as secondary adjustments.
                """)

                
            # ---- Feature importance reminder ----
            st.info("""
            **Note:** The most influential features are **lagged prices** (yesterday's price, last week's average).
            Changing GDP or inflation will have a smaller effect, especially for short‑term predictions.
            """)

else:
    # Welcome screen
    st.header("📋 Welcome")
    st.markdown("""
    Use the sidebar to enter economic indicators and a future date.  
    The model will predict fuel prices based on historical patterns and your inputs.

    **Tip:** Try different future dates or change economic indicators to see how predictions vary.
    """)
    if data_loaded:
        st.subheader("Latest actual prices")
        last = historical_df.iloc[-1]
        st.write(f"**Petrol 92:** LKR {last['LP_92']:.2f} on {last['Date'].date()}")
        st.write(f"**Auto Diesel:** LKR {last['LAD']:.2f} on {last['Date'].date()}")
    else:
        st.warning("Models not loaded.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center'>Data: Sri Lankan gov | Model: XGBoost/RF | ⚠️ Estimates only</div>",
            unsafe_allow_html=True)