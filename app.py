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
st.set_page_config(page_title="Sri Lanka Fuel Price Predictor", page_icon="⛽", layout="wide")

st.title("⛽ Sri Lanka Fuel Price Prediction Dashboard")
st.markdown("---")

# ========== LOAD MODELS & DATA ==========
@st.cache_resource
def load_models():
    """Load all 5 trained models, feature names, and scaler."""
    models = {}
    fuels = ['lp_92', 'lp_95', 'lad', 'lsd', 'lk']
    
    for f in fuels:
        try:
            models[f.upper()] = joblib.load(f'models/xgb_{f}_best.pkl')
        except FileNotFoundError:
            # Fallback to standard naming if best isn't found
            try:
                models[f.upper()] = joblib.load(f'models/xgb_{f}.pkl')
            except:
                st.error(f"Missing model for {f.upper()}")
                
    feature_names = joblib.load('models/feature_names.pkl')
    scaler = joblib.load('models/feature_scaler.pkl')
    return models, feature_names, scaler

@st.cache_data
def load_historical_data():
    df = pd.read_csv('data/merged_fuel_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date')

@st.cache_data
def load_background_data():
    X_train = pd.read_csv('feature_engineered_data/X_train.csv')
    return X_train.sample(n=min(100, len(X_train)), random_state=42)

try:
    models_dict, feature_names, scaler = load_models()
    historical_df = load_historical_data()
    background_df = load_background_data()
    data_loaded = len(models_dict) > 0
    st.sidebar.success(f"✅ Loaded {len(models_dict)} models")
except Exception as e:
    st.error(f"Error loading models/data: {e}")
    data_loaded = False

# ========== SIDEBAR INPUTS ==========
st.sidebar.header("📊 Input Parameters")
today = datetime.now().date()
prediction_date = st.sidebar.date_input("📅 Prediction Date", min_value=today, max_value=datetime(2030, 12, 31).date(), value=today)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Economic Indicators")

if data_loaded:
    latest = historical_df.iloc[-1]
    crude_oil = st.sidebar.number_input("🛢️ Crude Oil (USD/barrel)", 0.0, 200.0, float(latest['Crude_Oil_USD']), 0.1)
    exchange_rate = st.sidebar.number_input("💱 USD/LKR Exchange Rate", 0.0, 500.0, float(latest['Exchange Rate']), 0.1)
    gdp_growth = st.sidebar.number_input("📊 GDP Growth (%)", -20.0, 20.0, float(latest['GDP_Growth_Pct']), 0.1)
    inflation = st.sidebar.number_input("📈 Inflation (%)", -10.0, 100.0, float(latest['Inflation_Rate']), 0.1)
else:
    crude_oil, exchange_rate, gdp_growth, inflation = 75.0, 300.0, 3.0, 5.0

predict_button = st.sidebar.button("Predict All Fuels", type="primary", use_container_width=True)

# ========== FEATURE PREPARATION ==========
def prepare_features(target_date, df, crude, exch, gdp, inf):
    target_date = pd.to_datetime(target_date)
    hist = df[df['Date'] < target_date].tail(31).copy()
    if len(hist) < 1: return None, None

    targets = ['LP_92', 'LP_95', 'LAD', 'LSD', 'LK']
    last_prices = {t: hist[t].iloc[-1] for t in targets}
    
    # Base Features
    features = {
        'Month': target_date.month,
        'Month_Sin': np.sin(2*np.pi*target_date.month/12),
        'Month_Cos': np.cos(2*np.pi*target_date.month/12),
        'Quarter': (target_date.month-1)//3 + 1,
        'Is_Weekend': 1 if target_date.weekday() >= 5 else 0,
        'Crude_Oil_USD': crude, 'Crude_Oil_lag_7': crude, 'Crude_Oil_lag_30': crude,
        'Exchange Rate': exch, 'Exchange_Rate_pct_change': 0.0,
        'Crude_Oil_LKR': crude * exch, 'GDP_Growth_Pct': gdp,
        'Inflation_Rate': inf, 'Inflation_adj_factor': (inf/100) * exch,
        'Is_Sinhala_Tamil_New_Year': 1 if (target_date.month==4 and target_date.day in [13,14]) else 0,
        'Is_Vesak': 1 if (target_date.month==5 and 15<=target_date.day<=25) else 0,
        'Is_Christmas': 1 if (target_date.month==12 and target_date.day==25) else 0,
        'Is_New_Year': 1 if (target_date.month==1 and target_date.day==1) else 0,
        'Is_COVID_Period': 1 if target_date.year in [2020,2021] else 0,
        'Is_Crisis_2022': 1 if target_date.year==2022 else 0,
        'Is_Recovery_Period': 1 if target_date.year>=2023 else 0
    }

    # Dynamic lags for all 5 targets
    for t in targets:
        vals = hist[t].tail(30).tolist()
        while len(vals) < 30: vals.insert(0, vals[0])
        
        features[f'{t}_lag_1'] = vals[-1]
        features[f'{t}_lag_7'] = vals[-7] if len(vals)>=7 else vals[-1]
        features[f'{t}_lag_30'] = vals[0]
        features[f'{t}_rolling_mean_7'] = np.mean(vals[-7:])
        features[f'{t}_rolling_std_7'] = np.std(vals[-7:])
        
    return features, last_prices

# ========== MAIN LOGIC ==========
if data_loaded and predict_button:
    with st.spinner("Calculating predictions..."):
        feat_dict, last_prices = prepare_features(prediction_date, historical_df, crude_oil, exchange_rate, gdp_growth, inflation)
        
        if feat_dict is None:
            st.error("Not enough historical data.")
        else:
            feat_df = pd.DataFrame([feat_dict])
            # Filter and order columns
            for col in feature_names:
                if col not in feat_df.columns: feat_df[col] = 0
            feat_df = feat_df[feature_names]

            # Scale features
            binary_cols = ['Is_Weekend', 'Is_Sinhala_Tamil_New_Year', 'Is_Vesak', 'Is_Christmas', 'Is_New_Year', 'Is_COVID_Period', 'Is_Crisis_2022', 'Is_Recovery_Period']
            cyclic_cols = ['Month_Sin', 'Month_Cos']
            cols_to_scale = [c for c in feature_names if c not in binary_cols and c not in cyclic_cols]
            feat_df_scaled = feat_df.copy()
            feat_df_scaled[cols_to_scale] = scaler.transform(feat_df[cols_to_scale])

            st.success(f"✅ Predictions for {prediction_date.strftime('%B %d, %Y')}")
            
            # Predict & Display 5 Metric Cards
            metrics_config = [
                ("⛽ Petrol 92", "LP_92"),
                ("⚡ Petrol 95", "LP_95"),
                ("🚚 Auto Diesel", "LAD"),
                ("🚀 Super Diesel", "LSD"),
                ("🛢️ Kerosene", "LK")
            ]
            
            cols = st.columns(5)
            for idx, (title, key) in enumerate(metrics_config):
                if key in models_dict:
                    pred_val = models_dict[key].predict(feat_df_scaled)[0]
                    with cols[idx]:
                        st.metric(title, f"LKR {pred_val:.2f}", delta=f"{pred_val - last_prices[key]:.2f}")

            # Plotting History (For 92 and Auto Diesel as reference)
            st.subheader("📈 Historical Trends (Primary Fuels)")
            two_years_ago = datetime.now() - timedelta(days=1460)
            recent = historical_df[historical_df['Date'] >= two_years_ago]

            fig = make_subplots(rows=2, cols=1, subplot_titles=('Petrol 92', 'Auto Diesel'), shared_xaxes=True)
            fig.add_trace(go.Scatter(x=recent['Date'], y=recent['LP_92'], line=dict(color='blue'), name='P-92'), row=1, col=1)
            fig.add_trace(go.Scatter(x=recent['Date'], y=recent['LAD'], line=dict(color='green'), name='Diesel'), row=2, col=1)
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # SHAP Interpretation (Using Petrol 92 as primary example)
            st.subheader("🔍 Interpreting the Result (Petrol 92 Baseline)")
            background = background_df[feature_names]
            explainer = shap.Explainer(models_dict['LP_92'].predict, background)
            shap_values = explainer(feat_df_scaled)

            fig, ax = plt.subplots(figsize=(10,6))
            shap.waterfall_plot(shap_values[0], show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

else:
    st.header("📋 Welcome")
    st.markdown("Use the sidebar to enter economic indicators and a future date. The model will predict prices for Sri Lanka's 5 major consumer fuels.")
    
    if data_loaded:
        st.subheader("Latest actual prices")
        last = historical_df.iloc[-1]
        cols = st.columns(5)
        cols[0].write(f"**P-92:** LKR {last['LP_92']}")
        cols[1].write(f"**P-95:** LKR {last['LP_95']}")
        cols[2].write(f"**Diesel:** LKR {last['LAD']}")
        cols[3].write(f"**S-Diesel:** LKR {last['LSD']}")
        cols[4].write(f"**Kerosene:** LKR {last['LK']}")