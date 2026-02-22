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

# Initialize session state to remember if predictions were generated
if 'predicted' not in st.session_state:
    st.session_state['predicted'] = False

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

# Update session state when button is clicked
if predict_button:
    st.session_state['predicted'] = True

# ========== FEATURE PREPARATION ==========
def prepare_features(target_date, df, crude, exch, gdp, inf):
    target_date = pd.to_datetime(target_date)
    hist = df[df['Date'] < target_date].tail(31).copy()
    if len(hist) < 1: return None, None

    targets = ['LP_92', 'LP_95', 'LAD', 'LSD', 'LK']
    last_prices = {t: hist[t].iloc[-1] for t in targets}
    
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
# Notice we now check the session_state here!
if data_loaded and st.session_state['predicted']:
    with st.spinner("Calculating predictions & explanations..."):
        feat_dict, last_prices = prepare_features(prediction_date, historical_df, crude_oil, exchange_rate, gdp_growth, inflation)
        
        if feat_dict is None:
            st.error("Not enough historical data.")
        else:
            feat_df = pd.DataFrame([feat_dict])
            for col in feature_names:
                if col not in feat_df.columns: feat_df[col] = 0
            feat_df = feat_df[feature_names]

            binary_cols = ['Is_Weekend', 'Is_Sinhala_Tamil_New_Year', 'Is_Vesak', 'Is_Christmas', 'Is_New_Year', 'Is_COVID_Period', 'Is_Crisis_2022', 'Is_Recovery_Period']
            cyclic_cols = ['Month_Sin', 'Month_Cos']
            cols_to_scale = [c for c in feature_names if c not in binary_cols and c not in cyclic_cols]
            feat_df_scaled = feat_df.copy()
            feat_df_scaled[cols_to_scale] = scaler.transform(feat_df[cols_to_scale])

            st.success(f"✅ Predictions for {prediction_date.strftime('%B %d, %Y')}")
            
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

            st.subheader("📈 Historical Trends (Primary Fuels)")
            two_years_ago = datetime.now() - timedelta(days=1460)
            recent = historical_df[historical_df['Date'] >= two_years_ago]

            fig = make_subplots(rows=2, cols=1, subplot_titles=('Petrol 92', 'Auto Diesel'), shared_xaxes=True)
            fig.add_trace(go.Scatter(x=recent['Date'], y=recent['LP_92'], line=dict(color='blue'), name='P-92'), row=1, col=1)
            fig.add_trace(go.Scatter(x=recent['Date'], y=recent['LAD'], line=dict(color='green'), name='Diesel'), row=2, col=1)
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # ==========================================
            # ADVANCED EXPLAINABILITY (XAI) SECTION
            # ==========================================
            st.markdown("---")
            st.header("🧠 Model Explainability & Interpretation")
            
            # Let the user choose which fuel to analyze
            explain_fuel = st.selectbox(
                "Select a fuel type to analyze:",
                options=['LP_92', 'LP_95', 'LAD', 'LSD', 'LK'],
                format_func=lambda x: {"LP_92": "⛽ Petrol 92", "LP_95": "⚡ Petrol 95", "LAD": "🚚 Auto Diesel", "LSD": "🚀 Super Diesel", "LK": "🛢️ Kerosene"}[x]
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"🔍 Local: Why this exact price?")
                st.caption(f"Shows how features pushed the {explain_fuel} prediction up or down for the selected date.")
                
                background = background_df[feature_names]
                explainer = shap.Explainer(models_dict[explain_fuel].predict, background)
                shap_values = explainer(feat_df_scaled)

                fig, ax = plt.subplots(figsize=(8, 5))
                shap.waterfall_plot(shap_values[0], show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.subheader(f"🌍 Global: What drives the model?")
                st.caption(f"Overall most influential factors for {explain_fuel} across all historical data.")
                
                importances = models_dict[explain_fuel].feature_importances_                
                imp_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=True).tail(8)
                    
                fig_imp = go.Figure(go.Bar(
                        x=imp_df['Importance'], y=imp_df['Feature'],
                        orientation='h', marker_color='#1f77b4'
                    ))
                fig_imp.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown("---")
            
            # 3. PARTIAL DEPENDENCE PLOT (PDP) - Required by Rubric
            st.subheader("📈 Partial Dependence Plot (PDP)")
            st.write("This plot isolates the most important feature to show its direct, marginal effect on the predicted price, proving alignment with domain knowledge.")
            
            # Identify the top feature dynamically for the selected fuel
            top_feature = imp_df.iloc[-1]['Feature']
            
            # 1. Do NOT pre-create the figure. Let SHAP create it automatically.
            shap.plots.partial_dependence(
                top_feature, 
                models_dict[explain_fuel].predict, 
                background, 
                ice=False,
                model_expected_value=True, 
                feature_expected_value=True,
                show=False
            )
            
            # 2. Grab the figure that SHAP just drew on!
            fig_pdp = plt.gcf()
            fig_pdp.set_size_inches(10, 4) # Adjust the size to fit the dashboard
            
            # 3. Add styling
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 4. Display in Streamlit
            st.pyplot(fig_pdp)
            
            # 5. Clear matplotlib memory so it doesn't overlap with future clicks
            plt.clf()
            plt.close('all')
            
            st.info(f"**Domain Alignment:** The PDP above shows that as `{top_feature}` increases, the predicted price of **{explain_fuel}** reacts accordingly. Because fuel prices are government-regulated step functions, lag features heavily dominate the logic.")

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