# water_quality_forecast.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# ✅ Set up Streamlit page
st.set_page_config(page_title="💧 Water Quality Forecast", layout="wide")

# ------------------- Load Dataset -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    return df

df = load_data()

# ------------------- Feature Units -------------------
feature_units = {
    'ph': 'pH',
    'Hardness': 'mg/L',
    'Solids': 'mg/L',
    'Chloramines': 'µg/L',
    'Sulfate': 'mg/L',
    'Conductivity': 'µS/cm',
    'Organic_carbon': 'µg/L',
    'Trihalomethanes': 'µg/L',
    'Turbidity': 'NTU'
}

# ------------------- Forecast Function -------------------
def run_forecast(df, feature, extra_col, start_date, end_date):
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    monthly_df = df.groupby('Date').agg({feature: 'mean', extra_col: 'mean'}).reset_index()
    monthly_df.columns = ['ds', 'y', extra_col.lower()]
    monthly_df = monthly_df.dropna()

    model = Prophet(yearly_seasonality=True)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_regressor(extra_col.lower())
    model.fit(monthly_df)

    # Full forecast range: Jan 2023 to Dec 2034
    full_range = pd.date_range(start='2023-01-01', end='2034-12-01', freq='MS')
    last_val = monthly_df[extra_col.lower()].iloc[-1]
    future_df = pd.DataFrame({'ds': full_range})
    future_df[extra_col.lower()] = last_val

    forecast = model.predict(future_df)
    forecast['Date'] = forecast['ds']
    forecast = forecast[(forecast['Date'] >= pd.to_datetime(start_date)) & (forecast['Date'] <= pd.to_datetime(end_date))]

    # Plotting
    unit = feature_units.get(feature, '')
    st.subheader(f"📊 Forecast for {feature} ({unit})")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(forecast['Date'], forecast['yhat'], marker='o', linestyle='-', color='teal')
    ax.set_title(f"Forecast of {feature} from {start_date[:7]} to {end_date[:7]}")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{feature} ({unit})")
    ax.grid(True)
    st.pyplot(fig)

    return forecast[['ds', 'yhat']]

# ------------------- Streamlit UI -------------------
st.title("💧 Water Quality Forecasting Dashboard")
st.markdown("Forecast water quality parameters from **January 2023 to December 2034** using Prophet")

# Feature selection
features = list(feature_units.keys())
selected_feature = st.selectbox("🌊 Select Water Feature to Forecast", features)

# Date range selection dropdowns
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_to_num = {name: idx for idx, name in enumerate(months, 1)}

col1, col2 = st.columns(2)
with col1:
    start_month = st.selectbox("📅 Start Month", months, index=0)
    start_year = st.selectbox("📅 Start Year", list(range(2023, 2035)), index=0)

with col2:
    end_month = st.selectbox("📅 End Month", months, index=11)
    end_year = st.selectbox("📅 End Year", list(range(2023, 2035)), index=11)

# Run forecast
if st.button("🚀 Run Forecast"):
    try:
        start_date = f"{start_year}-{month_to_num[start_month]:02d}-01"
        end_date = f"{end_year}-{month_to_num[end_month]:02d}-01"

        if pd.to_datetime(start_date) > pd.to_datetime(end_date):
            st.error("❌ Start date must be earlier than end date.")
        else:
            forecast_output = run_forecast(df, selected_feature, 'Potability', start_date, end_date)
            st.subheader("📋 Forecast Output")
            st.dataframe(
                forecast_output.rename(columns={'ds': 'Date', 'yhat': f'{selected_feature} Forecast'})
                .style.format(subset=[f'{selected_feature} Forecast'], formatter="{:.2f}")
            )
    except Exception as e:
        st.error(f"⚠️ Error occurred: {e}")
