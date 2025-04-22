import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# ✅ Must be first Streamlit command
st.set_page_config(page_title="💧 Water Quality Forecast", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    return df

df = load_data()

# Forecast function
def run_prophet_forecast(df, feature, extra_col, forecast_type='monthly'):
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    monthly_df = df.groupby('Date').agg({feature: ['mean', 'std', 'median'], extra_col: 'mean'}).reset_index()
    monthly_df.columns = ['Date', f'{feature}_mean', f'{feature}_std', f'{feature}_median', extra_col.lower()]
    monthly_df = monthly_df.rename(columns={'Date': 'ds', f'{feature}_mean': 'y', extra_col.lower(): extra_col.lower()})
    monthly_df = monthly_df.dropna()

    model = Prophet(yearly_seasonality=True)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_regressor(extra_col.lower())
    model.fit(monthly_df)

    future_periods = 12 if forecast_type == 'monthly' else 36
    future = model.make_future_dataframe(periods=future_periods, freq='MS')

    last_value = monthly_df[extra_col.lower()].iloc[-1]
    future[extra_col.lower()] = last_value

    forecast = model.predict(future)

    if forecast_type == 'monthly':
        forecast_12 = forecast.tail(12)
        forecast_12['Month'] = forecast_12['ds'].dt.strftime('%b')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(forecast_12['Month'], forecast_12['yhat'], marker='o', color='teal', label='Forecast')
        for i, value in enumerate(forecast_12['yhat']):
            ax.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        ax.set_title(f"Monthly Forecast of {feature} using {extra_col}")
        ax.set_xlabel("Month")
        ax.set_ylabel(feature)
        ax.grid(True)
        st.pyplot(fig)
        return forecast_12[['Month', 'yhat']]

    elif forecast_type == 'yearly':
        fig = model.plot(forecast)
        plt.title(f"Yearly Forecast of {feature} using {extra_col}")
        plt.xlabel("Date")
        plt.ylabel(feature)
        plt.grid(True)
        st.pyplot(fig)
        return forecast[['ds', 'yhat']].tail(36)

# Streamlit UI
st.title("💧 Water Quality Forecasting Dashboard")

features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Conductivity',
            'Organic_carbon', 'Trihalomethanes', 'Turbidity']

selected_feature = st.selectbox("Select Water Feature to Forecast", features)
forecast_type = st.radio("Forecast Type", ['monthly', 'yearly'])

if st.button("Run Forecast"):
    st.info(f"Running {forecast_type} forecast for: {selected_feature}")
    forecast_output = run_prophet_forecast(df, selected_feature, 'Potability', forecast_type)
    st.subheader("Forecast Output")
    st.dataframe(forecast_output)
