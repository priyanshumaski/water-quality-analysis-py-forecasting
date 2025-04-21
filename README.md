# 💧 Water Quality Forecasting App using Streamlit & Prophet

This project is a **web-based interactive dashboard** built with **Streamlit** and powered by **Facebook Prophet**. It enables forecasting of key water quality features such as `pH`, `Hardness`, `Solids`, etc., using historical data.

The model also incorporates `Potability` (drinkability) as an external regressor for enhanced predictive power.

---

## 📊 What It Does

This tool provides **time-series forecasting** for multiple water quality metrics, with two major modes:

- **📆 Monthly Forecast**: Predicts the values for the next 12 months
- **📅 Yearly Forecast**: Predicts values over the next 3 years (36 months)

The forecast is performed using Facebook’s **Prophet** model with added regressors and custom seasonality components.

---

## 📁 Dataset

The dataset used is:

**`water_potability - 1.csv`**

It contains columns like:

| Year | Month | ph | Hardness | Solids | Chloramines | Conductivity | Organic_carbon | Trihalomethanes | Turbidity | Potability |
|------|-------|----|----------|--------|-------------|--------------|----------------|------------------|-----------|-------------|

Each row represents **monthly aggregated water quality metrics** for a given year.

---

## 🧠 Forecasting Logic

The model forecasts each feature (`ph`, `Solids`, etc.) using:

- 📌 `Mean`, `Median`, and `Standard Deviation` of the target feature grouped monthly
- 📎 `Potability` used as an external regressor
- 🌀 Custom monthly seasonality
- 📉 Prophet’s built-in yearly seasonality

### Sample Model Flow (for each feature):
```python
- Create datetime from Year + Month
- Group by Date and calculate mean, std, median
- Prepare data for Prophet
- Add Potability as regressor
- Forecast future values
- Visualize next 12 (monthly) or 36 (yearly) data points
# water-quality-analysis
# water-quality-analysis
