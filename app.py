
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# Load the trained SARIMAX model and dataset
model = SARIMAXResults.load("sarimax_model.sm")
data = pd.read_csv("retail_store_inventory_preprocessed.csv")
data['Date'] = pd.to_datetime(data['Date'])

# Preprocess the data (grouped by date)
daily_df = data.groupby(['Date', 'Region']).agg({
    'Units Sold': 'sum',
    'Inventory Level': 'mean',
    'Price': 'mean',
    'Discount': 'mean',
    'Weather Condition': 'mean',
    'Holiday/Promotion': 'mean',
    'Competitor Pricing': 'mean',
    'Seasonality': 'mean'
}).reset_index()

st.title("Retail Demand Forecasting with SARIMAX")
st.write("Forecast daily units sold by selecting region and discount level. Confidence intervals included.")

# Sidebar controls
region = st.selectbox("Select Region", sorted(daily_df['Region'].unique()))
discount_input = st.slider("Select Forecast Discount (%)", 0, 50, 10)
n_days = st.slider("Number of days to forecast", 7, 60, 30)

# Filter data based on region
region_df = daily_df[daily_df['Region'] == region].copy()

# Override discount to simulate what-if scenario
region_df['Discount'] = discount_input
region_df.set_index('Date', inplace=True)

# Forecasting function
def forecast_demand(model, df, days):
    test_df = df.iloc[-days:]
    endog = test_df['Units Sold']
    exog = test_df.drop(columns=['Units Sold', 'Region'])

    forecast_result = model.get_forecast(steps=days, exog=exog)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    return endog, forecast_mean, conf_int

# Run forecast
try:
    actual, forecast, interval = forecast_demand(model, region_df, n_days)

    # Plot forecast
    st.subheader("Forecast vs Actual with 95% Confidence Interval")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual.index, actual, label='Actual')
    ax.plot(actual.index, forecast, label='Forecast', linestyle='--')
    ax.fill_between(actual.index, interval.iloc[:, 0], interval.iloc[:, 1], color='gray', alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.set_title("Demand Forecast")
    ax.legend()
    st.pyplot(fig)

    # Metrics
    st.subheader("Prediction Range")
    st.write(f"From **{int(forecast.min())}** to **{int(forecast.max())}** units expected over the next {n_days} days.")

except Exception as e:
    st.error(f"Forecast failed: {str(e)}")
