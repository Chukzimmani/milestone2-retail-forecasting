
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

# Load model and data
model = joblib.load("sarimax_model.pkl")
data = pd.read_csv("retail_store_inventory_preprocessed.csv")
data['Date'] = pd.to_datetime(data['Date'])

# Preprocess data for forecasting
daily_df = data.groupby('Date').agg({
    'Units Sold': 'sum',
    'Inventory Level': 'mean',
    'Price': 'mean',
    'Discount': 'mean',
    'Weather Condition': 'mean',
    'Holiday/Promotion': 'mean',
    'Competitor Pricing': 'mean',
    'Seasonality': 'mean'
}).reset_index()

daily_df.set_index('Date', inplace=True)

# Forecasting input
st.title("Retail Demand Forecasting")
st.write("Forecast daily units sold using SARIMAX with exogenous features.")

# Forecast window
n_days = st.slider("Select number of days to forecast:", min_value=7, max_value=60, value=30)

# Prepare input
train = daily_df.iloc[:-n_days]
test = daily_df.iloc[-n_days:]

endog_test = test['Units Sold']
exog_test = test.drop(columns=['Units Sold'])

# Forecast
forecast = model.predict(start=test.index[0], end=test.index[-1], exog=exog_test)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(test.index, endog_test, label='Actual')
ax.plot(test.index, forecast, label='Forecast', linestyle='--')
ax.set_title("Forecast vs Actual")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.legend()
st.pyplot(fig)

# Metrics
st.subheader("Forecast Accuracy (Last {} Days)".format(n_days))
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae = mean_absolute_error(endog_test, forecast)
rmse = np.sqrt(mean_squared_error(endog_test, forecast))
st.write(f"**MAE**: {mae:.2f}")
st.write(f"**RMSE**: {rmse:.2f}")
