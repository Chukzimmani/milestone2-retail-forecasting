
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model
model = joblib.load("linear_regression_model.pkl")

# Title
st.title("Retail Demand Prediction using Linear Regression")
st.write("Enter details below to predict expected demand.")

# User inputs
region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
category = st.selectbox("Product Category", ["Clothing", "Electronics", "Home", "Grocery", "Sports"])
inventory_level = st.number_input("Inventory Level", min_value=0.0, value=500.0)
units_ordered = st.number_input("Units Ordered", min_value=0.0, value=500.0)
demand_forecast = st.number_input("Demand Forecast", min_value=0.0, value=500.0)
price = st.number_input("Price", min_value=0.0, value=10.0)
discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0)
competitor_pricing = st.number_input("Competitor Pricing", min_value=0.0, value=9.0)
weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Snowy", "Cloudy"])
holiday_promo = st.selectbox("Holiday or Promotion?", [0, 1])
seasonality = st.selectbox("Seasonality", ["High", "Medium", "Low"])
month = st.slider("Month", 1, 12, 5)
weekday = st.slider("Weekday (0=Mon, 6=Sun)", 0, 6, 0)

# Predict
if st.button("Predict Demand"):
    input_df = pd.DataFrame([{
        "Region": region,
        "Category": category,
        "Inventory Level": inventory_level,
        "Units Ordered": units_ordered,
        "Demand Forecast": demand_forecast,
        "Price": price,
        "Discount": discount,
        "Competitor Pricing": competitor_pricing,
        "Weather Condition": weather,
        "Holiday/Promotion": holiday_promo,
        "Seasonality": seasonality,
        "Month": month,
        "Weekday": weekday
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Units Sold: {prediction:.0f}")
