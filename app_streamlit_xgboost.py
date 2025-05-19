
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.pkl")
    return model

# UI
st.title("Product Demand Predictor")
st.markdown("This tool helps estimate expected **Units Sold** based on pricing and market conditions.")

# Inputs
price = st.number_input("Selling Price", value=30.0)
discount = st.slider("Discount (%)", 0, 100, 10) / 100.0
competitor_price = st.number_input("Competitor Price", value=28.0)
inventory = st.number_input("Inventory Level", value=200)
promotion = st.selectbox("Is there a Holiday or Promotion?", ["No", "Yes"]) == "Yes"
category = st.selectbox("Product Category", ["Electronics", "Groceries", "Toys", "Clothing", "Other"])
region = st.selectbox("Region", ["North", "South", "East", "West"])
seasonality = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])
weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy", "Snowy"])

# Feature engineering
price_after_discount = price * (1 - discount)
is_promo = int(promotion)

# Create input DataFrame
input_data = {
    "Price": price,
    "Discount": discount,
    "Competitor Pricing": competitor_price,
    "Holiday/Promotion": int(promotion),
    "Inventory Level": inventory,
    "Price_After_Discount": price_after_discount,
    "Is_Promo": is_promo
}

# Category encoding
for cat in ["Clothing", "Electronics", "Groceries", "Toys"]:
    input_data[f"Category_{cat}"] = int(cat == category)

# Region encoding
for r in ["East", "North", "South", "West"]:
    if r != "East":  # Assume 'East' is the dropped baseline
        input_data[f"Region_{r}"] = int(r == region)

# Seasonality encoding
for s in ["Autumn", "Spring", "Summer", "Winter"]:
    if s != "Spring":  # Assume 'Spring' is baseline
        input_data[f"Seasonality_{s}"] = int(s == seasonality)

# Weather encoding
for w in ["Cloudy", "Rainy", "Snowy", "Sunny"]:
    if w != "Cloudy":  # Assume 'Cloudy' is baseline
        input_data[f"Weather Condition_{w}"] = int(w == weather)

# Convert to DataFrame
X_input = pd.DataFrame([input_data])

# Load model and predict
model = load_model()
prediction = model.predict(X_input)[0]

st.success(f"🔮 Predicted Units Sold: {prediction:.2f}")
