
# 🛍️ Milestone 2 – Retail Demand Forecasting (Clothing Category)

This project was completed as part of the Business Analytics course. It addresses the problem of demand forecasting in retail by applying time series modeling and regression analysis on a cleaned dataset of clothing sales.

---

## 🎯 Business Problem

Retailers often overstock or understock due to poor demand forecasts. This project focuses on improving demand prediction for clothing items to support inventory planning and sales decisions.

---

## 📊 Dataset

- Source: `retail_store_inventory.csv`
- Focus: Clothing category only
- Key columns used:
  - Units Sold
  - Inventory Level
  - Price
  - Discount
  - Holiday/Promotion
  - Weather Condition
  - Seasonality
  - Competitor Pricing

---

## 📈 Models Used

### 1. SARIMAX (Seasonal ARIMA with eXogenous variables)
- Tuned for time series forecasting using external factors
- Performance:
  - MAE: ~672.6
  - RMSE: ~776.5
  - R²: -0.1173 (on volatile recent data)

### 2. Linear Regression
- Used to explain what drives sales
- Performance:
  - R²: 0.25
  - MAE: ~16.4
  - RMSE: ~20.7

### 3. Simple Regression (Inventory Only)
- R²: 0.40
- Shows Inventory Level is the most important single driver of sales

---

## 📂 Repository Structure

```
retail-forecasting/
├── data/
│   └── retail_store_inventory.csv
├── notebooks/
│   └── final_forecasting_analysis_notebook.ipynb
├── outputs/
│   ├── sarimax_vs_original_forecast_chart.png
│   ├── feature_importance_rf.png
│   ├── inventory_vs_sales_scatter.png
│   └── actual_vs_predicted_lr.png
├── milestone2_presentation.pptx
├── README.md
```

---

## 💡 Business Value

- Improved demand forecasts = better inventory decisions
- Reduced stockouts and overstocking
- Clear understanding of what drives clothing sales
- Actionable recommendations for both operations and strategy teams

---

## 🧪 How to Run

1. Open `final_forecasting_analysis_notebook.ipynb`
2. Run all cells in sequence (Python 3.10+, Jupyter environment)
3. Review the outputs and plots

---

## 📌 Recommendation

- Use SARIMAX for rolling 30-day forecasts
- Use regression insights to guide discounting and inventory strategies

