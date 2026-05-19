# ✈️ Flight Price Prediction Dashboard

## About
This project is an interactive Streamlit web app that analyzes and predicts flight ticket prices using real-world booking data. It explores how factors such as airline, route, departure time, class, and days left until departure influence pricing.

The app also trains a machine learning model (Linear Regression) to estimate flight prices based on these features.

---

## Features

### 📘 Introduction Page
- Dataset overview (300K+ records)
- Project objectives
- GitHub repository link
- Basic dataset preview and metadata

### 📊 Visualization Page
- Interactive heatmaps of route pricing
- Average price by departure time
- Airline-wise price distributions
- Correlation heatmap of features
- Filtered views per airline

### 🔮 Prediction Page
- Linear Regression model per airline and class
- Train/test split evaluation
- Performance metrics:
  - MAE
  - RMSE
  - R² score
- Actual vs predicted price plots

### 📌 Conclusion Page
- Insights on pricing trends by city and time
- Observations on airline and class-based pricing behavior

---

## Dataset
- Source: Kaggle Flight Price Prediction Dataset  
- Link: https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction  
- ~300,000 records
- Features include:
  - Airline
  - Source/Destination city
  - Departure/Arrival time
  - Stops
  - Class (Economy/Business)
  - Duration
  - Days left
  - Price

---

## Tech Stack
- Python
- Streamlit
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn

---

## Machine Learning Model
- Model: Linear Regression
- Preprocessing: Label Encoding for categorical variables
- Evaluation metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
