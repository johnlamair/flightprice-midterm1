import streamlit as st
import pandas as pd

st.title("✈️ Flight Price Prediction")
st.link_button("Github Repo", "https://github.com/johnlamair/flightprice-midterm1")

st.markdown("### 🎯 Objectives")
st.write(
    "The objective of this analysis is to develop a predictive model that estimates flight prices based on a range of influencing variables. " \
    "We aim to identify the relationships that most significantly affect ticket pricing."
)

st.markdown("### 🎯 Dataset")
st.write(
    "Dataset contains information about flight booking options from the website Easemytrip for flight travel between India's top 6 metro cities. There are 300261 datapoints and 11 features in the cleaned dataset."
)

df = pd.read_csv("flight-price.csv")
st.dataframe(df.head())

st.write(
    "Source: https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction?resource=download"
)
