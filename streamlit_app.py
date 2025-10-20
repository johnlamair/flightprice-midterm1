import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.title("Pages:")
page = st.sidebar.radio("Select Page",["Introduction 📘","Visualization 📊", "Prediction"])


st.title("✈️ Flight Price Prediction")
df = pd.read_csv("flight-price.csv")


if page == "Introduction 📘":
    st.link_button("Github Repo", "https://github.com/johnlamair/flightprice-midterm1")
    st.image("images.jpg", use_container_width=True)

    st.markdown("### 🎯 Objectives")
    st.write(
        "The objective of this analysis is to develop a predictive model that estimates flight prices based on a range of influencing variables. " \
        "We aim to identify the relationships that most significantly affect ticket pricing."
    )

    st.markdown("### 🎯 Dataset")
    st.write(
        "Dataset contains information about flight booking options from the website Easemytrip for flight travel between India's top 6 metro cities. There are 300261 datapoints and 11 features in the cleaned dataset."
    )

    df.columns = ["Serial Number", "Airline", "Flight", "Source City", "Departure Time", "Stops", "Arrival Time", "Destination City", "Class", "Duration", "Days Left", "Price"]
    st.dataframe(df.head(), hide_index=True)

    st.write(
        "Source: https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction?resource=download"
    )

elif page == "Visualization 📊":

    st.subheader("Data Visualization")

    airlines = ["SpiceJet", "AirAsia", "Vistara", "GO FIRST", "Indigo", "Air India"]

    # Create tabs
    tabs = st.tabs(airlines)

    for tab, airline in zip(tabs, airlines):
        with tab:
            st.subheader(f"Prices for {airline}")

            # Filter dataframe for selected airline
            df_airline = df[df['airline'] == airline][['source_city', 'destination_city', 'price']].copy()

            if df_airline.empty:
                st.write("No data available for this airline.")
            else:
                # Pivot table
                heatmap_data = df_airline.pivot_table(
                    index='source_city',
                    columns='destination_city',
                    values='price',
                    aggfunc='mean'
                )

                # Annotate as rupees whole numbers
                annot = heatmap_data.applymap(lambda x: f"₹{int(round(x))}" if pd.notnull(x) else "")

                plt.figure(figsize=(12, 8))
                sns.heatmap(
                    heatmap_data,
                    annot=annot,
                    fmt="",
                    cmap='coolwarm',
                    cbar_kws={'label': 'Price (₹)'}
                )
                plt.xlabel("Destination City", fontsize=12)
                plt.ylabel("Source City", fontsize=12)
                plt.title(f"Average Price from Source to Destination ({airline})", fontsize=14)
                st.pyplot(plt)



elif page == "Prediction":

    st.subheader("Prediction")


    
