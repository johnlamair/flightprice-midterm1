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

    # Map friendly tab names to dataset airline values
    airline_map = {
        "SpiceJet": "SpiceJet",
        "AirAsia": "AirAsia",
        "Vistara": "Vistara",
        "GO FIRST": "GO_FIRST",
        "Indigo": "Indigo",
        "Air India": "Air_India"
    }

    # Map friendly departure time labels to dataset values
    departure_time_map = {
        "Early Morning": "Early_Morning",
        "Morning": "Morning",
        "Afternoon": "Afternoon",
        "Evening": "Evening",
        "Late Night": "Late_Night"
    }
    # List of dataset values in order for sorting
    time_order = list(departure_time_map.values())

    # Create tabs including Overall tab
    tab_names = ["Overall"] + list(airline_map.keys())
    tabs = st.tabs(tab_names)

    for tab, name in zip(tabs, tab_names):
        with tab:
            if name == "Overall":
                st.subheader("Prices for All Airlines")
                df_tab = df[['source_city', 'destination_city', 'price', 'departure_time']].copy()
            else:
                st.subheader(f"Prices for {name}")
                airline_column_value = airline_map[name]
                df_tab = df[df['airline'] == airline_column_value][['source_city', 'destination_city', 'price', 'departure_time']].copy()

            if df_tab.empty:
                st.write("No data available.")
                continue

            # --- Heatmap ---
            heatmap_data = df_tab.pivot_table(
                index='source_city',
                columns='destination_city',
                values='price',
                aggfunc='mean'
            )

            # format price figures
            annot = heatmap_data.applymap(lambda x: f"₹{int(round(x))}" if pd.notnull(x) else "")

            plt.figure(figsize=(12, 8))
            sns.heatmap(
                heatmap_data,
                annot=annot,
                fmt="",
                cmap='coolwarm',  # built-in colormap
                cbar_kws={'label': 'Price (₹)'}
            )
            plt.xlabel("Destination City", fontsize=12)
            plt.ylabel("Source City", fontsize=12)
            plt.title(f"Average Price from Source to Destination ({name})", fontsize=14)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            st.pyplot(plt)
            plt.close()

            # --- Bar chart: Average Price by Departure Time ---
            # Convert departure_time to categorical with specified order
            df_tab['departure_time'] = pd.Categorical(df_tab['departure_time'], categories=time_order, ordered=True)

            # Group and sort by the specified order
            avg_price_by_departure = df_tab.groupby('departure_time')['price'].mean().reindex(time_order)

            plt.figure(figsize=(10,5))
            plt.bar(avg_price_by_departure.index, avg_price_by_departure.values, color='blue')

            # Replace x-axis tick labels with friendly names
            plt.xticks(ticks=range(len(time_order)), labels=list(departure_time_map.keys()), rotation=45)

            plt.title(f"Average Price by Departure Time ({name})", fontsize=14)
            plt.xlabel("Departure Time")
            plt.ylabel("Average Ticket Price (₹)")
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            st.pyplot(plt)
            plt.close()

            # --- Box and Whisker Plot (only for Overall tab) ---
            if name == "Overall":
                st.subheader("Box and Whisker Plot: Flight Prices by Airline")

                # Map dataset airline values to friendly names
                reverse_airline_map = {v: k for k, v in airline_map.items()}
                df_box = df.copy()
                df_box['Airline'] = df_box['airline'].map(reverse_airline_map)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x='Airline', y='price', data=df_box, palette='Set3', ax=ax)
                ax.set_title('Distribution of Flight Prices per Airline', fontsize=14)
                ax.set_xlabel('Airline', fontsize=12)
                ax.set_ylabel('Price (₹)', fontsize=12)
                plt.xticks(rotation=45)

                st.pyplot(fig)
                plt.close()
            
elif page == "Prediction":
    st.subheader("Prediction")


    
