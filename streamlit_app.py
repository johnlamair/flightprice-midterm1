import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.sidebar.title("Pages:")
page = st.sidebar.radio("Select Page", ["Introduction 📘", "Visualization 📊", "Prediction"])

st.title("✈️ Flight Price Prediction")
df = pd.read_csv("flight-price.csv")

# column names
df.columns = [
    "Serial Number", "Airline", "Flight", "Source City", "Departure Time",
    "Stops", "Arrival Time", "Destination City", "Class", "Duration", "Days Left", "Price"
]

airline_map = {
    "SpiceJet": "SpiceJet",
    "AirAsia": "AirAsia",
    "Vistara": "Vistara",
    "GO FIRST": "GO_FIRST",
    "Indigo": "Indigo",
    "Air India": "Air_India"
}

departure_time_map = {
    "Early Morning": "Early_Morning",
    "Morning": "Morning",
    "Afternoon": "Afternoon",
    "Evening": "Evening",
    "Late Night": "Late_Night"
}
departure_order = list(departure_time_map.values())

if page == "Introduction 📘":
    st.link_button("Github Repo", "https://github.com/johnlamair/flightprice-midterm1")
    st.image("images.jpg", use_container_width=True)

    st.markdown("### 🎯 Objectives")
    st.write(
        "The objective of this analysis is to develop a predictive model that estimates flight prices "
        "based on a range of influencing variables. We aim to identify the relationships that most "
        "significantly affect ticket pricing."
    )

    st.markdown("### 🎯 Dataset")
    st.write(
        "Dataset contains information about flight booking options from the website Easemytrip for "
        "flight travel between India's top 6 metro cities. There are 300261 datapoints and 11 features "
        "in the cleaned dataset."
    )
    st.dataframe(df.head(), hide_index=True)
    st.write("Source: https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction?resource=download")

elif page == "Visualization 📊":
    st.subheader("Data Visualization")
    tab_names = ["Overall"] + list(airline_map.keys())
    tabs = st.tabs(tab_names)

    for tab, name in zip(tabs, tab_names):
        with tab:
            # Filter data for this tab
            if name == "Overall":
                df_tab = df[['Source City', 'Destination City', 'Price', 'Departure Time', 'Airline']].copy()
            else:
                dataset_name = airline_map[name]
                df_tab = df[df['Airline'] == dataset_name][['Source City', 'Destination City', 'Price', 'Departure Time']].copy()

            if df_tab.empty:
                st.write("No data available.")
                continue

            # --- Heatmap ---
            heatmap_data = df_tab.pivot_table(
                index='Source City',
                columns='Destination City',
                values='Price',
                aggfunc='mean'
            )
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
            plt.title(f"Average Price from Source to Destination ({name})", fontsize=14)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            st.pyplot(plt)
            plt.close()

            # --- Bar chart: Average Price by Departure Time ---
            df_tab['Departure Time'] = pd.Categorical(df_tab['Departure Time'], categories=departure_order, ordered=True)
            avg_price_by_departure = df_tab.groupby('Departure Time')['Price'].mean().reindex(departure_order)
            plt.figure(figsize=(10,5))
            plt.bar(avg_price_by_departure.index, avg_price_by_departure.values)
            plt.xticks(ticks=range(len(departure_order)), labels=list(departure_time_map.keys()), rotation=45)
            plt.title(f"Average Price by Departure Time ({name})", fontsize=14)
            plt.xlabel("Departure Time")
            plt.ylabel("Average Ticket Price (₹)")
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            st.pyplot(plt)
            plt.close()

            # --- Box and Whisker Plot (Overall only) ---
            if name == "Overall":
                df_box = df.copy()
                reverse_airline_map = {v: k for k, v in airline_map.items()}
                df_box['Airline'] = df_box['Airline'].map(reverse_airline_map)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x='Airline', y='Price', data=df_box, ax=ax)
                ax.set_title('Distribution of Flight Prices per Airline', fontsize=14)
                ax.set_xlabel('Airline', fontsize=12)
                ax.set_ylabel('Price (₹)', fontsize=12)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close()

elif page == "Prediction":
    st.subheader("Predictions")

    # Make a copy of df for prediction preprocessing
    df_prediction = df.copy()

    # Encode all categorical columns
    categorical_cols = ["Airline", "Flight", "Source City", "Departure Time",
                        "Stops", "Arrival Time", "Destination City", "Class"]
    for col in categorical_cols:
        df_prediction[col] = df_prediction[col].astype("category").cat.codes

    # Create mappings for Airline and Class
    airline_categories = pd.Categorical(df["Airline"])
    airline_mapping = dict(zip(airline_categories.categories, range(len(airline_categories.categories))))

    class_categories = pd.Categorical(df["Class"])
    class_mapping = dict(zip(class_categories.categories, range(len(class_categories.categories))))

    # Features and target
    features = ["Airline", "Flight", "Source City", "Departure Time",
                "Stops", "Arrival Time", "Destination City", "Class",
                "Duration", "Days Left"]
    target = "Price"

    # Tabs for each airline
    tab_names = list(airline_mapping.keys())
    tabs = st.tabs(tab_names)

    for tab, airline_name in zip(tabs, tab_names):
        with tab:
            st.markdown(f"## Airline: {airline_name}")

            airline_code = airline_mapping[airline_name]
            df_airline = df_prediction[df_prediction["Airline"] == airline_code]

            # Loop over classes: Business and Economy
            for cls_name in ["Economy", "Business"]:
                if cls_name not in class_mapping:
                    st.write(f"No data available for {cls_name} class.")
                    continue

                cls_code = class_mapping[cls_name]
                df_class = df_airline[df_airline["Class"] == cls_code]

                if df_class.empty:
                    st.write(f"No data available for {cls_name} class.")
                    continue

                st.markdown(f"### Class: {cls_name}")

                X = df_class[features]
                y = df_class[target]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Linear Regression
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                y_pred = lr.predict(X_test)

                # Metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                st.write(f"**MAE:** ₹{mae:,.2f}  |  **RMSE:** ₹{rmse:,.2f}  |  **R²:** {r2:.4f}")

                # Plot actual vs predicted
                plt.figure(figsize=(8,5))
                plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel("Actual Price (₹)")
                plt.ylabel("Predicted Price (₹)")
                plt.title(f"Actual vs Predicted Prices - {cls_name} Class")
                plt.grid(alpha=0.3)
                st.pyplot(plt)
                plt.close()