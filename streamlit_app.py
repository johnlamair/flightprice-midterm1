import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 


st.sidebar.title("Pages:")
selected_page = st.sidebar.radio("Select Page", ["Introduction 📘", "Visualization 📊", "Prediction"])

st.title("✈️ Flight Price Prediction")
flight_data = pd.read_csv("flight-price.csv")

flight_data.columns = [
    "SerialNumber", "Airline", "FlightNumber", "SourceCity", "DepartureTime",
    "Stops", "ArrivalTime", "DestinationCity", "Class", "Duration", "DaysLeft", "Price"
]

departure_time_order = ["Early Morning", "Morning", "Afternoon", "Evening", "Late Night"]

if selected_page == "Introduction 📘":
    st.link_button("Github Repo", "https://github.com/johnlamair/flightprice-midterm1")
    st.image("images.jpg", use_container_width=True)

    st.markdown("### Objectives")
    st.write("Develop a predictive model that estimates flight prices based on variables that affect ticket pricing.")

    st.markdown("### Dataset")
    st.write("Flight booking data from Easemytrip for India's top 6 metro cities. 300,261 datapoints and 11 features.")
    st.dataframe(flight_data.head(), hide_index=True)
    st.write("Source: https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction?resource=download")

elif selected_page == "Visualization 📊":
    st.subheader("Data Visualization")
    
    airline_names = flight_data["Airline"].unique().tolist()
    tabs = st.tabs(["Overall"] + airline_names)

    for tab, airline_name in zip(tabs, ["Overall"] + airline_names):
        with tab:
            if airline_name == "Overall":
                df_filtered = flight_data[["SourceCity", "DestinationCity", "Price", "DepartureTime", "Airline"]]
            else:
                df_filtered = flight_data[flight_data["Airline"] == airline_name][["SourceCity", "DestinationCity", "Price", "DepartureTime"]]

            if df_filtered.empty:
                st.write("No data available.")
                continue

            # heatmap
            heatmap_matrix = df_filtered.pivot_table(index='SourceCity', columns='DestinationCity', values='Price', aggfunc='mean')
            annotations = heatmap_matrix.applymap(lambda x: f"₹{int(round(x))}" if pd.notnull(x) else "")
            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_matrix, annot=annotations, fmt="", cmap='coolwarm', cbar_kws={'label': 'Price (₹)'})
            plt.xlabel("Destination City")
            plt.ylabel("Source City")
            plt.title(f"Average Price from Source to Destination ({airline_name})")
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            st.pyplot(plt)
            plt.close()

            # bar chart by departure time
            df_filtered['DepartureTime'] = pd.Categorical(df_filtered['DepartureTime'], categories=departure_time_order, ordered=True)
            avg_price = df_filtered.groupby('DepartureTime')['Price'].mean().reindex(departure_time_order)
            plt.figure(figsize=(10,5))
            plt.bar(avg_price.index, avg_price.values)
            plt.xticks(rotation=45)
            plt.title(f"Average Price by Departure Time ({airline_name})")
            plt.xlabel("Departure Time")
            plt.ylabel("Average Price (₹)")
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            st.pyplot(plt)
            plt.close()

            # box plot only for Overall
            if airline_name == "Overall":
                plt.figure(figsize=(10,6))
                sns.boxplot(x='Airline', y='Price', data=flight_data)
                plt.title('Distribution of Prices per Airline')
                plt.xlabel('Airline')
                plt.ylabel('Price (₹)')
                plt.xticks(rotation=45)
                st.pyplot(plt)
                plt.close()

                # correlation matrix
                # select all columns except FlightNumber (we don't need it)
                corr_columns = ["Airline", "SourceCity", "DepartureTime", "Stops", "ArrivalTime",
                                "DestinationCity", "Class", "Duration", "DaysLeft", "Price"]

                df_small = flight_data[corr_columns].copy()

                # encode categorical variables
                encoded = LabelEncoder()
                for col in df_small.select_dtypes(include=['object']).columns:
                    df_small[col] = encoded.fit_transform(df_small[col])

                # compute correlation
                corr_matrix = df_small.corr()

                # plot heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
                plt.title("Correlation Between Variables")
                st.pyplot(plt)
                plt.close()

elif selected_page == "Prediction":
    st.subheader("Predictions")

    flight_model_data = flight_data.copy()
    features = ["Airline", "SourceCity", "DepartureTime", "Stops", "ArrivalTime", "DestinationCity", "Class", "Duration", "DaysLeft"]
    target = "Price"

    categorical_features = ["Airline", "SourceCity", "DepartureTime", "Stops", "ArrivalTime", "DestinationCity", "Class"]
    for feature in categorical_features:
        flight_model_data[feature] = flight_model_data[feature].astype("category").cat.codes

    airline_tab_names = flight_data["Airline"].unique().tolist()
    tabs = st.tabs(airline_tab_names)

    for tab, airline_name in zip(tabs, airline_tab_names):
        with tab:
            st.markdown(f"## Airline: {airline_name}")
            df_airline = flight_data[flight_data["Airline"] == airline_name]

            for travel_class in ["Economy", "Business"]:
                df_class = df_airline[df_airline["Class"] == travel_class]
                if df_class.empty:
                    st.write(f"No data available for {travel_class} class.")
                    continue

                st.markdown(f"### Class: {travel_class}")
                df_class_model = flight_model_data.loc[df_class.index]
                X = df_class_model[features]
                y = df_class_model[target]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                st.write(f"**MAE:** ₹{mae:,.2f}  |  **RMSE:** ₹{rmse:,.2f}  |  **R²:** {r2:.4f}")

                plt.figure(figsize=(8,5))
                plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel("Actual Price (₹)")
                plt.ylabel("Predicted Price (₹)")
                plt.title(f"Actual vs Predicted Prices - {travel_class} Class")
                plt.grid(alpha=0.3)
                st.pyplot(plt)
                plt.close()