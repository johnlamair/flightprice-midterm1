import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.title("Pages:")
page = st.sidebar.radio("Select Page",["Introduction 📘","Visualization 📊", "Automated Report 📑","Prediction"])


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

    # HEATMAP - price & destination city / source city
    df2 = df[['source_city', 'destination_city', 'price']].copy()
    heatmap_data = df2.pivot_table(index='source_city', 
                               columns='destination_city', 
                               values='price', 
                               aggfunc='mean')
    
    # Function to format the annotations as ₹ whole numbers
    annot = heatmap_data.applymap(lambda x: f"₹{int(round(x))}" if pd.notnull(x) else "")

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=annot, fmt="", cmap="coolwarm", cbar_kws={'label': 'Price (₹)'})
    plt.xlabel("Destination City", fontsize=12)
    plt.ylabel("Source City", fontsize=12)
    plt.title("Average Price from Source to Destination", fontsize=14)
    st.pyplot(plt)

    # Create the boxplot
    fig, ax = plt.subplots()
    sns.boxplot(x="stops", y="duration", data=df, ax=ax)
    # Add labels and title
    ax.set_title("Flight Duration by Number of Stops")
    ax.set_ylabel("Flight Duration (Hours)")
    # Display in Streamlit
    st.pyplot(fig)



elif page == "Prediction":

    st.subheader("Prediction")


    
