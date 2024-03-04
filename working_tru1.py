import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import pydeck as pdk
from datetime import datetime

# MongoDB connection
MONGO_URL = "mongodb+srv://upendrataral21:safety123@cluster0.x0pyhch.mongodb.net/avert?retryWrites=true&w=majority"
COLLECTION_NAME = "avert_data"
connection = pymongo.MongoClient(MONGO_URL)

database = connection.get_database()
collection = database[COLLECTION_NAME]
st.write("Connected to MongoDB Atlas")

# Fetch data from MongoDB
all_data = collection.find()

# Process data for plotting
data_points = []
sos_counts = {"true": 0, "false": 0}
for data in all_data:
    try:
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        sos = data["sos"]
        data_points.append((latitude, longitude, sos))
        sos_counts[sos] += 1
    except KeyError:
        st.warning("Latitude, Longitude, or S.O.S key is missing in one or more documents.")

# Debugging: Print data_points and sos_counts
print("Data Points:", data_points)
print("S.O.S Counts:", sos_counts)

# Plotting the data using Pydeck
if data_points:
    df = pd.DataFrame(data_points, columns=["latitude", "longitude", "sos"])
    st.subheader("Map of Data Points")
    st.map(df, color='#33ff33')

    # Cluster the data using KMeans
    cls_no = 5
    kmeans = KMeans(n_clusters=cls_no)
    X = np.array(df[["latitude", "longitude"]])
    kmeans.fit(X)

    # Add a button to update data
    if st.button("Update Data"):
        all_data = collection.find()
        updated_data_points = []
        sos_counts = {"true": 0, "false": 0}
        for data in all_data:
            try:
                latitude = float(data["latitude"])
                longitude = float(data["longitude"])
                sos = data["sos"]
                updated_data_points.append((latitude, longitude, sos))
                sos_counts[sos] += 1
            except KeyError:
                st.warning("Latitude, Longitude, or S.O.S key is missing in one or more documents.")
        if updated_data_points:
            df_updated = pd.DataFrame(updated_data_points, columns=["latitude", "longitude", "sos"])
            st.subheader("Updated Map of Data Points")
            st.map(df_updated)

    # Display the clusters
    st.subheader("Cluster Centers:")
    st.write(kmeans.cluster_centers_)

    # Analytical Panel
    st.sidebar.subheader("Analytical Panel")

    # Bar Chart for S.O.S counts
    st.sidebar.subheader("S.O.S Counts")
    st.sidebar.bar_chart(pd.DataFrame.from_dict(sos_counts, orient="index", columns=["Count"]))

    # Pie Chart for S.O.S distribution
    st.sidebar.subheader("S.O.S Distribution")
    fig, ax = plt.subplots()
    ax.pie(sos_counts.values(), labels=sos_counts.keys(), autopct='%1.1f%%')
    st.sidebar.pyplot(fig)

else:
    st.warning("No valid data points found.")
