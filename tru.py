import pymongo
import pandas as pd
from bson import ObjectId
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import pydeck as pdk
import math
from datetime import datetime 

st.header('3D Visualization of Location Clusters')

MONGO_URL = "mongodb+srv://upendrataral21:safety123@cluster0.x0pyhch.mongodb.net/avert?retryWrites=true&w=majority"
COLLECTION_NAME = "avert_data"
connection = pymongo.MongoClient(MONGO_URL)

database = connection.get_database()
collection = database[COLLECTION_NAME]
st.write("Connected to MongoDB Atlas")


#database connection done=======================

#collecting and processing data to fetch ........
all_data = collection.find()
lat=[]
long=[]
sos=[]
for data in all_data:
    lat.append(data["latitude"])
    long.append(data["longitude"])
    sos.append(data["sos"])
#live sos active.....
daf=pd.DataFrame({"lat":lat,"long":long,"sos":sos})
Xs=np.array(daf[daf["sos"]=="true"][["lat","long"]])
if st.button("Live S.O.S"):
    layer = pdk.Layer(
    "ScatterplotLayer",
    daf,
    pickable=True,
    opacity=0.8,
    stroked=True,
    filled=True,
    radius_scale=300,
    radius_min_pixels=5,
    radius_max_pixels=15,
    line_width_min_pixels=1,
    get_position=['long', 'lat'],
    #get_radius="exits_radius",
    get_fill_color=[200, 140, 120],
    get_line_color=[0, 0, 0],)
    #r.to_html("scatterplot_layer.html")
    view_state = pdk.ViewState(longitude=78.348516,
    latitude=22.824289, zoom=10, bearing=0, pitch=0)
    r = pdk.Deck(layers=[layer],initial_view_state=view_state)
    st.write(r)
    



# cls_no=5  ## no. of clusters
# kmeans=KMeans(n_clusters=cls_no)
# kmeans.fit(X)

# centroids = kmeans.cluster_centers_
# labels = kmeans.labels_
# cluster_counts = np.zeros(cls_no, dtype=int)

# for label in labels:
#     cluster_counts[label] += 1

# cluster_info = []

# for cluster_label, count, centroid in zip(range(cls_no), cluster_counts, centroids):
#     cluster_info.append([count,centroid.tolist()[1],centroid.tolist()[0]])

# cl_inf=pd.DataFrame(cluster_info)  #converted data into datframe



# xt = datetime.now()
# st.subheader("Data Updated on : ",xt)
# #####add feature to --like count ,date etc


# layer = pdk.Layer(
#     "ScatterplotLayer",
#     cl_inf,
#     pickable=True,
#     opacity=0.8,
#     stroked=True,
#     filled=True,
#     radius_scale=300,
#     radius_min_pixels=5,
#     radius_max_pixels=15,
#     line_width_min_pixels=1,
#     get_position=[['1','2']],
#     #get_radius="exits_radius",
#     get_fill_color=[200, 140, 120],
#     get_line_color=[0, 0, 0],
    
# )


# tooltip = {
#     "html": "<b>count: ,{0}</b>",
#     "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
# }

# view_state = pdk.ViewState(longitude=78.348516,
#     latitude=22.824289, zoom=10, bearing=0, pitch=0)

# r = pdk.Deck(layers=[layer],tooltip=tooltip,initial_view_state=view_state)
# #r.to_html("scatterplot_layer.html")
# st.write(r)
