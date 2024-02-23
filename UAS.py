# UAS.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.markdown("""
KELOMPOK : DUDA PIR'AAUUW 
- ANGGOTA :
- 10122361 - Muhammad. Asfan Sakti
- 10122381 - Kana Dianto
- 10122359 - Rafi Fadhlan Pratama
- 10122473 - Muhamad Kamal
- 10122380 - Muhamad Fardan Zawallu Syamsi
- 10122356 - Natasya Dita Apriliana Arsono
""", unsafe_allow_html=True)

# Load your dataset
data = pd.read_csv('orders_dataset.csv') 

# Tampilkan snapshot data jika diinginkan
if st.sidebar.checkbox('Tampilkan Snapshot Data', False):
    st.write(data.head())

# Preprocessing Data
features = ['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']
data[features] = data[features].apply(pd.to_datetime)  # Konversi kolom timestamp menjadi tipe data datetime

# Sidebar: Pengaturan Analisis
st.sidebar.header('Pengaturan')
n_clusters = st.sidebar.slider('Jumlah Cluster', min_value=2, max_value=10, value=3)

# Clustering dengan KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(data[features])

# Visualisasi Clustering dengan Matplotlib
fig, ax = plt.subplots()
for cluster in data['cluster'].unique():
    ax.scatter(data[data['cluster'] == cluster]['order_approved_at'], data[data['cluster'] == cluster]['order_delivered_customer_date'], label=f'Cluster {cluster}')
ax.set_title('Clustering Hasil Berdasarkan Timestamp')
ax.set_xlabel('order_approved_at')
ax.set_ylabel('order_delivered_customer_date')
st.pyplot(fig)

# PCA Visualisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
data['pca1'] = components[:, 0]
data['pca2'] = components[:, 1]

fig, ax = plt.subplots()
scatter = ax.scatter(data['pca1'], data['pca2'], c=data['cluster'])
legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)
ax.set_title('PCA: Visualisasi Data Multidimensi')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
st.pyplot(fig)

# Korelasi Fitur
if st.sidebar.checkbox('Tampilkan Heatmap Korelasi', False):
    corr_matrix = data[features].corr()
    fig, ax = plt.subplots()
    cax = ax.matshow(corr_matrix, cmap='coolwarm')
    fig.colorbar(cax)
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    ax.set_title('Heatmap Korelasi Fitur')
    st.pyplot(fig)
