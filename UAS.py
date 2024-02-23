import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt

st.markdown("""
KELOMPOK : DUDA PIR'AAUUW 
ANGGOTA :
10122361 - Muhammad. Asfan Sakti
10122381 - Kana Dianto
10122359 - Rafi Fadhlan Pratama
10122473 - Muhamad Kamal
10122380 - Muhamad Fardan Zawallu Syamsi
10122356 - Natasya Dita Apriliana Arsono
""", unsafe_allow_html=True)

# Load Data
df = pd.read_csv('orders_dataset.csv')  # Replace 'your_dataset.csv' with 'orders_dataset.csv' or the actual path

# Menampilkan data
st.write("Dataframe:")
st.dataframe(df.head())

# Pilih kolom numerik untuk korelasi
numeric_columns = df.select_dtypes(include=[np.number]).columns

# Analisis sederhana
st.write("Statistik Deskriptif:")
st.write(df.describe())

# Visualisasi data
selected_column = st.selectbox("Select Column for Histogram", df.columns)
st.write(f"Histogram for {selected_column}:")
chart = alt.Chart(df).mark_bar().encode(
    x=selected_column,
    y='count()',
    tooltip=[selected_column, 'count()']
).interactive()

st.altair_chart(chart, use_container_width=True)

# Visualisasi korelasi heatmap
st.write("Correlation Heatmap:")
correlation_matrix = df[numeric_columns].corr()

# Display the correlation matrix using Altair
heatmap = alt.Chart(correlation_matrix.reset_index().melt(id_vars='index')).mark_rect().encode(
    x='index:N',
    y='variable:N',
    color='value:Q'
).properties(
    width=500,
    height=500
)

st.altair_chart(heatmap, use_container_width=True)
