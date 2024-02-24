import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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

# Load data
url = "orders_dataset.csv"
df = pd.read_csv(url)

# Tampilkan snapshot data
st.title("Snapshot Data")
st.write(df.head())

# Preprocessing Data
st.title("Preprocessing Data")
# Konversi kolom waktu menjadi tipe datetime
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
df['order_delivered_carrier_date'] = pd.to_datetime(df['order_delivered_carrier_date'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
st.write(df.dtypes)

# Visualisasi Waktu Pengiriman
st.title("Visualisasi Waktu Pengiriman")
# Hitung waktu pengiriman aktual
df['actual_delivery_time'] = df['order_delivered_customer_date'] - df['order_purchase_timestamp']

# Visualisasi distribusi waktu pengiriman
fig, ax = plt.subplots()
sns.histplot(df['actual_delivery_time'].dt.days, bins=20, kde=True, ax=ax)
plt.xlabel("Waktu Pengiriman (hari)")
plt.ylabel("Jumlah Pesanan")
plt.title("Distribusi Waktu Pengiriman")
st.pyplot(fig)

# Korelasi Fitur
st.title("Korelasi Fitur")
# Pastikan kedua kolom memiliki tipe data yang sesuai
df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
df['actual_delivery_time'] = pd.to_timedelta(df['actual_delivery_time']).dt.total_seconds() / (60 * 60 * 24)

# Visualisasi korelasi antara waktu persetujuan dan waktu pengiriman
fig, ax = plt.subplots()
sns.scatterplot(x='order_approved_at', y='actual_delivery_time', data=df, ax=ax)
plt.xlabel("Waktu Persetujuan Pesanan")
plt.ylabel("Waktu Pengiriman (hari)")
plt.title("Korelasi antara Waktu Persetujuan dan Waktu Pengiriman")
st.pyplot(fig)
