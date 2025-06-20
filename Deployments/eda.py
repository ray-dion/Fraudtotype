import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def run():

    # Title
    st.title("🛡️ Exploratory Data Analysis")
    st.write("#### Pola dan karakteristik transaksi yang tergolong sebagai fraud")
    st.markdown('---')

    st.write("### 📊 Data Overview")
    # Checkbox show/hide
    show = st.checkbox("Show Dataset Information", value=False)
    df = pd.read_csv('Finpro_data_clean.csv')

    # Tampilkan hanya jika dicentang
    if show:
        st.dataframe(df)

    # EDA 1: Distribusi Fraud dan Non-Fraud
    st.subheader("🍩 Distribusi Fraud dan Non-Fraud")

    # Hitung distribusi is_fraud
    fraud_counts = df['is_fraud'].value_counts()
    labels = ['Tidak Fraud', 'Fraud']
    colors = ['skyblue', 'salmon']

    # Buat pie chart
    fig, ax = plt.subplots()
    ax.pie(
        fraud_counts,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white'}
    )
    ax.axis('equal')

    # Tampilkan chart
    st.pyplot(fig)

    # EDA 2: Visualisasi Transaksi Fraud
    st.subheader("⏰  Top 5 Jam Paling Rentan terhadap Fraud")

    # Ubah kolom tanggal jadi datetime dan ambil jam
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour

    # Bikin legend
    df['fraud_label'] = df['is_fraud'].map({0: 'Not Fraud', 1: 'Fraud'})

    # Hitung fraud percentage per jam
    fraud_stats = df.groupby('hour')['is_fraud'].agg(['sum', 'count'])
    fraud_stats['fraud_percent'] = (fraud_stats['sum'] / fraud_stats['count']) * 100

    # Ambil 5 jam dengan fraud rate tertinggi
    top_5_hours = fraud_stats.sort_values(by='fraud_percent', ascending=False).head(5).index.tolist()

    # Filter hanya jam tersebut
    filtered_df = df[df['hour'].isin(top_5_hours)]

    # Urutkan berdasarkan fraud_percent tertinggi (descending)
    ordered_hours = fraud_stats.loc[top_5_hours].sort_values(by='fraud_percent', ascending=False).index.tolist()

    # Buat chart
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(x='hour', hue='fraud_label', data=filtered_df, ax=ax, order=ordered_hours)
    ax.set_xlabel('Jam Transaksi')
    ax.set_ylabel('Jumlah Transaksi')
    ax.legend(title='Status fraud')

    # Tampilkan chart
    st.pyplot(fig)

    # EDA 3: Visualisasi Transaksi Fraud Berdasarkan Kategori
    st.subheader("🚨 Top 5 Kategori dengan Jumlah Fraud Terbanyak")

    # Hitung top 5 kategori fraud terbanyak
    top_5 = df[df['is_fraud'] == 1]['category'].value_counts().head(5)

    # Buat chart
    colors = sns.color_palette("Set2", n_colors=5)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(top_5.index, top_5.values, color=colors)
    ax.set_xlabel("Kategori Produk")
    ax.set_ylabel("Jumlah Fraud")
    ax.tick_params(axis='x', rotation=30)
    ax.bar_label(bars, labels=[str(v) for v in top_5.values], padding=3)

    # Bikin legend
    legend_elements = [Patch(facecolor=colors[i], label=top_5.index[i]) for i in range(len(top_5))]
    ax.legend(handles=legend_elements, title='Kategori', loc='upper right')

    # Tampilkan chart
    st.pyplot(fig)

    # EDA 4: Visualisasi Jumlah Fraud Berdasarkan Gender dan Usia
    st.subheader("👥 Jumlah Fraud berdasarkan Gender dan Usia")

    # Convert DOB to datetime
    df['dob'] = pd.to_datetime(df['dob'])
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    latest_date = df['trans_date_trans_time'].max()

    # Hitung umur berdasarkan latest date
    df['age'] = (latest_date - df['dob']).dt.days // 365

    # Buat age group
    bins = [15, 35, 55, 75, 95]
    labels = ['15-35 tahun', '36-55 tahun','56-75 tahun', '76-95 tahun']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

    # Filter data fraud saja
    fraud_data = df[df['is_fraud'] == 1]

    # Group dan hitung jumlah
    fraud_counts = fraud_data.groupby(['age_group', 'gender']).size().reset_index(name='jumlah_fraud')

    # Buat plot
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(
        data=fraud_counts.sort_values(by='jumlah_fraud', ascending=False),
        x='age_group',
        y='jumlah_fraud',
        hue='gender',
        palette='Set2',
        ax=ax
    )
    ax.set_xlabel('Kelompok Usia')
    ax.set_ylabel('Jumlah Kasus Fraud')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    # Tampilkan chart
    st.pyplot(fig)

    # EDA 5: Distribusi Jumlah Transaksi pada Kasus Fraud
    st.subheader("💸 Distribusi Jumlah Transaksi pada Kasus Fraud")

    # Filter data dengan fraud == 1
    fraud_amt = df[df['is_fraud'] == 1]['amt']

    # Buat histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(fraud_amt, bins=30, kde=True, color='red', ax=ax)
    ax.set_xlabel('Jumlah Transaksi (amt)')
    ax.set_ylabel('Frekuensi')

    # Tampilkan chart
    st.pyplot(fig)

    # EDA 5: Distribusi Fraud berdasarkan Negara Bagian
    st.subheader("🗺️ Top 10 States dengan Jumlah Fraud Terbanyak")

    # Mapping kode state ke nama lengkap
    us_state_names = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'DC': 'District of Columbia',
        'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois',
        'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
        'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
        'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
        'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
        'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota',
        'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia',
        'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
    }

    # Hitung top 10 state dengan fraud terbanyak
    top_10_states = df[df['is_fraud'] == 1].groupby('state').size().sort_values(ascending=False).head(10)
    top_10_df = top_10_states.reset_index(name='Jumlah Kasus Fraud')
    top_10_df['State'] = top_10_df['state'].map(us_state_names)

    # Plot barchart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(top_10_df['State'], top_10_df['Jumlah Kasus Fraud'], color='skyblue')
    ax.set_xlabel('State')
    ax.set_ylabel('Number of Fraud Cases')
    ax.tick_params(axis='x', rotation=45)

    # Tampilkan chart
    st.pyplot(fig)

if __name__ == '__main__':
    run()