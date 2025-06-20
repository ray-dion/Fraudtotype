# app.py
import streamlit as st
import eda
import prediction
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Logo
st.sidebar.image('logo.png', use_container_width=True)

# CSS
st.markdown("""
    <style>
        .stButton > button {
            background-color: #5B4E65; 
            color: #FFF !important;
            border: none;
            padding: 10px 24px;
            font-size: 16px;
            cursor: pointer;
            border-radius: 5px;
            width: 100%;
            transition: background-color 0.3s ease, color 0.3s ease;

        }
        .stButton > button:hover {
            background-color: #4C1973;
        }

        .stRadio > button[aria-checked="true"] {
            background-color: #4C1973 !important;
            color: #FFF !important;
        }

        .stRadio > button[aria-checked="false"] {
            background-color: #5B4E65;
        }
            
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Pilih Menu")

# Session state
if 'selected_menu' not in st.session_state:
    st.session_state.selected_menu = "Home"

# Buttons
home_button = st.sidebar.button("Home")
eda_button = st.sidebar.button("Exploratory Data Analysis")
prediction_button = st.sidebar.button("Prediction")

if home_button:
    st.session_state.selected_menu = "Home"
elif eda_button:
    st.session_state.selected_menu = "EDA"
elif prediction_button:
    st.session_state.selected_menu = "Prediction"

# Content
if st.session_state.selected_menu == "Home":
    df = pd.read_csv('Finpro_data_clean.csv')

    st.title("🛡️ Welcome to Fraudtotype")
    st.subheader("Aplikasi pintar pencegah potensi aktivitas penipuan")

    # image
    image = Image.open('fraud-detection.png')
    st.image(image, caption='', use_container_width=True)

    st.markdown('---')

    # Section 1: Project Introduction
    st.markdown("### 📌 Introduction")
    st.write("""
    Fraudtotype adalah aplikasi pintar yang dirancang untuk menganalisa dan memprediksi potensi aktivitas fraud. Dengan kombinasi analisa pola transaksi dengan teknologi machine learning, Fraudtotype membantu bisnis mengidentifikasi perilaku mencurigakan sebelum menyebabkan kerugian yang signifikan.
    """)

    # Section 2: Objective
    st.markdown("### 🎯 Objective")
    st.write("""
    Penipuan dalam transaksi digital semakin meningkat setiap tahunnya, menimbulkan kerugian besar bagi pelaku industri finansial dan e-commerce.

    Banyak sistem keamanan masih bersifat reaktif, bukan preventif, sehingga penipuan baru diketahui setelah kerugian terjadi.

    Menurut laporan KPMG Global Banking Fraud Survey, lebih dari 50% institusi keuangan global mengalami peningkatan signifikan dalam kasus penipuan digital di tahun 2019.
    """)

    # Section 3: Dataset
    column_data = {
        'Column Name': ['Trans_date_trans_time', 'Cc_num', 'Merchant', 'Category', 'Amt', 
                        'First', 'Last', 'Gender', 'Street', 'City', 'State', 'Zip',
                        'Lat', 'Long', 'City_pop', 'Job', 'Dob', 'Trans_num', 'Unix_time',
                        'Merch_lat', 'Merch_long', 'Is_fraud'],
        'Description': [
            'Timestamp of the transaction (date and time)',
            'Unique customer identification number',
            'The merchant involved in the transaction',
            'Transaction type (e.g., personal, childcare)',
            'Transaction amount',
            'User first name',
            'User last name',
            'User gender',
            'User street address',
            'User city of residence',
            'User state of residence',
            'User zip code',
            'Latitude of User location',
            'Longitude of User location',
            'Population of the User city',
            'User job title',
            'User date of birth',
            'Unique transaction identifier',
            'Transaction timestamp (Unix format)',
            'Merchant location (latitude)',
            'Merchant location (longitude)',
            'Fraudulent transaction indicator'
        ]
    }
    df_columns = pd.DataFrame(column_data)
    st.markdown("### 📊 Dataset")
    st.write("Deskripsi setiap kolom dalam dataset:")
    st.dataframe(df_columns)

    # Section 4: Tools and Libraries
    st.markdown("### 🛠️ Tools and Libraries")
    st.write("""
    - Python: Untuk praproses data, pemodelan, dan penyebaran
    - Pandas: Untuk manipulasi dan analisis data
    - Numpy: Untuk operasi numerik dan penanganan array
    - Matplotlib & Seaborn: Untuk visualisasi data
    - Scikit-learn: Pipeline, RandomForestClassifier, LogisticRegression
    - GridSearchCV: Untuk hyperparameter tuning
    - Joblib: Untuk penyimpanan model
    """)

    # Section 4: Modeling Algorithms
    st.markdown("### 🧠 Models Used")
    st.write("""
    Aplikasi dibuat dengan dua model klasifikasi:
    - Logistic Regression
    - Random Forest Classifier
    """)

    # Section 5: Model Performance
    st.markdown("### 🎯 Evaluation Results")
    st.write("""
    Berdasarkann classification reports dan cross-validation:
    - Recall: 98%
    - Hyperparameter tuning using `GridSearchCV`
    - Final model chosen: `Random Forest Classifier with Hyperparameter Tuning`
    """)

elif st.session_state.selected_menu == "EDA":
    eda.run()

elif st.session_state.selected_menu == "Prediction":
    prediction.run()