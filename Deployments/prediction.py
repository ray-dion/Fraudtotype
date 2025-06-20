import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model
best_model = joblib.load('final_model.pkl')

def run():
    # Load dataset
    df = pd.read_csv("Finpro_data_clean.csv")

    # Map state abbreviations to full names
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

    df['state_full'] = df['state'].map(us_state_names)
    state_list = df['state_full'].dropna().unique().tolist()
    state_list.sort()
    category_map = {cat.replace('_', ' ').title(): cat for cat in df['category'].dropna().unique()}
    category_list = sorted(category_map.keys())

    # Title
    st.title("🛡️ Fraud Detection Form")

    with st.form("fraud_form"):
        trans_hour = st.number_input('Jam Transaksi (0-23):', min_value=0, max_value=23, value=0, help = 'Input jam transaksi')
        amt = st.number_input('Amount Transaksi ($):', min_value=0.0, value=0.0, step=1.0, help = 'Input amount transaksi')
        age = st.number_input('Umur Customer:', min_value=0, max_value=120, value=0, help = 'Input umur user')
        category_label = st.selectbox('Kategori Transaksi:', category_list, help = 'Input kategori transaksi')
        state_full = st.selectbox('Lokasi User:', state_list, help = 'Input lokasi user')

        submitted = st.form_submit_button('Predict')

    # Reverse maps
    reverse_state_map = {v: k for k, v in us_state_names.items()}
    state = reverse_state_map.get(state_full)
    category = category_map.get(category_label)

    # Inference DataFrame
    data_inf = pd.DataFrame([{
        'trans_hour': trans_hour,
        'amt': amt,
        'age': age,
        'category': category,
        'state': state
    }])

    st.write("### 🔎 Input Summary")
    st.dataframe(data_inf)

    if submitted:
        prediction_result = best_model.predict(data_inf)[0]
        prediction_proba = best_model.predict_proba(data_inf)[0]
        confidence = prediction_proba[prediction_result] * 100

        if prediction_result == 0:
            # Transaksi Legit (centered green box)
            st.markdown(
                f"""
                <div style="background-color:#e6f4ea;padding:20px;border-radius:10px;text-align:center;">
                    <div style="font-size:24px;font-weight:bold;color:#207744;">✅ Transaksi Legit</div>
                    <div style="font-size:16px;margin-top:6px;">Confidence: {confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Fraud Detected (centered red box)
            st.markdown(
                f"""
                <div style="background-color:#fdecea;padding:20px;border-radius:10px;text-align:center;">
                    <div style="font-size:24px;font-weight:bold;color:#a30000;">🟥 Fraud Terdeteksi!</div>
                    <div style="font-size:16px;margin-top:6px;">Confidence: {confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == '__main__':
    run()