import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

st.title("💻 Laptop Price Predictor")
st.write("Select laptop features to predict the price")

# -----------------------
# Dropdown Inputs
# -----------------------

company = st.selectbox("Brand", df['Company'].unique())
type_name = st.selectbox("Laptop Type", df['TypeName'].unique())
os = st.selectbox("Operating System", df['OS'].unique())
screen = st.selectbox("Screen Type", df['Screen'].unique())
gpu_company = st.selectbox("GPU Company", df['GPU_company'].unique())
cpu_brand = st.selectbox("CPU Brand", df['CPU_Brand'].unique())

# -----------------------
# Numeric Inputs
# -----------------------

ram = st.selectbox("RAM (GB)", sorted(df['Ram'].unique()))
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)
ppi = st.number_input("PPI", min_value=50.0, max_value=500.0, step=1.0)

touchscreen = st.selectbox("Touchscreen", [0, 1])
ips = st.selectbox("IPS Panel", [0, 1])
retina = st.selectbox("Retina Display", [0, 1])

ssd = st.selectbox("SSD (GB)", sorted(df['SSD'].unique()))
hdd = st.selectbox("HDD (GB)", sorted(df['HDD'].unique()))

# -----------------------
# Prediction
# -----------------------

if st.button("Predict Price 💰"):

    input_df = pd.DataFrame([{
        'Company': company,
        'TypeName': type_name,
        'OS': os,
        'Screen': screen,
        'GPU_company': gpu_company,
        'CPU_Brand': cpu_brand,
        'Ram': ram,
        'Weight': weight,
        'ppi': ppi,
        'Touchscreen': touchscreen,
        'IPSpanel': ips,
        'RetinaDisplay': retina,
        'SSD': ssd,
        'HDD': hdd
    }])

    # Predict log price
    log_price = pipe.predict(input_df)[0]

    # Convert back to actual price
    price = np.exp(log_price)

    st.success(f"💶 Estimated Laptop Price: **€ {int(price)}**")