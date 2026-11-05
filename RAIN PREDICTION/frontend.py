
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Rain Prediction System",
    page_icon="🌧️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #F5F7FB;
}
.title {
    text-align: center;
    color: #4B4E83;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #666666;
    font-size: 18px;
}
.stButton>button {
    background-color: #7779A8;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Load trained model and scaler
model = joblib.load("C:\\Users\\aisha\\AVSCODE\\RAIN PREDICTION\\rain_prediction_model.pkl")
scaler = joblib.load("C:\\Users\\aisha\\AVSCODE\\RAIN PREDICTION\\scaler.pkl")

# Header
st.markdown(
    '<p class="title">🌧️ Rain Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict whether it will rain tomorrow using Artificial Neural Network</p>',
    unsafe_allow_html=True
)

st.divider()

st.subheader("🌤️ Enter Weather Information")

# Input fields
col1, col2, col3 = st.columns(3)

with col1:
    location = st.selectbox(
        "Location",
        ["Sydney", "Melbourne", "Brisbane", "Perth",
         "Adelaide", "Canberra", "Darwin", "Hobart"]
    )

    min_temp = st.number_input("Minimum Temperature (°C)", value=15.0)
    max_temp = st.number_input("Maximum Temperature (°C)", value=25.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0)
    evaporation = st.number_input("Evaporation (mm)", min_value=0.0, value=5.0)

with col2:
    sunshine = st.number_input("Sunshine (hours)", min_value=0.0, max_value=24.0, value=7.0)
    wind_gust_speed = st.number_input("Wind Gust Speed", min_value=0.0, value=35.0)
    wind_speed_9am = st.number_input("Wind Speed 9am", min_value=0.0, value=10.0)
    wind_speed_3pm = st.number_input("Wind Speed 3pm", min_value=0.0, value=15.0)
    humidity_9am = st.number_input("Humidity 9am (%)", min_value=0.0, max_value=100.0, value=70.0)

with col3:
    humidity_3pm = st.number_input("Humidity 3pm (%)", min_value=0.0, max_value=100.0, value=50.0)
    pressure_9am = st.number_input("Pressure 9am (hPa)", value=1015.0)
    pressure_3pm = st.number_input("Pressure 3pm (hPa)", value=1012.0)
    cloud_9am = st.number_input("Cloud 9am", min_value=0.0, max_value=9.0, value=4.0)
    cloud_3pm = st.number_input("Cloud 3pm", min_value=0.0, max_value=9.0, value=5.0)

temp_9am = st.number_input("Temperature 9am (°C)", value=20.0)
temp_3pm = st.number_input("Temperature 3pm (°C)", value=24.0)

# Date inputs
date = st.date_input("Select Date")

st.divider()

# Prediction button
if st.button("🔍 Predict Rain Tomorrow"):

    st.warning(
        "This example input form must be connected to the exact "
        "preprocessing pipeline used during model training."
    )

    # Prepare features in the same order used during training
    month = date.month
    day = date.day

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    day_sin = np.sin(2 * np.pi * day / 31)
    day_cos = np.cos(2 * np.pi * day / 31)

    year = date.year

    # IMPORTANT:
    # Replace this example with the exact 26 training features,
    # including encoded categorical columns.
    input_data = pd.DataFrame([[
        min_temp, max_temp, rainfall, evaporation,
        sunshine, wind_gust_speed, wind_speed_9am,
        wind_speed_3pm, humidity_9am, humidity_3pm,
        pressure_9am, pressure_3pm, cloud_9am,
        cloud_3pm, temp_9am, temp_3pm,
        year, month_sin, month_cos, day_sin, day_cos,
        0, 0, 0, 0, 0
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0][0] > 0.5:
        st.error("🌧️ Rain is predicted tomorrow!")
        st.subheader("Prediction: YES")
    else:
        st.success("☀️ No rain is predicted tomorrow!")
        st.subheader("Prediction: NO")

    st.write(
        f"Rain probability: {prediction[0][0] * 100:.2f}%"
    )

st.divider()

st.markdown(
    "<center>Developed using Python, Streamlit and Artificial Neural Network</center>",
    unsafe_allow_html=True
)
