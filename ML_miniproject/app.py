import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('../model/crop_model.pkl')

# App Title
st.title("🌾 Crop Recommendation System")

# Input fields
N = st.number_input("Nitrogen (N)", 0, 140)
P = st.number_input("Phosphorus (P)", 5, 145)
K = st.number_input("Potassium (K)", 5, 205)
temperature = st.number_input("Temperature (°C)", 10.0, 45.0)
humidity = st.number_input("Humidity (%)", 10.0, 100.0)
ph = st.number_input("Soil pH", 3.5, 9.5)
rainfall = st.number_input("Rainfall (mm)", 20.0, 300.0)

# Predict button
if st.button("Predict Best Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    st.success(f"✅ Recommended Crop to Grow: **{prediction[0]}**")
