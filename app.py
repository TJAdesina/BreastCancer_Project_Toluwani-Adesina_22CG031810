import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model/breast_cancer_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("Breast Cancer Prediction System")
st.write("This system predicts whether a tumor is Benign or Malignant. Educational use only.")

# Input fields
radius = st.number_input("Radius Mean")
texture = st.number_input("Texture Mean")
perimeter = st.number_input("Perimeter Mean")
area = st.number_input("Area Mean")
smoothness = st.number_input("Smoothness Mean")

if st.button("Predict"):
    input_data = np.array([[radius, texture, perimeter, area, smoothness]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("Prediction: BENIGN")
    else:
        st.error("Prediction: MALIGNANT")
