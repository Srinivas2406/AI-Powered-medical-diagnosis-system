import streamlit as st
import pandas as pd
import joblib
import numpy as np


def set_background():
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("https://newtechnology.news/wp-content/uploads/2024/01/AI-in-Healthcare-Revolutionizing-Diagnosis-and-Treatment-1024x652.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background()  # Apply background

# App title
st.markdown("<h1 style='text-align: center; color: orange;'>AI-Powered Medical Diagnosis</h1>", unsafe_allow_html=True)

# Disease selection below the title
st.markdown("<h2 style='color: orange;'>Choose a Disease</h2>", unsafe_allow_html=True)
disease = st.selectbox("Select Disease", ["Diabetes", "Liver Disease", "Thyroid"], key="disease", label_visibility="collapsed")

# Load models
diabetes_model = joblib.load("models/diabetes_model.pkl")
liver_model = joblib.load("models/liver_model.pkl")
thyroid_model = joblib.load("models/thyroid_model.pkl")


if disease == "Diabetes":
    st.subheader("Diabetes")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20)
    glucose = st.number_input("Glucose", min_value=0, max_value=200)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
    insulin = st.number_input("Insulin", min_value=0, max_value=900)
    bmi = st.number_input("BMI", min_value=0.0, max_value=50.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
    age = st.number_input("Age", min_value=1, max_value=120)


    features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    user_input = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age
    ]], columns=features)


    if st.button("Predict"):
        if any(val == 0 for val in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]):
            st.error("Please enter all values!")
        else:
            prediction = diabetes_model.predict(user_input)[0]
            st.success(f"Prediction: {'Diabetic' if prediction == 1 else 'Not Diabetic'}")

elif disease == "Liver Disease":
    st.subheader("Liver Disease")
    age = st.number_input("Age", min_value=1, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female"])
    total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, max_value=10.0)
    direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, max_value=10.0)
    alkaline_phosphotase = st.number_input("Alkaline Phosphotase", min_value=0, max_value=1000)
    alamine_aminotransferase = st.number_input("Alamine Aminotransferase", min_value=0, max_value=500)
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=0, max_value=500)
    total_proteins = st.number_input("Total Proteins", min_value=0.0, max_value=10.0)
    albumin = st.number_input("Albumin", min_value=0.0, max_value=10.0)
    albumin_globulin_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, max_value=5.0)

    # Convert gender to numeric
    gender = 1 if gender == "Male" else 0


    features = ["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase", 
                "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Proteins", "Albumin", 
                "Albumin_and_Globulin_Ratio"]
    user_input = pd.DataFrame([[
        age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase, 
        alamine_aminotransferase, aspartate_aminotransferase, total_proteins, albumin, albumin_globulin_ratio
    ]], columns=features)


    if st.button("Predict"):
        if any(val == 0 for val in [age, total_bilirubin, direct_bilirubin, alkaline_phosphotase, 
                                    alamine_aminotransferase, aspartate_aminotransferase, 
                                    total_proteins, albumin, albumin_globulin_ratio]):
            st.error("Please enter all values!")
        else:
            prediction = liver_model.predict(user_input)[0]
            st.success(f"Prediction: {'Liver Disease Detected' if prediction == 1 else 'No Liver Disease'}")

elif disease == "Thyroid":
    st.subheader("Thyroid")
    tsh = st.number_input("TSH", min_value=0.0, max_value=100.0)
    t3 = st.number_input("T3", min_value=0.0, max_value=10.0)
    t4 = st.number_input("T4", min_value=0.0, max_value=20.0)
    tt4 = st.number_input("TT4", min_value=0.0, max_value=300.0)
    t4u = st.number_input("T4U", min_value=0.0, max_value=3.0)


    features = ["TSH", "T3", "T4", "TT4", "T4U"]
    user_input = pd.DataFrame([[
        tsh, t3, t4, tt4, t4u
    ]], columns=features)


    if st.button("Predict"):
        if any(val == 0 for val in [tsh, t3, t4, tt4, t4u]):
            st.error("Please enter all values!")
        else:
            prediction = thyroid_model.predict(user_input)[0]
            st.success(f"Prediction: {'Thyroid Disease Detected' if prediction == 1 else 'No Thyroid Disease'}")
