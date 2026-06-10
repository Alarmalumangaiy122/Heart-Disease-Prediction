import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("heart_model.pkl", "rb"))

# -----------------------------
# Title
# -----------------------------
st.title("Heart Disease Prediction App")

st.write("Enter patient details below:")

# -----------------------------
# Input Fields
# -----------------------------
age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", [0, 1])
cp = st.number_input("Chest Pain Type (cp)", 0, 3, 0)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.number_input("Rest ECG", 0, 2, 1)
thalach = st.number_input("Max Heart Rate", 60, 250, 150)
exang = st.selectbox("Exercise Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
slope = st.number_input("Slope", 0, 2, 1)
ca = st.number_input("CA", 0, 4, 0)
thal = st.number_input("Thal", 0, 3, 2)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):

    input_data = pd.DataFrame([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]], columns=[
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease")
