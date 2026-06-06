import joblib
import streamlit as st

# Load Save Model
model = joblib.load("student_placement_model.pkl")

print("Model loaded!")

# Predict new student placement
result = model.predict([[
    107,
    6.61,
    6.28,
    8,
    0,
    8,
    8,
    4,
]])

print("Prediction:", result)