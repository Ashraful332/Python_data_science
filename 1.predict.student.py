import joblib
import pandas as pd
import streamlit as st

# Load model
model = joblib.load("student_placement_model.pkl")

st.title("🎓 Student Placement Predictor")

# Input fields
iq = st.number_input("IQ", min_value=0, max_value=200, value=107)
prev_sem = st.number_input("Previous Semester Result", min_value=0.0, max_value=10.0, value=6.61)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=6.28)
academic = st.slider("Academic Performance", 0, 10, 8)

internship = st.selectbox(
    "Internship Experience",
    ["No", "Yes"]
)

extra = st.slider("Extra Curricular Score", 0, 10, 8)
communication = st.slider("Communication Skills", 0, 10, 8)
projects = st.number_input("Projects Completed", min_value=0, value=4)

# Convert Yes/No to 1/0
internship = 1 if internship == "Yes" else 0

if st.button("Predict Placement"):
    student = pd.DataFrame([{
        "IQ": iq,
        "Prev_Sem_Result": prev_sem,
        "CGPA": cgpa,
        "Academic_Performance": academic,
        "Internship_Experience": internship,
        "Extra_Curricular_Score": extra,
        "Communication_Skills": communication,
        "Projects_Completed": projects
    }])

    result = model.predict(student)

    if result[0] == 1:
        st.success("✅ Placement Likely")
    else:
        st.error("❌ Placement Unlikely")


# ---- old code ----

# import joblib
# import streamlit as st

# # Load Save Model
# model = joblib.load("student_placement_model.pkl")

# print("Model loaded!")

# # Predict new student placement
# result = model.predict([[
#     107,
#     6.61,
#     6.28,
#     8,
#     0,
#     8,
#     8,
#     4,
# ]])

# print("Prediction:", result)

