'''Documentation

- Columns Description :
    College_ID	            Unique ID of the college (e.g., CLG0001 to CLG0100)
    IQ	                    Student's IQ score (normally distributed around 100)
    Prev_Sem_Result	        GPA from the previous semester (range: 5.0 to 10.0)
    CGPA	                Cumulative Grade Point Average (range: ~5.0 to 10.0)
    Academic_Performance	Annual academic rating (scale: 1 to 10)
    Internship_Experience	Whether the student has completed any internship (Yes/No)
    Extra_Curricular_Score	Involvement in extracurriculars (score from 0 to 10)
    Communication_Skills	Soft skill rating (scale: 1 to 10)
    Projects_Completed	    Number of academic/technical projects completed (0 to 5)
    Placement	            Final placement result (Yes = Placed, No = Not Placed)

- Work :
    make a Predictive modeling of placement outcomes
    my model get student info and predict there placement
    model must be effecent and have GUI and batter user experince 

'''# ---- Start Project ----

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# =========================
# LOAD CSV
# =========================

df = pd.read_csv("data/college_student_placement_dataset.csv")

def Graph(df):
    # CGPA/Placement Graph
    plt.title("CGPA/Placement Graph Example")
    plt.scatter(df['CGPA'],df['Placement'])
    plt.xlabel("CGPA")
    plt.ylabel("Placement")
    plt.savefig(
        "data/graph/graph.cgpa-placement.2.png",
        dpi=300
    )


# Graph(df)


# =========================
# TRAIN MODEL
# =========================

# convert Placement Yes/No into 1/0
df["Placement"] = df["Placement"].map({
    "No": 0,
    "Yes": 1
})

# convert Internship Experience Yes/No into 1/0
df["Internship_Experience"] = df["Internship_Experience"].map({
    "No": 0,
    "Yes": 1
})

X = df[[
    "IQ",
    "Prev_Sem_Result",
    "CGPA",
    "Academic_Performance",
    "Internship_Experience",
    "Extra_Curricular_Score",
    "Communication_Skills",
    "Projects_Completed",
]]


Y = df["Placement"]


# Split Data
X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,test_size=0.2,random_state=42
)

# Set Model
model = LogisticRegression(max_iter=5000)

# Train Model
model.fit(X_train, Y_train)

print("Training Complete!")

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "student_placement_model.pkl")

print("Model saved!")

