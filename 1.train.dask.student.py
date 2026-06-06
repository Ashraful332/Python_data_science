import joblib
import pandas as pd
import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Read CSV lazily
df = dd.read_csv(
    "data/college_student_placement_dataset.csv"
)

# Convert categorical columns
df["Placement"] = df["Placement"].map(
    {
        "No": 0,
        "Yes": 1
    },
    meta=("Placement", "int64")
)

df["Internship_Experience"] = (
    df["Internship_Experience"]
    .map(
        {
            "No": 0,
            "Yes": 1
        },
        meta=("Internship_Experience", "int64")
    )
)

features = [
    "IQ",
    "Prev_Sem_Result",
    "CGPA",
    "Academic_Performance",
    "Internship_Experience",
    "Extra_Curricular_Score",
    "Communication_Skills",
    "Projects_Completed"
]

X = df[features]
Y = df["Placement"]

# Convert to pandas if dataset fits in RAM
X = X.compute()
Y = Y.compute()

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

joblib.dump(model, "train_dask_student_placement_model.pkl")

print("Model saved!")