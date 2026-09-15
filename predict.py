import joblib
import pandas as pd

# Same function as in main.py
def featureEngineering(data):
    e = data.copy()

    if "StudyTimeWeekly" in e.columns and "Absences" in e.columns:
        e["Effort_Ratio"] = e["StudyTimeWeekly"] / (e["Absences"] + 1)

    support_cols = ["Tutoring", "ParentalSupport"]
    if all(col in e.columns for col in support_cols):
        e["Total_Support"] = e[support_cols].sum(axis=1)

    activities = ["Sports", "Music", "Volunteering", "Extracurricular"]
    if all(col in e.columns for col in activities):
        e["Total_Activities"] = e[activities].sum(axis=1)

    if "Absences" in e.columns:
        e["Absences_Sq"] = e["Absences"] ** 2

    return e


# Load best model
model = joblib.load("Misc/bestGPAModel.joblib")

# Input student values to predict with
studentData= pd.DataFrame(
    [
        {
            "Age": 17,
            "Gender": 1,
            "Ethnicity": 0,
            "ParentalEducation": 2,
            "StudyTimeWeekly": 19.83,
            "Absences": 7,
            "Tutoring": 1,
            "ParentalSupport": 2,
            "Extracurricular": 0,
            "Sports": 0,
            "Music": 1,
            "Volunteering": 0,
        }
    ]
)


processed = featureEngineering(studentData)

# Predict their GPA
predictedGPA = model.predict(processed)
print("\n")
print(f"Predicted GPA: {predictedGPA[0]:.2f}")
print("\n")