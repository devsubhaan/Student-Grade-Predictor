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

""" 
Ethnicity
0: Caucasian
1: African American
2: Asian
3: Other

Gender
0: Male
1: Female

Parental Education
0: None
1: High School
2: Some College
3: Bachelor's
4: Higher

Parental Support
0: None
1: Low
2: Moderate
3: High
4: Very High

StudyTimeWeekly
1-20: Hours a week

Others
1: Yes
0: No

"""

studentData= pd.DataFrame(
    [
        {
            "Age": 18,
            "Gender": 1,
            "Ethnicity": 0,
            "ParentalEducation": 2,
            "StudyTimeWeekly": 20,
            "Absences": 0,
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