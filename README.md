# 🎓 Student GPA Prediction Pipeline 

Machine learning pipeline designed to predict student GPAs

Dataset used: https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset

---

## 📌 Project Overview
An automated machine learning pipeline that predicts student GPAs.
* Data Pipeline: Handles feature engineering, scaling, and missing data automatically.
* Model Tuning: Uses Optuna to fine-tune Ridge, CatBoost, and XGBoost.
* Automated Selection: Compares model performance and deploys whichever algorithm achieves the lowest Mean Absolute Error (MAE).
  
---

## 📊 Dataset & Features

The model trains on `performance.csv` and predicts **GPA** on a continuous 0.0–4.0 scale.

### Engineered Features
Features are added which can help the model predict a lot more easier.

* **`Effort_Ratio`**: Study hours relative to absences $\left(\frac{\text{StudyTimeWeekly}}{\text{Absences} + 1}\right)$
* **`Total_Support`**: Combined academic and home support score (`Tutoring` + `ParentalSupport`)
* **`Total_Activities`**: Total extracurricular involvement across sports, music, and volunteering
* **`Absences_Sq`**: Quadratic penalty term reflecting non-linear impacts of frequent absenteeism

---

The model can be run by simply running `predict.py`, it can be retuned by running `modelTune.py` where the settings can be changed.
Student data can be entered into `predict.py` to produce a result of ~95-96% accuracy of their GPA.

---

## 📄 License & Attribution

The dataset used in this project is sourced from Kaggle and licensed under the **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)** license. 
Under this license, you are free to share, adapt, and build upon this material for any purpose, provided appropriate credit is given to the original dataset creator.

  

