# Student GPA Prediction (WIP)

## 📌 About the Project

This project uses machine learning to predict a student's GPA based on information about their academic performance.

The project is an **end-to-end tabular machine learning project**, covering data loading, data preparation, training, prediction, and evaluation.

## 📊 Dataset

The dataset is stored in `performance.csv`.

Each row represents a student, with different features describing their academic performance.

The target variable is:

* **GPA** - the GPA to predict

## 🤖 Model

I used **XGBoost Regression** to predict GPA.

The dataset is split into:

* **80% training data** - used to train the model
* **20% testing data** - used to evaluate the model

The split uses `random_state=1` so the results are reproducible.

## 📏 Evaluation

Because GPA is a continuous numerical value, this is a **regression problem** rather than a classification problem.

I use **Mean Absolute Error (MAE)** to evaluate the model.

MAE measures the average difference between the predicted GPA and the actual GPA.

A lower MAE means the model's predictions are closer to the actual GPAs.

## 🛠️ Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* Catboost
* Ridge
* KFold
* Seaborn (Visualisation)
* Optuna (Parameter tuning)
  

