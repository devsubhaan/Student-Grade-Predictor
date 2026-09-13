# Student GPA Prediction (WIP)

## 📌 About the Project

This project uses machine learning to predict a student's GPA based on information about their academic performance.

The project is an **end-to-end tabular machine learning project**, covering data loading, data preparation, training, prediction, and evaluation.

## 📊 Dataset

The dataset is stored in `performance.csv`.

Each row represents a student, with different features describing their academic performance.

The target variable is:

* **GPA** - the GPA I am trying to predict

## 🤖 Model

I used **XGBoost Regression** to predict GPA.

The dataset is split into:

* **80% training data** - used to train the model
* **20% testing data** - used to evaluate the model

The split uses `random_state=42` so the results are reproducible.

## 📏 Evaluation

Because GPA is a continuous numerical value, this is a **regression problem** rather than a classification problem.

I use **Mean Absolute Error (MAE)** to evaluate the model.

MAE measures the average difference between the predicted GPA and the actual GPA.

A lower MAE means the model's predictions are closer to the actual GPAs.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost

## 🚀 How to Run

1. Clone or download this repository.
2. Make sure `performance.csv` is in the project folder.
3. Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost
```

4. Run the Python script.

