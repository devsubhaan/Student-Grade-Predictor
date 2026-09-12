import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


df = pd.read_csv("performance.csv", index_col="StudentID")
df.dropna(inplace=True, subset=["GPA"], axis=0) #drop rows with missing values
X = df.drop(columns=["GPA"])
y = df["GPA"]

numberCols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categoricalCols = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, #20% of the data will be used for testing
    random_state=42
)

model = XGBRegressor(random_state=0, learning_rate=0.27, n_estimators=50, max_depth=3)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numberCols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricalCols),
    ]
)

pipeline = Pipeline(
    steps =[
        ("preprocessor", preprocessor),
        ("model", model)
    ]

)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")

