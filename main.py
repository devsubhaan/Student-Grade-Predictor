import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


df = pd.read_csv("performance.csv", index_col="StudentID")
df.dropna(inplace=True, subset=["GPA"], axis=0) #drop rows with missing values
X = df.drop(columns=["GPA"])
y = df["GPA"]


kf = KFold(n_splits=5, shuffle=True, random_state=1)

numberCols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categoricalCols = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, #20% of the data will be used for testing
    random_state=1
)

def testRegressor(model):
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
    scores = -cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")
    return mae, scores

modelXGB = XGBRegressor(random_state=1, learning_rate=0.27, n_estimators=50, max_depth=3)
modelCat = CatBoostRegressor(random_state=1, learning_rate=0.27, n_estimators=50, max_depth=3, verbose=0)
modelRidge = Ridge(random_state=1, alpha=0.5)

maeXGB, scoresXGB = testRegressor(modelXGB)
maeCat, scoresCat = testRegressor(modelCat)
maeRidge, scoresRidge = testRegressor(modelRidge)

"""Plot each model's cross-validation scores in a single figure with 3 subplots, one for each model."""

plt.subplot(3, 1, 1)
plt.plot(scoresXGB, marker="o", label="XGBRegressor")
plt.xlabel("Fold")
plt.ylabel("MAE")
plt.title("Cross-Validation Scores")
plt.legend()

print("XGB Regressor Results:")
print(f"MAE: {maeXGB:.2f}")
print(f"Cross-Validation Scores: {scoresXGB}")
print(f"Mean CV Score: {scoresXGB.mean():.2f}")

plt.subplot(3, 1, 2)
plt.plot(scoresCat, marker="o", label="CatBoostRegressor")
plt.xlabel("Fold")
plt.ylabel("MAE")
plt.title("Cross-Validation Scores")
plt.legend()

print("CatBoost Regressor Results:")
print(f"MAE: {maeCat:.2f}")
print(f"Cross-Validation Scores: {scoresCat}")
print(f"Mean CV Score: {scoresCat.mean():.2f}")

plt.subplot(3, 1, 3)
plt.plot(scoresRidge, marker="o", label="Ridge Regression")
plt.xlabel("Fold")
plt.ylabel("MAE")
plt.title("Cross-Validation Scores")
plt.legend()

print(f"Ridge Regression Results:")
print(f"MAE: {maeRidge:.2f}")
print(f"Cross-Validation Scores: {scoresRidge}")
print(f"Mean CV Score: {scoresRidge.mean():.2f}")

print("\n")
print("Best Model:")
if scoresXGB.mean() <= scoresCat.mean() and scoresXGB.mean() <= scoresRidge.mean():
    print("XGB Regressor")
    print(f"MAE: {maeXGB:.2f}")
elif scoresCat.mean() <= scoresRidge.mean():
    print("CatBoost Regressor")
    print(f"MAE: {maeCat:.2f}")
else:
    print("Ridge Regression")
    print(f"MAE: {maeRidge:.2f}")

print("\n")


plt.tight_layout()
plt.show()