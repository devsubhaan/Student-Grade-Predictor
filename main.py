import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# Load data
df = pd.read_csv("performance.csv", index_col="StudentID")
df.dropna(subset=["GPA"], axis=0, inplace=True)
if "GradeClass" in df.columns:
    df.drop("GradeClass", axis=1, inplace=True)

X = df.drop(columns=["GPA"])
y = df["GPA"]


# Feature Engineering
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

X = featureEngineering(X)

# Show distribution of GPA
plt.figure(figsize=(6, 4))
sns.histplot(y, kde=True, color="teal")
plt.title("GPA Distribution")
plt.xlabel("GPA")
plt.show()

# Show correlation matrix
plt.figure(figsize=(10, 8))
full_engineered_df = pd.concat([X, y], axis=1)
num_df = full_engineered_df.select_dtypes(include=["int64", "float64"])
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

numberCols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categoricalCols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

kf = KFold(n_splits=5, shuffle=True, random_state=1)


# Pipeline
def testRegressor(model):
    numTransformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    catTransformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numTransformer, numberCols),
            ("cat", catTransformer, categoricalCols),
        ]
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model)]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    scores = -cross_val_score(
        pipeline, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error"
    )

    return mae, scores


# Models
modelXGB = XGBRegressor(
    random_state=1, learning_rate=0.27, n_estimators=50, max_depth=3
)
modelCat = CatBoostRegressor(
    random_state=1,
    learning_rate=0.27,
    n_estimators=50,
    max_depth=3,
    verbose=0,
)
modelRidge = Ridge(random_state=1, alpha=0.5)

maeXGB, scoresXGB = testRegressor(modelXGB)
maeCat, scoresCat = testRegressor(modelCat)
maeRidge, scoresRidge = testRegressor(modelRidge)

print(f"XGBoost MAE: {maeXGB:.4f} | CV Mean: {scoresXGB.mean():.4f} +/- {scoresXGB.std():.4f}")
print(f"CatBoost MAE: {maeCat:.4f} | CV Mean: {scoresCat.mean():.4f} +/- {scoresCat.std():.4f}")
print(f"Ridge MAE: {maeRidge:.4f} | CV Mean: {scoresRidge.mean():.4f} +/- {scoresRidge.std():.4f}")

# Subplots
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

axes[0].plot(scoresXGB, marker="o", color="blue", label="XGBRegressor")
axes[0].set_ylabel("MAE")
axes[0].set_title("XGBoost 5-Fold Cross-Validation MAE")
axes[0].legend()

axes[1].plot(scoresCat, marker="o", color="orange", label="CatBoostRegressor")
axes[1].set_ylabel("MAE")
axes[1].set_title("CatBoost 5-Fold Cross-Validation MAE")
axes[1].legend()

axes[2].plot(scoresRidge, marker="o", color="green", label="Ridge Regression")
axes[2].set_xlabel("Fold")
axes[2].set_ylabel("MAE")
axes[2].set_title("Ridge 5-Fold Cross-Validation MAE")
axes[2].legend()

plt.tight_layout()
plt.show()
