import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


df = pd.read_csv("performance.csv", index_col="StudentID")
df.dropna(inplace=True, subset=["GPA"], axis=0) #drop rows with missing values
X = df.drop(columns=["GPA"])
y = df["GPA"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, #20% of the data will be used for testing
    random_state=42
)

def findBest(learning_rate, n_estimators=50):
    model = XGBRegressor(random_state=0, learning_rate=learning_rate, n_estimators=50, max_depth=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae


"""
bestN_estimators = {}
for n_estimators in np.array([n for n in range(25,200, 25)]):
    mae = findBest(learning_rate= min(bestLR, key=bestLR.get), n_estimators= n_estimators)
    bestN_estimators[n_estimators] = mae

plt.subplot(1, 2, 2)
plt.plot(list(bestN_estimators.keys()), list(bestN_estimators.values()), marker='o', ms=0.5)
plt.xlabel("Number of Estimators")
plt.ylabel("Mean Absolute Error")
plt.title("Model Performance vs Number of Estimators")
plt.show()


print("Best Number of Estimators:", min(bestN_estimators, key=bestN_estimators.get))
print("Best Mean Absolute Error (N Estimators):", bestN_estimators[min(bestN_estimators, key=bestN_estimators.get)])



#Find the best learning rate by testing different values and plotting the results
bestLR = {}

for learningRate in np.array([n for n in range(1,100)]):
    mae = findBest(learningRate/100)
    bestLR[learningRate] = mae


plt.subplot(1, 2, 1)
plt.plot(list(bestLR.keys()), list(bestLR.values()), marker='o', ms=0.5)
plt.xlabel("Learning Rate")
plt.ylabel("Mean Absolute Error")
plt.title("Model Performance vs Learning Rate")
plt.show()


print("Best Learning Rate:", min(bestLR, key=bestLR.get))

"""

model = XGBRegressor(random_state=0, learning_rate=0.27, n_estimators=50, max_depth=3)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")

