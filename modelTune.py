import optuna
import json
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

from main import loadData, testRegressor

X,y = loadData()

numberCols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categoricalCols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

kf = KFold(n_splits=5, shuffle=True, random_state=1)
    
def tuneXGB(trial):
    params = {
        "random_state": 1,
        "n_jobs": -1, 
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.005, 0.2, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    model = XGBRegressor(**params)
    return testRegressor(model, tune=True)

def tuneCat(trial):
    params = {
        "random_state": 1,
        "verbose": 0,
        "thread_count": -1, 
        "iterations": trial.suggest_int("iterations", 50, 350),
        "depth": trial.suggest_int("depth", 2, 8),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.005, 0.2, log=True
        ),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", 1e-3, 20.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "random_strength": trial.suggest_float(
            "random_strength", 1e-3, 10.0, log=True
        ),
    }

    model = CatBoostRegressor(**params)
    return testRegressor(model, tune=True)


def tuneRidge(trial):
    params = {
        "random_state": 1,
        "alpha": trial.suggest_float("alpha", 1e-3, 1000.0, log=True),
        "solver": trial.suggest_categorical(
            "solver", ["auto", "svd", "cholesky", "lsqr", "sag"]
        ),
    }

    model = Ridge(**params)
    return testRegressor(model, tune=True)

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    #Ridge
    print("Tuning Ridge...")
    study_ridge = optuna.create_study(direction="minimize")
    study_ridge.enqueue_trial({"alpha": 0.5, "solver": "auto"})
    study_ridge.optimize(tuneRidge, n_trials=50)

    print("Tuned!")

    #XGBoost
    print("Tuning XGBoost...")
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.enqueue_trial(
        {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.15,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 1e-8,
            "reg_lambda": 1.0,
            "min_child_weight": 1,
        }
    )
    study_xgb.optimize(tuneXGB, n_trials=100)

    print("Tuned!")

    # CatBoost
    print("Tuning CatBoost...")
    study_cat = optuna.create_study(direction="minimize")
    study_cat.enqueue_trial(
        {
            "iterations": 50,
            "depth": 3,
            "learning_rate": 0.15,
            "l2_leaf_reg": 3.0,
            "subsample": 1.0,
            "random_strength": 1.0,
        }
    )
    study_cat.optimize(tuneCat, n_trials=100)
    
    print("Tuned!")
    bestParams = {
        "ridge": study_ridge.best_params,
        "xgboost": study_xgb.best_params,
        "catboost": study_cat.best_params,
    }

    with open("Misc/bestParams.json", "w") as f:
        json.dump(bestParams, f, indent=4)

    print("Save successful!")