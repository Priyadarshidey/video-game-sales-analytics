import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "vgsales.csv")   # unchanged location: data/vgsales.csv
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 3
GRID_SCORING = "neg_mean_absolute_error"


def safe_mape(y_true, y_pred):
    """Mean Absolute Percentage Error in %, safe against zero actual values.

    Rows where the actual value is exactly 0 are excluded (division by zero).
    Returns (mape_percent, number_of_rows_used, number_of_rows_excluded).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    used = int(mask.sum())
    if used == 0:
        return float("nan"), 0, int(len(y_true))
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    return mape, used, int(len(y_true) - used)


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    mape, used, excluded = safe_mape(y_test, pred)
    return {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_test, pred)),
        "MAPE": mape,                 # percent, zero-actual rows excluded
        "MAPE_samples_used": used,
        "MAPE_samples_excluded": excluded,
    }


def clean_feature_name(name):
    """'cat__Platform_PS4' -> 'Platform_PS4', 'num__Year' -> 'Year'."""
    for prefix in ("cat__", "num__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def main():
    # Kaggle Video Game Sales dataset:
    # Name, Platform, Year, Genre, Publisher, NA_Sales, EU_Sales,
    # JP_Sales, Other_Sales, Global_Sales
    df = pd.read_csv(DATA_PATH)
    raw_rows = len(df)

    # Keep the fields used by the project.
    required = ["Platform", "Year", "Genre", "Publisher", "Global_Sales"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df[required].copy()

    # Basic cleaning.
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Global_Sales"] = pd.to_numeric(df["Global_Sales"], errors="coerce")
    df = df.drop_duplicates()
    df = df.dropna(subset=["Global_Sales"])

    X = df[["Platform", "Year", "Genre", "Publisher"]]
    y = df["Global_Sales"]

    categorical_features = ["Platform", "Genre", "Publisher"]
    numeric_features = ["Year"]

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("cat", categorical_pipe, categorical_features),
        ("num", numeric_pipe, numeric_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Linear Regression baseline.
    linear_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])
    linear_model.fit(X_train, y_train)

    # Random Forest with GridSearchCV.
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 15],
        "model__min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        rf_pipeline,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring=GRID_SCORING,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    rf_model = grid.best_estimator_

    linear_metrics = evaluate(linear_model, X_test, y_test)
    rf_metrics = evaluate(rf_model, X_test, y_test)

    joblib.dump(linear_model, os.path.join(MODEL_DIR, "linear_model.joblib"))
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "random_forest_model.joblib"))

    # ---- extra metadata for the analytics dashboard -------------------------
    # Real test-set predictions (used for Actual vs Predicted / residual charts).
    linear_pred = linear_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    # Feature importance with real feature names after One-Hot Encoding.
    fitted_pre = rf_model.named_steps["preprocessor"]
    feature_names = [clean_feature_name(n) for n in fitted_pre.get_feature_names_out()]
    importances = rf_model.named_steps["model"].feature_importances_
    order = np.argsort(importances)[::-1][:15]
    feature_importance = [
        {"feature": feature_names[i], "importance": float(importances[i])} for i in order
    ]

    best_rf = rf_model.named_steps["model"]
    metadata = {
        "linear_metrics": linear_metrics,
        "random_forest_metrics": rf_metrics,
        "best_params": grid.best_params_,
        "platforms": sorted(df["Platform"].dropna().astype(str).unique().tolist()),
        "genres": sorted(df["Genre"].dropna().astype(str).unique().tolist()),
        "publishers": sorted(df["Publisher"].dropna().astype(str).unique().tolist()),

        # --- dashboard additions ---
        "feature_importance": feature_importance,
        "n_encoded_features": int(len(feature_names)),
        "param_grid": {k.replace("model__", ""): v for k, v in param_grid.items()},
        "best_params_clean": {k.replace("model__", ""): v for k, v in grid.best_params_.items()},
        "rf_config": {
            "n_estimators": best_rf.n_estimators,
            "max_depth": best_rf.max_depth,
            "min_samples_split": best_rf.min_samples_split,
            "random_state": best_rf.random_state,
        },
        "grid_search": {
            "cv_folds": CV_FOLDS,
            "scoring": GRID_SCORING,
            "n_candidates": int(len(grid.cv_results_["params"])),
            "best_cv_mae": float(-grid.best_score_),
        },
        "dataset_info": {
            "raw_rows": int(raw_rows),
            "rows_after_cleaning": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_platforms": int(df["Platform"].nunique()),
            "n_genres": int(df["Genre"].nunique()),
            "n_publishers": int(df["Publisher"].nunique()),
            "missing_year": int(df["Year"].isna().sum()),
            "missing_publisher": int(df["Publisher"].isna().sum()),
            "target_min": float(y.min()),
            "target_max": float(y.max()),
        },
        "test_predictions": {
            "actual": [round(float(v), 4) for v in y_test],
            "linear": [round(float(v), 4) for v in linear_pred],
            "random_forest": [round(float(v), 4) for v in rf_pred],
        },
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(metadata, os.path.join(MODEL_DIR, "metadata.joblib"))

    print("\nTraining complete.")
    print("\nLinear Regression:", linear_metrics)
    print("\nRandom Forest:", rf_metrics)
    print("\nBest Random Forest parameters:", grid.best_params_)

    # (Re)build the SQLite star-schema data warehouse as part of the pipeline.
    try:
        import warehouse
        info = warehouse.run_etl()
        print(f"\nData warehouse rebuilt: {warehouse.DB_PATH} ({info['fact_rows']} fact rows)")
    except Exception as exc:  # warehouse is also auto-built by app.py
        print("\nWarehouse build skipped:", exc)


if __name__ == "__main__":
    main()
