# Video Game Sales Predictor - Data Warehousing + Data Mining + ML Analytics Dashboard

A Flask web application that combines a **star-schema data warehouse (SQLite)**, a **data-mining / preprocessing pipeline** and two **regression models** (Linear Regression and Random Forest) to predict `Global_Sales` of video games, presented as a dark analytics dashboard.

## Dataset
Kaggle *Video Game Sales* - save as `data/vgsales.csv`.
Columns: Rank, Name, Platform, Year, Genre, Publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales.
- **Features:** Platform, Year, Genre, Publisher  **Target:** Global_Sales (million units)
- Regional sales are not features: they add up to Global_Sales (target leakage). Name/Rank are identifiers.

## Data warehousing architecture (`warehouse.py`)
Implemented ETL: `extract()` (pandas) -> `transform()` -> `load()` (SQLite `datawarehouse.db`).
- **Transform:** trim text, numeric Year/sales, drop invalid rows and duplicate releases (Name+Platform+Year+Genre+Publisher), missing Publisher/Genre -> `Unknown`, missing Year -> NULL "unknown" year member, surrogate keys for dimensions.
- **Star schema:** `fact_sales` (Sales_ID, Game_ID, Platform_ID, Genre_ID, Publisher_ID, Year_ID, NA/EU/JP/Other/Global_Sales) with `dim_game`, `dim_platform`, `dim_genre`, `dim_publisher`, `dim_year`. An extra `etl_log` table stores real run statistics shown in the UI.
- The database is **created automatically** when the app starts (or run `python warehouse.py`; `python train_model.py` also rebuilds it). It is rebuilt if the CSV is newer.
- Scope note: the ML models are trained from `data/vgsales.csv`, not by querying SQLite; the link between the warehouse and the ML dataset is a conceptual mapping (a join of the fact table with its dimensions).

## Data mining process / preprocessing
Cleaning -> missing-value imputation -> feature selection -> One-Hot Encoding -> StandardScaler -> 80/20 split (random_state=42) -> training -> evaluation -> prediction, using `SimpleImputer`, `OneHotEncoder(handle_unknown="ignore")`, `StandardScaler`, `ColumnTransformer`, `Pipeline`.

## ML algorithms and evaluation
- Linear Regression (baseline)
- Random Forest Regressor tuned with `GridSearchCV` (n_estimators [100,200], max_depth [None,15], min_samples_split [2,5], 3-fold CV, scoring = negative MAE)
- Metrics: MAE, MSE, RMSE, R², and a **zero-safe MAPE** (rows with actual = 0 are excluded). They are stored with feature importance, best parameters, split sizes and the real test-set predictions in `models/metadata.joblib`.
- The recommended model on the Comparison page is computed from the metrics (number of metrics won, R² tie-break); it is not assumed.

## Technology stack
Python, Flask, Pandas, NumPy, Scikit-learn, GridSearchCV, Joblib, SQLite, HTML5/CSS3/JavaScript, Chart.js (bundled in `static/vendor/`, works offline).

## Project structure
```
app.py              Flask routes + JSON APIs + prediction
train_model.py      training, evaluation, metadata, rebuilds warehouse
warehouse.py        ETL, SQLite queries, dataset statistics
content.py          pipeline stage explanations
data/vgsales.csv    dataset
models/             linear_model / random_forest_model / metadata (.joblib)
datawarehouse.db    generated SQLite warehouse
templates/          base, dashboard, warehouse, ML, result pages
static/             style.css, app.js, vendor/chart.umd.js, gaming-bg.png
```

## Installation, training, running
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python train_model.py          # trains models, writes metrics, builds datawarehouse.db
python app.py                  # open http://127.0.0.1:5000/
```
Existing model files stay compatible (same pipelines). If `metadata.joblib` comes from the old script, prediction still works but metrics/charts show a notice: re-run `python train_model.py`.

## Dashboard modules / routes
`/` dashboard · `/dataset` · `/data-warehouse` · `/data-warehouse/star-schema` · `/data-warehouse/fact-table` (10/50/100/500/1000 per page, search + filters) · `/data-warehouse/etl` · `/data-mining` · `/data-mining/preprocessing` · `/data-mining/feature-engineering` · `/models/linear-regression` · `/models/random-forest` · `/models/comparison` · `/models/evaluation` · `/prediction` · `POST /predict` (result opens in a new tab) · `/architecture`.

JSON APIs: `/api/dataset-stats`, `/api/warehouse/summary`, `/api/warehouse/fact-table`, `/api/warehouse/dimension/<name>`, `/api/etl-log`, `/api/metrics`, `/api/charts/{linear,random-forest,comparison}`.
