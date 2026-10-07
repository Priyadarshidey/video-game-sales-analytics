"""Explanatory content for the dashboard.

Every stage is tagged either ACTUAL (implemented in code in this project) or
CONCEPTUAL (an analytical mapping that is explained but not a separate piece of code).
Numbers are filled in from the real ETL log / training metadata at request time.
"""

ACTUAL = "ACTUAL IMPLEMENTATION"
CONCEPT = "CONCEPTUAL MAPPING"


def fmt(n):
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def etl_stages(etl):
    e = etl or {}
    return [
        {"key": "csv", "icon": "🗂", "title": "Kaggle CSV", "impl": ACTUAL,
         "component": "data/vgsales.csv",
         "what": "The raw Kaggle Video Game Sales dataset - the single source for the whole project.",
         "project": f"{fmt(e.get('raw_rows'))} rows x {e.get('raw_columns')} columns (Rank, Name, Platform, Year, Genre, Publisher, NA/EU/JP/Other/Global sales)."},
        {"key": "extract", "icon": "⇩", "title": "Extract", "impl": ACTUAL,
         "component": "warehouse.extract()",
         "what": "Reads the CSV into a pandas DataFrame without changing it.",
         "project": "pandas.read_csv('data/vgsales.csv'). The raw file is never modified."},
        {"key": "clean", "icon": "🧹", "title": "Data Cleaning", "impl": ACTUAL,
         "component": "warehouse.transform()",
         "what": "Removes unusable and repeated records and fixes data types.",
         "project": (f"Text fields are trimmed; Year and the sales columns are converted to numbers; "
                     f"{fmt(e.get('dropped_invalid'))} invalid rows dropped (no name/platform/global sales or negative sales); "
                     f"{fmt(e.get('duplicates_removed'))} duplicate game releases removed (same Name+Platform+Year+Genre+Publisher).")},
        {"key": "transform", "icon": "⚙", "title": "Transformation", "impl": ACTUAL,
         "component": "warehouse.transform()",
         "what": "Handles missing values so every record can be stored in the star schema.",
         "project": (f"{fmt(e.get('missing_year'))} missing Year values are kept and linked to an explicit 'Unknown' (NULL) year member; "
                     f"{fmt(e.get('missing_publisher'))} missing Publisher and {fmt(e.get('missing_genre'))} missing Genre values become 'Unknown'.")},
        {"key": "dims", "icon": "▦", "title": "Dimension Creation", "impl": ACTUAL,
         "component": "warehouse.transform()",
         "what": "Each descriptive attribute is normalised into its own dimension table with a surrogate integer key.",
         "project": (f"dim_game {fmt(e.get('dim_game_rows'))}, dim_platform {fmt(e.get('dim_platform_rows'))}, "
                     f"dim_genre {fmt(e.get('dim_genre_rows'))}, dim_publisher {fmt(e.get('dim_publisher_rows'))}, "
                     f"dim_year {fmt(e.get('dim_year_rows'))} rows.")},
        {"key": "fact", "icon": "◈", "title": "Fact Table Creation", "impl": ACTUAL,
         "component": "warehouse.transform()",
         "what": "The fact table keeps the numeric measures and only foreign keys to the dimensions.",
         "project": f"fact_sales: {fmt(e.get('fact_rows'))} rows (one per game-on-platform release) with 5 measures: NA, EU, JP, Other and Global sales."},
        {"key": "star", "icon": "✦", "title": "Star Schema (Load)", "impl": ACTUAL,
         "component": "warehouse.load() -> datawarehouse.db",
         "what": "Loads the tables into SQLite with primary keys, foreign keys and indexes.",
         "project": f"datawarehouse.db is created automatically (built {e.get('built_at', 'at start-up')}). The Fact & Dimension page and the pagination API query this database."},
        {"key": "ml", "icon": "⌬", "title": "ML Dataset", "impl": CONCEPT,
         "component": "train_model.py",
         "what": "The columns the models learn from: Platform, Year, Genre, Publisher -> Global_Sales.",
         "project": ("train_model.py reads data/vgsales.csv directly (it does not query SQLite). The same four features and the target "
                     "could be produced from the warehouse with one join of fact_sales and its dimensions - this link is a conceptual mapping.")},
        {"key": "models", "icon": "♣", "title": "Prediction Models", "impl": ACTUAL,
         "component": "train_model.py / app.py",
         "what": "Linear Regression and Random Forest Regression are trained, evaluated and saved with Joblib.",
         "project": "Flask loads models/*.joblib and serves live predictions on the Prediction page."},
    ]


def dm_stages(meta):
    di = (meta or {}).get("dataset_info", {})
    ne = (meta or {}).get("n_encoded_features")
    return [
        {"key": "raw", "icon": "🗂", "title": "Raw Dataset", "impl": ACTUAL, "component": "data/vgsales.csv",
         "what": "Start from the unmodified Kaggle file.",
         "project": f"{fmt(di.get('raw_rows'))} rows are read by train_model.py with pandas."},
        {"key": "clean", "icon": "🧹", "title": "Data Cleaning", "impl": ACTUAL, "component": "train_model.py",
         "what": "Remove unusable records and fix data types.",
         "project": f"Year and Global_Sales are converted with pd.to_numeric; duplicates are dropped; rows without Global_Sales are removed. {fmt(di.get('rows_after_cleaning'))} rows remain."},
        {"key": "missing", "icon": "◌", "title": "Missing Value Handling", "impl": ACTUAL, "component": "SimpleImputer (inside the Pipeline)",
         "what": "Fill gaps so the algorithms receive complete data.",
         "project": "Categorical columns: most_frequent imputation. Year: median imputation. Imputers are fitted on the training split only, inside the model pipeline."},
        {"key": "select", "icon": "☑", "title": "Feature Selection", "impl": ACTUAL, "component": "train_model.py",
         "what": "Choose the input columns and the target.",
         "project": "Features: Platform, Year, Genre, Publisher. Target: Global_Sales. Name, Rank and the regional sales columns are not used (see the Dataset page)."},
        {"key": "encode", "icon": "01", "title": "Encoding", "impl": ACTUAL, "component": "OneHotEncoder(handle_unknown='ignore')",
         "what": "Convert text categories to numeric columns.",
         "project": f"Platform, Genre and Publisher become {fmt(ne)} binary columns in total. Unseen categories are ignored instead of raising an error."},
        {"key": "scale", "icon": "↔", "title": "Scaling", "impl": ACTUAL, "component": "StandardScaler",
         "what": "Put numeric values on a comparable scale.",
         "project": "Only Year is numeric: it is centred to mean 0 and scaled to unit variance."},
        {"key": "split", "icon": "✂", "title": "Train/Test Split", "impl": ACTUAL, "component": "train_test_split(test_size=0.20, random_state=42)",
         "what": "Hold back unseen data to measure generalisation.",
         "project": f"{fmt(di.get('n_train'))} training rows (80%) and {fmt(di.get('n_test'))} test rows (20%)."},
        {"key": "train", "icon": "⚙", "title": "Model Training", "impl": ACTUAL, "component": "LinearRegression / RandomForestRegressor + GridSearchCV",
         "what": "Fit the two regression models on the training split.",
         "project": "Linear Regression is the baseline. Random Forest is tuned with GridSearchCV over n_estimators, max_depth and min_samples_split (3-fold CV, scoring = negative MAE)."},
        {"key": "eval", "icon": "📊", "title": "Evaluation", "impl": ACTUAL, "component": "train_model.evaluate()",
         "what": "Score the models on the test split.",
         "project": "MAE, MSE, RMSE, R² and a zero-safe MAPE are computed for both models and stored in models/metadata.joblib."},
        {"key": "predict", "icon": "🎯", "title": "Prediction", "impl": ACTUAL, "component": "app.py make_prediction()",
         "what": "Use the saved models on new input.",
         "project": "Flask builds a one-row DataFrame from the form and calls predict() on both pipelines. Negative outputs are clipped to 0 for display."},
    ]


ARCHITECTURE = [
    ("CSV Dataset", "data/vgsales.csv - the Kaggle source file."),
    ("Pandas", "Reads and manipulates the data (warehouse.py, train_model.py)."),
    ("Data Cleaning", "Type conversion, duplicate removal, missing-value handling."),
    ("Data Warehouse / Star Schema", "SQLite datawarehouse.db with fact_sales + 5 dimensions (built by warehouse.py)."),
    ("Preprocessing Pipeline", "SimpleImputer, OneHotEncoder, StandardScaler inside a ColumnTransformer."),
    ("Train/Test Split", "80/20 split with random_state=42."),
    ("Linear Regression + Random Forest", "Baseline model and the GridSearchCV-tuned ensemble."),
    ("Evaluation", "MAE, MSE, RMSE, R², MAPE on the test split."),
    ("Joblib Model Storage", "models/linear_model.joblib, random_forest_model.joblib, metadata.joblib."),
    ("Flask Backend", "app.py: page routes, JSON APIs and the prediction endpoint."),
    ("HTML/CSS/JS Dashboard", "Jinja templates, style.css, app.js and Chart.js (bundled locally)."),
]

TECH_STACK = [
    ("Backend", ["Python", "Flask"]),
    ("Data Processing", ["Pandas", "NumPy"]),
    ("Machine Learning", ["Scikit-learn"]),
    ("Models", ["Linear Regression", "Random Forest Regression"]),
    ("Model Optimization", ["GridSearchCV"]),
    ("Model Persistence", ["Joblib"]),
    ("Database / Data Warehouse", ["SQLite (datawarehouse.db)"]),
    ("Frontend", ["HTML5", "CSS3", "JavaScript"]),
    ("Visualization", ["Chart.js (bundled in static/vendor)"]),
]
