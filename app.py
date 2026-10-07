from flask import Flask, render_template, request, jsonify, redirect, url_for, abort
import joblib
import numpy as np
import pandas as pd
import os

import content
import warehouse

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
linear_path = os.path.join(MODEL_DIR, "linear_model.joblib")
rf_path = os.path.join(MODEL_DIR, "random_forest_model.joblib")
meta_path = os.path.join(MODEL_DIR, "metadata.joblib")

if not all(os.path.exists(p) for p in [linear_path, rf_path, meta_path]):
    raise FileNotFoundError("Models are not available. Run: python train_model.py")

linear_model = joblib.load(linear_path)
rf_model = joblib.load(rf_path)
metadata = joblib.load(meta_path)

# Build the SQLite star-schema data warehouse automatically if needed.
warehouse.ensure_warehouse()


# ---------------------------------------------------------------------------
# Prediction (unchanged behaviour)
# ---------------------------------------------------------------------------
def make_prediction(form):
    if not all(form.values()):
        raise ValueError("Please fill in all fields.")

    try:
        year = float(form["year"])
    except ValueError:
        raise ValueError("Release year must be a number.")
    input_df = pd.DataFrame([{
        "Platform": form["platform"],
        "Year": year,
        "Genre": form["genre"],
        "Publisher": form["publisher"]
    }])

    linear_pred = max(0, float(linear_model.predict(input_df)[0]))
    rf_pred = max(0, float(rf_model.predict(input_df)[0]))

    return {
        "linear": round(linear_pred, 3),
        "random_forest": round(rf_pred, 3)
    }


# ---------------------------------------------------------------------------
# Model analytics helpers
# ---------------------------------------------------------------------------
METRIC_DEFS = [
    ("MAE", "Mean Absolute Error (MAE)", "lower"),
    ("MSE", "Mean Squared Error (MSE)", "lower"),
    ("RMSE", "Root Mean Squared Error (RMSE)", "lower"),
    ("R2", "R² Score", "higher"),
    ("MAPE", "Mean Absolute Percentage Error (MAPE)", "lower"),
]


def ml_ready():
    """True when models/metadata.joblib was produced by the upgraded train_model.py."""
    return ("test_predictions" in metadata
            and "MAPE" in metadata.get("linear_metrics", {})
            and "MAPE" in metadata.get("random_forest_metrics", {}))


def build_comparison():
    lm = metadata.get("linear_metrics", {})
    rm = metadata.get("random_forest_metrics", {})
    rows, wins = [], {"linear": 0, "random_forest": 0}
    for key, label, better in METRIC_DEFS:
        a, b = lm.get(key), rm.get(key)
        if a is None or b is None or (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
            continue
        if a == b:
            winner = "tie"
        elif better == "lower":
            winner = "linear" if a < b else "random_forest"
        else:
            winner = "linear" if a > b else "random_forest"
        if winner in wins:
            wins[winner] += 1
        rows.append({"key": key, "label": label, "better": better,
                     "linear": a, "random_forest": b, "winner": winner})
    if wins["linear"] > wins["random_forest"]:
        rec = "linear"
    elif wins["random_forest"] > wins["linear"]:
        rec = "random_forest"
    else:  # tie-break on R²
        rec = "linear" if lm.get("R2", 0) >= rm.get("R2", 0) else "random_forest"
    names = {"linear": "Linear Regression", "random_forest": "Random Forest Regression"}
    return {
        "rows": rows, "wins": wins, "recommended": rec, "recommended_name": names[rec],
        "reason": (f"{names[rec]} is better on {wins[rec]} of {len(rows)} metrics "
                   f"(lower MAE/MSE/RMSE/MAPE and higher R² are better)."),
    }


def histogram(values, bins=30, clip=None):
    arr = np.asarray(values, dtype=float)
    if clip is not None:
        lo, hi = np.percentile(arr, [100 - clip, clip])
        arr = arr[(arr >= lo) & (arr <= hi)]
    counts, edges = np.histogram(arr, bins=bins)
    labels = [round(float((edges[i] + edges[i + 1]) / 2), 3) for i in range(len(counts))]
    return {"labels": labels, "counts": [int(c) for c in counts], "shown": int(len(arr)), "total": int(len(values))}


def chart_payload(model_key):
    tp = metadata["test_predictions"]
    actual = np.array(tp["actual"])
    pred = np.array(tp[model_key])
    resid = actual - pred
    return {
        "actual": tp["actual"], "predicted": tp[model_key],
        "residual_hist": histogram(resid, bins=30, clip=99),
        "n_test": int(len(actual)),
        "max_axis": float(max(actual.max(), pred.max())),
    }


def page(template, **ctx):
    ctx.setdefault("ready", ml_ready())
    return render_template(template, **ctx)


# ---------------------------------------------------------------------------
# Navigation shared by every page
# ---------------------------------------------------------------------------
TOP_NAV = [
    ("Dashboard", "/", "/"),
    ("Dataset", "/dataset", "/dataset"),
    ("Data Warehousing", "/data-warehouse", "/data-warehouse"),
    ("Data Mining & ML", "/data-mining", "/data-mining"),
    ("Model Comparison", "/models/comparison", "/models/comparison"),
    ("Prediction", "/prediction", "/prediction"),
]

SIDEBAR = [
    {"title": "Data Warehousing", "items": [
        ("▣", "Data Warehouse Overview", "/data-warehouse"),
        ("✦", "Star Schema", "/data-warehouse/star-schema"),
        ("▦", "Fact & Dimension Tables", "/data-warehouse/fact-table"),
        ("⇉", "ETL Pipeline", "/data-warehouse/etl"),
        ("🗂", "Dataset Explorer", "/dataset"),
    ]},
    {"title": "Data Mining & Machine Learning", "items": [
        ("⌁", "Data Preprocessing", "/data-mining/preprocessing"),
        ("⚙", "Feature Engineering", "/data-mining/feature-engineering"),
        ("📈", "Linear Regression", "/models/linear-regression"),
        ("♣", "Random Forest Regression", "/models/random-forest"),
        ("⚖", "Model Comparison", "/models/comparison"),
        ("📊", "Model Evaluation", "/models/evaluation"),
        ("🎯", "Sales Prediction", "/prediction"),
    ]},
    {"title": "Project", "items": [
        ("⌬", "Architecture & Tech Stack", "/architecture"),
    ]},
]


@app.context_processor
def inject_nav():
    return {"top_nav": TOP_NAV, "sidebar": SIDEBAR, "current_path": request.path}


def predict_form_context(form=None):
    return dict(
        form=form or {"platform": "", "genre": "", "publisher": "", "year": ""},
        platforms=metadata["platforms"],
        genres=metadata["genres"],
        publishers=metadata["publishers"],
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    s = warehouse.summary()
    ds = warehouse.dataset_stats()
    return page("index.html", s=s, ds=ds, comp=build_comparison() if ml_ready() else None,
                etl_stages=content.etl_stages(warehouse.etl_log()),
                **predict_form_context())


@app.route("/prediction", methods=["GET"])
def prediction_page():
    return page("prediction.html", **predict_form_context())


@app.route("/dataset")
def dataset_page():
    return page("dataset.html", ds=warehouse.dataset_stats())


@app.route("/data-warehouse")
def dw_overview():
    return page("dw_overview.html", s=warehouse.summary(), etl=warehouse.etl_log(),
                opts=warehouse.filter_options(), sizes=warehouse.ALLOWED_PAGE_SIZES)


@app.route("/data-warehouse/star-schema")
def dw_star():
    return page("star_schema.html", s=warehouse.summary())


@app.route("/data-warehouse/fact-table")
def dw_fact():
    return page("fact_table.html", s=warehouse.summary(), opts=warehouse.filter_options(),
                sizes=warehouse.ALLOWED_PAGE_SIZES)


@app.route("/data-warehouse/etl")
def dw_etl():
    etl = warehouse.etl_log()
    return page("etl.html", etl=etl, stages=content.etl_stages(etl))


@app.route("/data-mining")
def dm_overview():
    return page("data_mining.html", stages=content.dm_stages(metadata), meta=metadata)


@app.route("/data-mining/preprocessing")
def dm_preprocessing():
    return page("preprocessing.html", meta=metadata)


@app.route("/data-mining/feature-engineering")
def dm_features():
    return page("feature_engineering.html", meta=metadata, ds=warehouse.dataset_stats())


@app.route("/models/linear-regression")
def model_linear():
    return page("linear.html", meta=metadata, m=metadata.get("linear_metrics", {}))


@app.route("/models/random-forest")
def model_rf():
    return page("forest.html", meta=metadata, m=metadata.get("random_forest_metrics", {}))


@app.route("/models/comparison")
def model_comparison():
    return page("comparison.html", meta=metadata, comp=build_comparison() if ml_ready() else None)


@app.route("/models/evaluation")
def model_evaluation():
    return page("evaluation.html", meta=metadata, comp=build_comparison() if ml_ready() else None)


@app.route("/architecture")
def architecture():
    return page("architecture.html", arch=content.ARCHITECTURE, stack=content.TECH_STACK)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("prediction_page"))

    form = {
        "platform": request.form.get("platform", "").strip(),
        "genre": request.form.get("genre", "").strip(),
        "publisher": request.form.get("publisher", "").strip(),
        "year": request.form.get("year", "").strip()
    }

    try:
        prediction = make_prediction(form)
        error = None
    except Exception as e:
        prediction = None
        error = str(e)

    diff = None
    if prediction:
        lin, rf = prediction["linear"], prediction["random_forest"]
        avg = (lin + rf) / 2
        diff = {
            "absolute": round(rf - lin, 3),
            "abs_value": round(abs(rf - lin), 3),
            "percent": round(abs(rf - lin) / avg * 100, 1) if avg > 0 else None,
            "higher": "Random Forest" if rf > lin else "Linear Regression" if lin > rf else None,
        }

    return page("result.html", prediction=prediction, error=error, form=form, diff=diff,
                comp=build_comparison() if ml_ready() else None)


# ---------------------------------------------------------------------------
# JSON APIs (all numbers come from the CSV, the warehouse or the trained models)
# ---------------------------------------------------------------------------
@app.route("/api/dataset-stats")
def api_dataset_stats():
    return jsonify(warehouse.dataset_stats())


@app.route("/api/warehouse/summary")
def api_wh_summary():
    return jsonify(warehouse.summary())


@app.route("/api/etl-log")
def api_etl_log():
    return jsonify(warehouse.etl_log())


@app.route("/api/warehouse/fact-table")
def api_fact_table():
    def to_int(v, d):
        try:
            return int(v)
        except (TypeError, ValueError):
            return d
    data = warehouse.fact_page(
        page=to_int(request.args.get("page"), 1),
        per_page=to_int(request.args.get("per_page"), 10),
        q=request.args.get("q", "").strip(),
        platform=request.args.get("platform", "").strip(),
        genre=request.args.get("genre", "").strip(),
        publisher=request.args.get("publisher", "").strip(),
        year=request.args.get("year", "").strip(),
    )
    return jsonify(data)


@app.route("/api/warehouse/dimension/<name>")
def api_dimension(name):
    try:
        limit = min(max(int(request.args.get("limit", 10)), 1), 100)
    except ValueError:
        limit = 10
    try:
        return jsonify(warehouse.dimension_preview(name, limit))
    except KeyError:
        abort(404)


@app.route("/api/metrics")
def api_metrics():
    return jsonify({
        "ready": ml_ready(),
        "linear": metadata.get("linear_metrics"),
        "random_forest": metadata.get("random_forest_metrics"),
        "best_params": metadata.get("best_params_clean", metadata.get("best_params")),
        "comparison": build_comparison() if ml_ready() else None,
    })


@app.route("/api/charts/linear")
def api_chart_linear():
    if not ml_ready():
        return jsonify({"error": "Run python train_model.py to generate chart data."}), 409
    return jsonify(chart_payload("linear"))


@app.route("/api/charts/random-forest")
def api_chart_rf():
    if not ml_ready():
        return jsonify({"error": "Run python train_model.py to generate chart data."}), 409
    p = chart_payload("random_forest")
    p["feature_importance"] = metadata.get("feature_importance", [])
    return jsonify(p)


@app.route("/api/charts/comparison")
def api_chart_comparison():
    if not ml_ready():
        return jsonify({"error": "Run python train_model.py to generate chart data."}), 409
    tp = metadata["test_predictions"]
    n = 40
    return jsonify({
        "comparison": build_comparison(),
        "linear": chart_payload("linear"),
        "random_forest": chart_payload("random_forest"),
        "first_n": {"n": n, "actual": tp["actual"][:n], "linear": tp["linear"][:n],
                    "random_forest": tp["random_forest"][:n]},
    })


if __name__ == "__main__":
    app.run(debug=True)
