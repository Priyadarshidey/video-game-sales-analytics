"""
Data Warehouse module for the Video Game Sales Predictor.

Implements a real (local) ETL process that turns data/vgsales.csv into a
star schema stored in SQLite (datawarehouse.db):

    fact_sales  -> dim_game, dim_platform, dim_genre, dim_publisher, dim_year

The database is created automatically (see ensure_warehouse) the first time the
Flask app starts, or whenever `python warehouse.py` / `python train_model.py`
is executed.  Nothing has to be created by hand.
"""
import os
import sqlite3
import time
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "vgsales.csv")
DB_PATH = os.path.join(BASE_DIR, "datawarehouse.db")

REGION_COLS = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
MEASURES = REGION_COLS + ["Global_Sales"]
ALLOWED_PAGE_SIZES = [10, 50, 100, 500, 1000]
DIM_TABLES = {
    "dim_game": ("Game_ID", ["Game_ID", "Name"]),
    "dim_platform": ("Platform_ID", ["Platform_ID", "Platform"]),
    "dim_genre": ("Genre_ID", ["Genre_ID", "Genre"]),
    "dim_publisher": ("Publisher_ID", ["Publisher_ID", "Publisher"]),
    "dim_year": ("Year_ID", ["Year_ID", "Year"]),
}

SCHEMA_SQL = """
CREATE TABLE dim_game      (Game_ID INTEGER PRIMARY KEY, Name TEXT NOT NULL);
CREATE TABLE dim_platform  (Platform_ID INTEGER PRIMARY KEY, Platform TEXT NOT NULL);
CREATE TABLE dim_genre     (Genre_ID INTEGER PRIMARY KEY, Genre TEXT NOT NULL);
CREATE TABLE dim_publisher (Publisher_ID INTEGER PRIMARY KEY, Publisher TEXT NOT NULL);
CREATE TABLE dim_year      (Year_ID INTEGER PRIMARY KEY, Year INTEGER);  -- NULL = year unknown
CREATE TABLE fact_sales (
    Sales_ID      INTEGER PRIMARY KEY,
    Game_ID       INTEGER NOT NULL REFERENCES dim_game(Game_ID),
    Platform_ID   INTEGER NOT NULL REFERENCES dim_platform(Platform_ID),
    Genre_ID      INTEGER NOT NULL REFERENCES dim_genre(Genre_ID),
    Publisher_ID  INTEGER NOT NULL REFERENCES dim_publisher(Publisher_ID),
    Year_ID       INTEGER NOT NULL REFERENCES dim_year(Year_ID),
    NA_Sales      REAL NOT NULL,
    EU_Sales      REAL NOT NULL,
    JP_Sales      REAL NOT NULL,
    Other_Sales   REAL NOT NULL,
    Global_Sales  REAL NOT NULL
);
CREATE INDEX idx_fact_game      ON fact_sales(Game_ID);
CREATE INDEX idx_fact_platform  ON fact_sales(Platform_ID);
CREATE INDEX idx_fact_genre     ON fact_sales(Genre_ID);
CREATE INDEX idx_fact_publisher ON fact_sales(Publisher_ID);
CREATE INDEX idx_fact_year      ON fact_sales(Year_ID);
CREATE TABLE etl_log (stage TEXT, metric TEXT, value TEXT);
"""


# --------------------------------------------------------------------------
# ETL
# --------------------------------------------------------------------------
def extract(csv_path=CSV_PATH):
    """EXTRACT: read the raw Kaggle CSV with pandas."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Place vgsales.csv in the data/ folder.")
    return pd.read_csv(csv_path)


def transform(raw):
    """TRANSFORM: clean the data and build dimension tables + fact table.

    Returns (dims, fact, log) where `log` holds real counts for the UI.
    """
    log = {"raw_rows": int(len(raw)), "raw_columns": int(raw.shape[1])}
    df = raw.copy()

    # --- cleaning -------------------------------------------------------
    for c in ["Name", "Platform", "Genre", "Publisher"]:
        df[c] = df[c].apply(lambda v: v.strip() if isinstance(v, str) else v)
        df[c] = df[c].replace("", np.nan)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    for c in MEASURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    log["missing_year"] = int(df["Year"].isna().sum())
    log["missing_publisher"] = int(df["Publisher"].isna().sum())
    log["missing_genre"] = int(df["Genre"].isna().sum())

    # rows that cannot be placed in the warehouse (no identity / no target / negative sales)
    invalid = (
        df["Name"].isna() | df["Platform"].isna()
        | df["Global_Sales"].isna() | (df["Global_Sales"] < 0)
    )
    log["dropped_invalid"] = int(invalid.sum())
    df = df[~invalid].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["Name", "Platform", "Year", "Genre", "Publisher"], keep="first")
    log["duplicates_removed"] = int(before - len(df))

    # --- missing value handling ----------------------------------------
    df["Genre"] = df["Genre"].fillna("Unknown")
    df["Publisher"] = df["Publisher"].fillna("Unknown")
    for c in REGION_COLS:
        df[c] = df[c].fillna(0.0)
    df = df.reset_index(drop=True)

    # --- dimensions (surrogate keys start at 1) ------------------------
    def make_dim(col, id_col):
        values = sorted(df[col].astype(str).unique().tolist())
        dim = pd.DataFrame({id_col: range(1, len(values) + 1), col: values})
        return dim, dict(zip(dim[col], dim[id_col]))

    dim_game, game_map = make_dim("Name", "Game_ID")
    dim_platform, platform_map = make_dim("Platform", "Platform_ID")
    dim_genre, genre_map = make_dim("Genre", "Genre_ID")
    dim_publisher, publisher_map = make_dim("Publisher", "Publisher_ID")

    years = sorted(df["Year"].dropna().astype(int).unique().tolist())
    dim_year = pd.DataFrame({"Year_ID": range(1, len(years) + 1), "Year": years})
    year_map = dict(zip(dim_year["Year"], dim_year["Year_ID"]))
    unknown_year_id = None
    if df["Year"].isna().any():  # keep rows with an unknown year via an explicit NULL member
        unknown_year_id = len(years) + 1
        dim_year = pd.concat(
            [dim_year, pd.DataFrame({"Year_ID": [unknown_year_id], "Year": [pd.NA]})],
            ignore_index=True,
        )

    # --- fact table ----------------------------------------------------
    fact = pd.DataFrame({
        "Sales_ID": range(1, len(df) + 1),
        "Game_ID": df["Name"].astype(str).map(game_map),
        "Platform_ID": df["Platform"].astype(str).map(platform_map),
        "Genre_ID": df["Genre"].astype(str).map(genre_map),
        "Publisher_ID": df["Publisher"].astype(str).map(publisher_map),
        "Year_ID": df["Year"].map(lambda y: year_map[int(y)] if pd.notna(y) else unknown_year_id),
    })
    for c in MEASURES:
        fact[c] = df[c].astype(float).round(4)

    log.update({
        "fact_rows": int(len(fact)),
        "dim_game_rows": int(len(dim_game)),
        "dim_platform_rows": int(len(dim_platform)),
        "dim_genre_rows": int(len(dim_genre)),
        "dim_publisher_rows": int(len(dim_publisher)),
        "dim_year_rows": int(len(dim_year)),
    })
    dims = {
        "dim_game": dim_game, "dim_platform": dim_platform, "dim_genre": dim_genre,
        "dim_publisher": dim_publisher, "dim_year": dim_year,
    }
    return dims, fact, log


def _to_rows(frame):
    """DataFrame -> list of tuples with NaN/NA converted to None and numpy ints to int."""
    out = []
    for row in frame.itertuples(index=False, name=None):
        out.append(tuple(None if pd.isna(v) else (int(v) if isinstance(v, (np.integer,)) else
                                                  float(v) if isinstance(v, (np.floating,)) else v)
                         for v in row))
    return out


def load(dims, fact, log, db_path=DB_PATH):
    """LOAD: write dimensions + fact table into a fresh SQLite database."""
    tmp_path = db_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    con = sqlite3.connect(tmp_path)
    try:
        con.executescript(SCHEMA_SQL)
        for name, frame in dims.items():
            cols = list(frame.columns)
            con.executemany(
                f"INSERT INTO {name} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                _to_rows(frame),
            )
        cols = list(fact.columns)
        con.executemany(
            f"INSERT INTO fact_sales ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            _to_rows(fact),
        )
        log = dict(log)
        log["built_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        con.executemany(
            "INSERT INTO etl_log (stage, metric, value) VALUES ('etl', ?, ?)",
            [(k, str(v)) for k, v in log.items()],
        )
        con.commit()
    finally:
        con.close()
    os.replace(tmp_path, db_path)
    return log


def run_etl(csv_path=CSV_PATH, db_path=DB_PATH):
    """Run the complete Extract -> Transform -> Load process."""
    t0 = time.time()
    raw = extract(csv_path)
    dims, fact, log = transform(raw)
    log = load(dims, fact, log, db_path)
    log["seconds"] = round(time.time() - t0, 2)
    return log


def ensure_warehouse():
    """Create datawarehouse.db automatically if it does not exist (or is older than the CSV)."""
    needs_build = not os.path.exists(DB_PATH)
    if not needs_build and os.path.exists(CSV_PATH):
        needs_build = os.path.getmtime(CSV_PATH) > os.path.getmtime(DB_PATH)
    if needs_build:
        return run_etl()
    return None


# --------------------------------------------------------------------------
# Queries used by Flask
# --------------------------------------------------------------------------
def _connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def etl_log():
    con = _connect()
    try:
        rows = con.execute("SELECT metric, value FROM etl_log WHERE stage='etl'").fetchall()
    finally:
        con.close()
    out = {}
    for r in rows:
        v = r["value"]
        out[r["metric"]] = int(v) if v.lstrip("-").isdigit() else v
    return out


def summary():
    con = _connect()
    try:
        one = lambda q: con.execute(q).fetchone()[0]
        s = {
            "total_records": one("SELECT COUNT(*) FROM fact_sales"),
            "games": one("SELECT COUNT(*) FROM dim_game"),
            "platforms": one("SELECT COUNT(*) FROM dim_platform"),
            "genres": one("SELECT COUNT(*) FROM dim_genre"),
            "publishers": one("SELECT COUNT(*) FROM dim_publisher"),
            "year_min": one("SELECT MIN(Year) FROM dim_year"),
            "year_max": one("SELECT MAX(Year) FROM dim_year"),
            "year_members": one("SELECT COUNT(*) FROM dim_year"),
            "global_sales": round(one("SELECT COALESCE(SUM(Global_Sales),0) FROM fact_sales"), 2),
            "na_sales": round(one("SELECT COALESCE(SUM(NA_Sales),0) FROM fact_sales"), 2),
            "eu_sales": round(one("SELECT COALESCE(SUM(EU_Sales),0) FROM fact_sales"), 2),
            "jp_sales": round(one("SELECT COALESCE(SUM(JP_Sales),0) FROM fact_sales"), 2),
            "other_sales": round(one("SELECT COALESCE(SUM(Other_Sales),0) FROM fact_sales"), 2),
        }
    finally:
        con.close()
    return s


def filter_options():
    con = _connect()
    try:
        return {
            "platforms": [r[0] for r in con.execute("SELECT Platform FROM dim_platform ORDER BY Platform")],
            "genres": [r[0] for r in con.execute("SELECT Genre FROM dim_genre ORDER BY Genre")],
            "publishers": [r[0] for r in con.execute("SELECT Publisher FROM dim_publisher ORDER BY Publisher")],
            "years": [r[0] for r in con.execute("SELECT Year FROM dim_year WHERE Year IS NOT NULL ORDER BY Year")],
            "has_unknown_year": bool(con.execute("SELECT 1 FROM dim_year WHERE Year IS NULL").fetchone()),
        }
    finally:
        con.close()


def fact_page(page=1, per_page=10, q="", platform="", genre="", publisher="", year=""):
    """Paginated + filtered view of fact_sales joined with its dimensions."""
    if per_page not in ALLOWED_PAGE_SIZES:
        per_page = 10
    where, params = [], []
    if q:
        where.append("g.Name LIKE ?")
        params.append(f"%{q}%")
    if platform:
        where.append("p.Platform = ?")
        params.append(platform)
    if genre:
        where.append("ge.Genre = ?")
        params.append(genre)
    if publisher:
        where.append("pub.Publisher = ?")
        params.append(publisher)
    if year:
        if str(year).lower() == "unknown":
            where.append("y.Year IS NULL")
        else:
            try:
                year_int = int(year)
            except ValueError:
                year_int = None
            if year_int is not None:
                where.append("y.Year = ?")
                params.append(year_int)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    joins = """
        FROM fact_sales f
        JOIN dim_game g       ON g.Game_ID = f.Game_ID
        JOIN dim_platform p   ON p.Platform_ID = f.Platform_ID
        JOIN dim_genre ge     ON ge.Genre_ID = f.Genre_ID
        JOIN dim_publisher pub ON pub.Publisher_ID = f.Publisher_ID
        JOIN dim_year y       ON y.Year_ID = f.Year_ID
    """
    con = _connect()
    try:
        total = con.execute(f"SELECT COUNT(*) {joins} {where_sql}", params).fetchone()[0]
        pages = max(1, -(-total // per_page))
        page = min(max(1, page), pages)
        rows = con.execute(
            f"""SELECT f.Sales_ID, f.Game_ID, g.Name, f.Platform_ID, p.Platform, f.Genre_ID, ge.Genre,
                       f.Publisher_ID, pub.Publisher, f.Year_ID, y.Year,
                       f.NA_Sales, f.EU_Sales, f.JP_Sales, f.Other_Sales, f.Global_Sales
                {joins} {where_sql} ORDER BY f.Sales_ID LIMIT ? OFFSET ?""",
            params + [per_page, (page - 1) * per_page],
        ).fetchall()
    finally:
        con.close()
    return {
        "rows": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "pages": pages,
        "per_page": per_page,
        "start": 0 if total == 0 else (page - 1) * per_page + 1,
        "end": min(page * per_page, total),
    }


def dimension_preview(name, limit=10):
    if name not in DIM_TABLES:
        raise KeyError(name)
    pk, cols = DIM_TABLES[name]
    con = _connect()
    try:
        total = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        rows = con.execute(f"SELECT {', '.join(cols)} FROM {name} ORDER BY {pk} LIMIT ?", (limit,)).fetchall()
    finally:
        con.close()
    return {"table": name, "columns": cols, "rows": [dict(r) for r in rows], "total": total}


# --------------------------------------------------------------------------
# Dataset statistics (computed from the real CSV)
# --------------------------------------------------------------------------
_stats_cache = {"mtime": None, "value": None}


def dataset_stats():
    mtime = os.path.getmtime(CSV_PATH)
    if _stats_cache["mtime"] == mtime:
        return _stats_cache["value"]
    df = extract()
    year = pd.to_numeric(df["Year"], errors="coerce")
    gs = pd.to_numeric(df["Global_Sales"], errors="coerce")
    columns = []
    for c in df.columns:
        columns.append({
            "name": c,
            "dtype": str(df[c].dtype),
            "missing": int(df[c].isna().sum()),
            "unique": int(df[c].nunique(dropna=True)),
        })
    by_genre = (df.assign(g=gs).groupby("Genre")["g"].sum().sort_values(ascending=False))
    by_platform = df["Platform"].value_counts().head(12)
    by_year = df.assign(y=year, g=gs).dropna(subset=["y"]).groupby("y")["g"].sum().sort_index()
    top_games = df.assign(g=gs).sort_values("g", ascending=False).head(10)[["Name", "Platform", "Year", "Global_Sales"]]
    stats = {
        "records": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": columns,
        "year_min": int(year.min()), "year_max": int(year.max()),
        "games": int(df["Name"].nunique()),
        "platforms": int(df["Platform"].nunique()),
        "genres": int(df["Genre"].nunique()),
        "publishers": int(df["Publisher"].nunique()),
        "global_sales_total": round(float(gs.sum()), 2),
        "global_sales": {
            "mean": round(float(gs.mean()), 4), "median": round(float(gs.median()), 4),
            "std": round(float(gs.std()), 4), "min": round(float(gs.min()), 4),
            "max": round(float(gs.max()), 4),
            "q75": round(float(gs.quantile(.75)), 4),
        },
        "region_totals": {c: round(float(pd.to_numeric(df[c], errors="coerce").sum()), 2) for c in REGION_COLS},
        "genre_sales": {"labels": by_genre.index.tolist(), "values": [round(float(v), 2) for v in by_genre.values]},
        "platform_counts": {"labels": by_platform.index.tolist(), "values": [int(v) for v in by_platform.values]},
        "year_sales": {"labels": [int(y) for y in by_year.index], "values": [round(float(v), 2) for v in by_year.values]},
        "top_games": [
            {"Name": r.Name, "Platform": r.Platform,
             "Year": None if pd.isna(r.Year) else int(r.Year), "Global_Sales": float(r.Global_Sales)}
            for r in top_games.itertuples()
        ],
    }
    _stats_cache.update({"mtime": mtime, "value": stats})
    return stats


if __name__ == "__main__":
    info = run_etl()
    print("Data warehouse built:", DB_PATH)
    for k, v in info.items():
        print(f"  {k}: {v}")
