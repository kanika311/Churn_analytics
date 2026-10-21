import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "telco_churn.csv"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def load_clean(keep_id=False):
    """Load raw CSV and clean it. Returns the cleaned DataFrame."""
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)          # new customers, tenure = 0
    df = df.drop_duplicates()
    if not keep_id and "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    # feature engineering
    df["tenure_group"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["num_services"] = (df[addons] == "Yes").sum(axis=1)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    return df


def get_sqlite_engine():
    """Create in-memory SQLite connection loaded with telco customers table."""
    import sqlite3
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    # Load raw with original text churn for exact compatibility with SQL queries
    raw_df = pd.read_csv(DATA)
    raw_df["TotalCharges"] = pd.to_numeric(raw_df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0)
    raw_df.to_sql("customers", conn, if_exists="replace", index=False)
    return conn
