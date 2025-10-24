
"""Train a baseline salary prediction model with confidence intervals.

Data source: data/silver/unified_salaries (Parquet mirror written by Spark)
Features: currency (categorical), period (categorical)
Target: salary_amount (numeric)

Models:
- Mean estimator: RandomForestRegressor
- Lower/Upper bounds: GradientBoostingRegressor (quantile, alpha=0.1/0.9)
"""
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

ARTIFACT = Path("ml/models/salary_model.pkl")

def load_data() -> pd.DataFrame:
    parquet_dir = Path("data/silver/unified_salaries")
    parquet_file = Path("data/silver/unified_salaries.parquet")
    if parquet_dir.exists():
        df = pd.read_parquet(parquet_dir)
    elif parquet_file.exists():
        df = pd.read_parquet(parquet_file)
    else:
        raise FileNotFoundError("unified_salaries parquet not found. Run Spark ETL.")
    # Basic cleaning
    if "salary_amount" not in df.columns:
        raise ValueError("salary_amount column missing in unified_salaries")
    df = df.copy()
    df["salary_amount"] = pd.to_numeric(df["salary_amount"], errors="coerce")
    df = df.dropna(subset=["salary_amount"]) 
    df = df[df["salary_amount"] > 0]
    # Ensure features exist
    for col in ["currency", "period"]:
        if col not in df.columns:
            df[col] = "UNKNOWN"
    return df[["currency", "period", "salary_amount"]]

def train_models(df: pd.DataFrame):
    X = df[["currency", "period"]]
    y = df["salary_amount"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_features = ["currency", "period"]
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
        remainder="drop",
    )

    mean_est = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    lower_est = GradientBoostingRegressor(loss="quantile", alpha=0.1, random_state=42)
    upper_est = GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=42)

    mean_pipe = Pipeline(steps=[("pre", pre), ("est", mean_est)])
    lower_pipe = Pipeline(steps=[("pre", pre), ("est", lower_est)])
    upper_pipe = Pipeline(steps=[("pre", pre), ("est", upper_est)])

    mean_pipe.fit(X_train, y_train)
    lower_pipe.fit(X_train, y_train)
    upper_pipe.fit(X_train, y_train)

    y_pred = mean_pipe.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return mean_pipe, lower_pipe, upper_pipe, metrics

def save_artifact(mean_pipe, lower_pipe, upper_pipe, metrics):
    payload = {
        "status": "trained",
        "version": 1,
        "trained_at": datetime.utcnow().isoformat(),
        "features": ["currency", "period"],
        "mean_model": mean_pipe,
        "lower_model": lower_pipe,
        "upper_model": upper_pipe,
        "metrics": metrics,
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT, "wb") as f:
        pickle.dump(payload, f)
    print(f"Wrote trained model to {ARTIFACT} with metrics: {metrics}")

def main():
    df = load_data()
    mean_pipe, lower_pipe, upper_pipe, metrics = train_models(df)
    save_artifact(mean_pipe, lower_pipe, upper_pipe, metrics)

if __name__ == "__main__":
    main()
