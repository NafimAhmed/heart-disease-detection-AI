import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump
import json

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    csv_path = DATA_DIR / "heart.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected {csv_path} to exist.")
    df = pd.read_csv(csv_path)
    if "target" not in df.columns and "num" in df.columns:
        df = df.rename(columns={"num": "target"})
    expected = {'age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df

def main():
    df = load_data()
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    categorical = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]
    numeric = [c for c in X.columns if c not in categorical]

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])

    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_tr, y_tr)
    y_pr = pipe.predict(X_te)
    print("\nClassification report:\n")
    print(classification_report(y_te, y_pr, digits=4))

    dump(pipe, MODEL_DIR / "heart_model.joblib")
    meta = {"model_type": "LogisticRegression","features": X.columns.tolist()}
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved model to model/heart_model.joblib")

if __name__ == "__main__":
    main()