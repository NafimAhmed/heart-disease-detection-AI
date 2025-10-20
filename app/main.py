import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from joblib import load
import pandas as pd
from .schemas import HeartInput

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "heart_model.joblib"
META_PATH = ROOT / "model" / "model_meta.json"

app = FastAPI(title="Heart Disease Predictor", version="1.0.0")

_model = None
_meta = None

def load_artifacts():
    global _model, _meta
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Model not found. Train first.")
        _model = load(MODEL_PATH)
    if _meta is None:
        _meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    load_artifacts()
    return {"model_type": _meta.get("model_type","unknown"), "features": _meta.get("features",[])}

@app.post("/predict")
def predict(payload: HeartInput):
    load_artifacts()
    try:
        df = pd.DataFrame([payload.model_dump()])
        pred = int(_model.predict(df)[0])
        return {"heart_disease": bool(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))