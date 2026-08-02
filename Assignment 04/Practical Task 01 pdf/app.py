#!/usr/bin/env python3
"""
app.py
--------
Practical Task 1 — Model Deployment Simulation.

A minimal FastAPI REST API that serves the trained network intrusion
detection model (from Task 2 / Task 3) for real-time inference.

Endpoint:
    POST /predict
        Body: {"features": {"Flow Duration": 123.4, "Total Fwd Packets": 5, ...}}
        Returns: {"prediction": ..., "confidence": ..., "class_probabilities": {...}, "explanation": ...}

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Interactive docs (Swagger UI) once running:
    http://localhost:8000/docs

Author: Lokesh Kumar
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACT_DIR = Path(__file__).parent / "model_artifacts"

# The 49 raw flow-level feature names, in the exact order the model's
# preprocessing pipeline (imputer -> scaler -> feature selector) was fit on.
# Kept in sync with generate_network_dataset.py's FEATURE_NAMES.
from generate_network_dataset import FEATURE_NAMES as RAW_FEATURE_NAMES

app = FastAPI(
    title="THE ARZENS — Network Intrusion Detection API",
    description="Serves the Assignment 4 Random Forest threat detection model. "
                "Submit raw CICIDS2017-style flow features and receive a "
                "prediction, confidence, per-class probabilities, and a "
                "plain-English explanation of the decision.",
    version="1.0.0",
)

# --- Load model + preprocessing pipeline once at startup ---
_model = joblib.load(ARTIFACT_DIR / "best_model.pkl")
_prep = joblib.load(ARTIFACT_DIR / "preprocessing_pipeline.pkl")
_imputer = _prep["imputer"]
_scaler = _prep["scaler"]
_label_encoder = _prep["label_encoder"]
_selector = _prep["selector"]
with open(ARTIFACT_DIR / "feature_list.txt") as f:
    _selected_features = [line.strip() for line in f if line.strip()]
with open(ARTIFACT_DIR / "model_metadata.json") as f:
    _meta = json.load(f)

_explainer = shap.TreeExplainer(_model)
_classes = list(_label_encoder.classes_)
_benign_idx = _classes.index("BENIGN")


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description=(
            "Raw flow-level features (see /schema for the full list of 49 "
            "expected keys). Missing keys are imputed automatically using "
            "the training pipeline's median imputer, so a partial feature "
            "set is accepted but a full set gives the most reliable result."
        ),
        examples=[{
            "Flow Duration": 850.0, "Total Fwd Packets": 62.0, "Total Backward Packets": 2.0,
            "Fwd Packet Length Mean": 78.0, "Bwd Packet Length Mean": 55.0,
            "Flow Bytes/s": 45000.0, "Flow Packets/s": 900.0,
            "SYN Flag Count": 1.0, "RST Flag Count": 1.0, "ACK Flag Count": 0.0,
        }],
    )


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    class_probabilities: Dict[str, float]
    explanation: str
    inference_time_ms: float
    model_version: str


def _vectorize(features: Dict[str, float]) -> "pd.DataFrame":
    """Build a (1, 49) raw feature row in the fixed training column order,
    filling any unspecified feature with NaN so the imputer handles it
    exactly as it does for missing values seen during training."""
    row = {name: features.get(name, np.nan) for name in RAW_FEATURE_NAMES}
    return pd.DataFrame([row], columns=RAW_FEATURE_NAMES)


def _explain(x_selected: np.ndarray, predicted_idx: int, top_k: int = 3) -> str:
    sv = _explainer.shap_values(x_selected)  # shape (1, n_features, n_classes)
    contrib = sv[0, :, predicted_idx]
    top_idx = np.argsort(np.abs(contrib))[::-1][:top_k]
    parts = []
    for i in top_idx:
        direction = "increased" if contrib[i] > 0 else "decreased"
        parts.append(f"{_selected_features[i]} {direction} the likelihood of this classification (SHAP={contrib[i]:+.3f})")
    predicted_label = _classes[predicted_idx]
    if predicted_label == "BENIGN":
        lead = "Classified as BENIGN because the flow's characteristics resembled normal traffic patterns."
    else:
        lead = f"Flagged as {predicted_label} because:"
    return lead + " " + "; ".join(parts) + "."


@app.get("/")
def root():
    return {
        "service": "THE ARZENS Network Intrusion Detection API",
        "model": _meta["best_model_name"],
        "version": _meta["model_version"],
        "endpoints": {"predict": "POST /predict", "schema": "GET /schema", "health": "GET /health"},
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "classes": _classes}


@app.get("/schema")
def schema():
    return {
        "raw_feature_names": RAW_FEATURE_NAMES,
        "n_raw_features": len(RAW_FEATURE_NAMES),
        "n_selected_features_used_by_model": len(_selected_features),
        "classes": _classes,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    unknown_keys = set(req.features.keys()) - set(RAW_FEATURE_NAMES)
    if unknown_keys:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown feature name(s): {sorted(unknown_keys)}. See GET /schema for valid feature names.",
        )

    t0 = time.time()
    x_raw = _vectorize(req.features)
    x_imputed = _imputer.transform(x_raw)
    x_scaled = _scaler.transform(x_imputed)
    x_selected = _selector.transform(x_scaled) if _selector is not None else x_scaled

    proba = _model.predict_proba(x_selected)[0]
    predicted_idx = int(np.argmax(proba))
    predicted_label = _classes[predicted_idx]
    confidence = float(proba[predicted_idx])

    explanation = _explain(x_selected, predicted_idx)
    elapsed_ms = (time.time() - t0) * 1000

    return PredictResponse(
        prediction=predicted_label,
        confidence=round(confidence, 4),
        class_probabilities={c: round(float(p), 4) for c, p in zip(_classes, proba)},
        explanation=explanation,
        inference_time_ms=round(elapsed_ms, 2),
        model_version=_meta["model_version"],
    )
