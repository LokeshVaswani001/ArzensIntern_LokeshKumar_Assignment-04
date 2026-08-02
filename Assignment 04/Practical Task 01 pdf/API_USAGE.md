# API Usage — Network Intrusion Detection Model

Practical Task 1 (Bonus Prep, Week 5) — Model Deployment Simulation.

A FastAPI REST API serving the Random Forest model trained in Task 2 and
evaluated in Task 3. Given raw CICIDS2017-style flow features, it returns a
prediction, confidence, per-class probabilities, and a SHAP-based
plain-English explanation.

## Running the API

```bash
cd api
pip install -r ../requirements_ml.txt fastapi uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger docs: **http://localhost:8000/docs**

## Endpoints

### `GET /`
Service info: model name, version, and a list of available endpoints.

### `GET /health`
Liveness check. Returns `{"status": "ok", "model_loaded": true, "classes": [...]}`.

### `GET /schema`
Returns the full list of 49 expected raw feature names (in required order),
the number of features the model actually uses after selection (30), and
the 7 output classes.

### `POST /predict`
**Request body:**
```json
{
  "features": {
    "Flow Duration": 850.0,
    "Total Fwd Packets": 62.0,
    "Total Backward Packets": 2.0,
    "Fwd Packet Length Mean": 78.0,
    "SYN Flag Count": 1.0,
    "RST Flag Count": 1.0
    /* ... any subset of the 49 features from GET /schema.
       Missing features are median-imputed automatically. */
  }
}
```

**Response body:**
```json
{
  "prediction": "DoS",
  "confidence": 1.0,
  "class_probabilities": {
    "BENIGN": 0.0, "Botnet": 0.0, "BruteForce": 0.0,
    "DoS": 1.0, "Infiltration": 0.0, "PortScan": 0.0, "WebAttack": 0.0
  },
  "explanation": "Flagged as DoS because: Max Packet Length increased the likelihood of this classification (SHAP=+0.108); Fwd Packets/s increased the likelihood of this classification (SHAP=+0.106); Total Fwd Packets increased the likelihood of this classification (SHAP=+0.102).",
  "inference_time_ms": 26.05,
  "model_version": "v1.0_20260801"
}
```

**Error responses:**
- `400 Bad Request` — a submitted feature name doesn't match any name in `GET /schema` (e.g. a typo).
- `422 Unprocessable Entity` — the request body is missing the required `"features"` key or has the wrong JSON shape (standard FastAPI/Pydantic validation error).

## Example: curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Flow Duration": 850.0,
      "Total Fwd Packets": 62.0,
      "Total Backward Packets": 2.0,
      "Fwd Packet Length Mean": 78.0,
      "Bwd Packet Length Mean": 55.0,
      "Flow Bytes/s": 45000.0,
      "Flow Packets/s": 900.0,
      "SYN Flag Count": 1.0,
      "RST Flag Count": 1.0,
      "ACK Flag Count": 0.0
    }
  }'
```

## Example: Postman

1. Method: `POST`, URL: `http://localhost:8000/predict`
2. Headers: `Content-Type: application/json`
3. Body → raw → JSON: paste the same `{"features": {...}}` payload as above.
4. Send — response appears in the body panel with `prediction`, `confidence`, `class_probabilities`, and `explanation`.

## Verified test results (this submission)

The API was started locally and tested end-to-end with real curl requests
against real flow records sampled from `network_traffic_dataset.csv`:

| Sample class (ground truth) | API prediction | Confidence | Inference time |
|---|---|---|---|
| BENIGN | BENIGN | 0.9996 | 53.98 ms |
| DoS | DoS | 1.0000 | 49.21 ms |
| PortScan | PortScan | 1.0000 | 49.69 ms |

(Inference time here includes the per-request SHAP explanation computation;
the raw model prediction alone — as measured in Task 3's production
readiness check — is ~6ms/sample. SHAP adds overhead but is included by
default so every response is explainable out of the box.)

A malformed request (missing the required `"features"` key) correctly
returned `HTTP 422` with a descriptive validation error, and a request
containing an unrecognized feature name correctly returned `HTTP 400`.

## Notes

- This is a **simulation/demo deployment** for the assignment, run with
  Uvicorn's development server. A production deployment would run behind a
  process manager (e.g. Gunicorn with Uvicorn workers) and a reverse proxy,
  with authentication, rate limiting, and structured logging added.
- The API loads the model and preprocessing pipeline once at startup (not
  per-request) for efficiency.
- SHAP explanations are computed per-request using `shap.TreeExplainer`,
  which is fast for tree-based models like the Random Forest used here but
  would need a different (and typically slower) explainer if the neural
  network candidate were deployed instead.
