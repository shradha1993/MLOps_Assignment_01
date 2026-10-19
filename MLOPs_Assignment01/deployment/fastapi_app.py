# ============================================================
# 🚀 FastAPI app with Prometheus monitoring
# ============================================================

from fastapi import FastAPI, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Create FastAPI instance
app = FastAPI(title="MLOps Assignment API")

# Define Prometheus metrics
REQUEST_COUNT = Counter(
    "request_count_total",
    "Total number of requests received",
    ["endpoint"]
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)

# Example prediction endpoint
@app.get("/predict")
def predict():
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    with REQUEST_LATENCY.labels(endpoint="/predict").time():
        # Example dummy prediction logic
        result = {"predictions": [1]}
    return result

# Prometheus metrics endpoint
@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
 