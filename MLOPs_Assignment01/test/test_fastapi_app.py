from fastapi.testclient import TestClient
from deployment.fastapi_app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_get_endpoint():
    response = client.get("/predict")
    assert response.status_code == 200
    assert "predictions" in response.json()


def test_predict_post_endpoint():
    payload = {"inputs": [[50, 1, 2, 130, 225, 0, 2, 110, 1, 2.0, 2, 3.0, 3.0]]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "confidence" in data
    assert len(data["predictions"]) == 1
    assert len(data["confidence"]) == 1


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "request_count_total" in body
    assert "model_prediction_count_total" in body
