import pytest
from unittest.mock import patch

def test_health_endpoint(api_client):
    """Verifies that the /health endpoint is online and returns status."""
    response = api_client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert "timestamp" in json_data

def test_root_endpoint(api_client):
    """Verifies welcome response at API root."""
    response = api_client.get("/")
    assert response.status_code == 200
    assert "opsFlow" in response.json()["app"]

def test_predict_endpoint_success(api_client, mock_telemetry_data):
    """Verifies /api/model/predict handles validation and serves classification results."""
    mock_prediction = {
        "prediction": 0,
        "probability": 0.045,
        "model_used": "random_forest",
        "engineered_features": {"temp_diff": 10.0, "power": 60000.0, "wear_torque_ratio": 2.5}
    }
    
    with patch("services.ml_service.ml_service.predict_failure", return_value=mock_prediction):
        response = api_client.post("/api/model/predict", json=mock_telemetry_data)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 0
        assert data["probability"] == 0.045
        assert "engineered_features" in data

def test_model_status_endpoint(api_client):
    """Verifies /api/model/status endpoint returns model summary metadata."""
    mock_status = {
        "run_timestamp": "2026-05-20T07:39:09",
        "best_model": "Random Forest",
        "best_f1": 0.9421,
        "best_roc_auc": 0.9856,
        "best_params": {"n_estimators": 100},
        "top_features": ["Torque [Nm]"],
        "failure_rate_in_dataset": 0.0339
    }
    
    with patch("services.ml_service.ml_service.get_model_status", return_value=mock_status):
        response = api_client.get("/api/model/status")
        assert response.status_code == 200
        data = response.json()
        assert data["best_model"] == "Random Forest"
        assert data["best_f1"] == 0.9421

def test_clear_cache_endpoint(api_client):
    """Verifies caching flush endpoints return success status."""
    response = api_client.post("/api/clear_cache")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
