from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_root():
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_valid_input():
    """Test prediction with valid input."""
    payload = {
        "feature1": 5.1,
        "feature2": 3.5,
        "feature3": 1.4,
        "feature4": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)
    assert 0 <= data["probability"] <= 1


def test_predict_missing_field():
    """Test that missing fields return 422 validation error."""
    payload = {"feature1": 5.1}  # Missing features 2, 3, 4
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_type():
    """Test that wrong data types return 422 validation error."""
    payload = {
        "feature1": "not_a_number",
        "feature2": 3.5,
        "feature3": 1.4,
        "feature4": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422