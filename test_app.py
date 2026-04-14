# test_app.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")  # Suppress MLflow warnings during tests
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_valid_prediction():
    response = client.post("/predict", json={
        "pickup_hour": 14,
        "pickup_day_of_week": 2,
        "trip_duration_minutes": 12.5,
        "trip_distance": 3.2,
        "passenger_count": 2,
        "fare_amount": 15.0,
        "pickup_location": "Queens",
        "dropoff_location": "Manhattan"
    })
    assert response.status_code == 200
    data = response.json()
    assert "tip_amount" in data
    assert "model_version" in data
    assert "prediction_id" in data

def test_batch_prediction():
    response = client.post("/predict/batch", json=[
        {
            "pickup_hour": 10,
            "pickup_day_of_week": 1,
            "trip_duration_minutes": 20,
            "trip_distance": 5.0,
            "passenger_count": 1,
            "fare_amount": 25.0,
            "pickup_location": "Brooklyn",
            "dropoff_location": "Manhattan"
        },
        {
            "pickup_hour": 22,
            "pickup_day_of_week": 5,
            "trip_duration_minutes": 15,
            "trip_distance": 2.5,
            "passenger_count": 3,
            "fare_amount": 30.0,
            "pickup_location": "Queens",
            "dropoff_location": "Bronx"
        }
    ])
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2

def test_invalid_prediction():
    response = client.post("/predict", json={
        "pickup_hour": 25,  # invalid
        "pickup_day_of_week": 2,
        "trip_duration_minutes": -10,  # invalid
        "trip_distance": 0,  # invalid
        "passenger_count": 0,  # invalid
        "fare_amount": -5,  # invalid
        "pickup_location": "Queens",
        "dropoff_location": "Manhattan"
    })
    assert response.status_code == 422

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_version"] == 1

def test_edge_case_zero_distance():
    response = client.post("/predict", json={
        "pickup_hour": 8,
        "pickup_day_of_week": 0,
        "trip_duration_minutes": 10,
        "trip_distance": 0.0001,  # near zero
        "passenger_count": 2,
        "fare_amount": 9999.99,   # extreme fare
        "pickup_location": "Manhattan",
        "dropoff_location": "Queens"
    })
    assert response.status_code == 200
    data = response.json()
    assert "tip_amount" in data
