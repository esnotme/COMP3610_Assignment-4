# app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from uuid import uuid4
from mlflow import pyfunc
import pandas as pd
import numpy as np
import mlflow


# Load model once at startup
MODEL_NAME = "taxi-tip-regressor"
MODEL_VERSION = 2
model = mlflow.pyfunc.load_model("models:/taxi-tip-regressor/2")


# Expected schema (from sklearn feature_names_in_)
EXPECTED_COLS = [
    "pickup_hour", "pickup_day_of_week", "trip_duration_minutes",
    "trip_speed_mph", "log_trip_distance", "fare_per_mile", "fare_per_minute",
    "PU_Bronx", "PU_Brooklyn", "PU_EWR", "PU_Manhattan", "PU_Queens",
    "PU_Staten Island", "PU_Unknown", "DO_Bronx", "DO_Brooklyn", "DO_EWR",
    "DO_Manhattan", "DO_Queens", "DO_Staten Island", "DO_Unknown"
]

app = FastAPI(title="Taxi Tip Prediction API")

# Input schema
class TripFeatures(BaseModel):
    pickup_hour: int = Field(..., ge=0, le=23)
    pickup_day_of_week: int = Field(..., ge=0, le=6)
    trip_duration_minutes: float = Field(..., gt=0)
    trip_distance: float = Field(..., gt=0)
    passenger_count: int = Field(..., ge=1, le=6)
    fare_amount: float = Field(..., gt=0)
    pickup_location: str = Field(..., pattern="^(Bronx|Brooklyn|Queens|Manhattan|EWR|Staten Island|Unknown)$")
    dropoff_location: str = Field(..., pattern="^(Bronx|Brooklyn|Queens|Manhattan|EWR|Staten Island|Unknown)$")

# Helper to build features
def build_features(features: TripFeatures):
    fare_per_mile = features.fare_amount / features.trip_distance
    fare_per_minute = features.fare_amount / features.trip_duration_minutes
    log_trip_distance = np.log1p(features.trip_distance)
    trip_speed_mph = features.trip_distance / (features.trip_duration_minutes / 60)

    data = {
        "pickup_hour": features.pickup_hour,
        "pickup_day_of_week": features.pickup_day_of_week,
        "trip_duration_minutes": features.trip_duration_minutes,
        "fare_per_mile": fare_per_mile,
        "fare_per_minute": fare_per_minute,
        "log_trip_distance": log_trip_distance,
        "trip_speed_mph": trip_speed_mph,
    }

    for loc in ["Bronx","Brooklyn","Queens","Manhattan","EWR","Staten Island","Unknown"]:
        data[f"PU_{loc}"] = (features.pickup_location == loc)
        data[f"DO_{loc}"] = (features.dropoff_location == loc)

    return pd.DataFrame([data])[EXPECTED_COLS]

# Single prediction
@app.post("/predict")
def predict(features: TripFeatures):
    input_df = build_features(features)
    prediction = model.predict(input_df)[0]
    return {
        "prediction_id": str(uuid4()),
        "model_version": MODEL_VERSION,
        "tip_amount": round(float(prediction), 2)
    }

# Batch prediction
@app.post("/predict/batch")
def predict_batch(records: list[TripFeatures]):
    if len(records) > 100:
        return JSONResponse(status_code=400, content={"error": "Maximum 100 records allowed"})
    dfs = [build_features(r) for r in records]
    input_df = pd.concat(dfs, ignore_index=True)
    preds = model.predict(input_df)
    return {
        "model_version": MODEL_VERSION,
        "predictions": [
            {"prediction_id": str(uuid4()), "tip_amount": round(float(p), 2)}
            for p in preds
        ]
    }

# Health check
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION
    }

# Model info
@app.get("/model/info")
def model_info():
    try:
        feature_names = list(model._model_impl.sklearn_model.feature_names_in_)
    except Exception:
        feature_names = []
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "feature_names": feature_names,
        "metrics": {
            "MAE": "N/A",  # replace with actual metrics if logged
            "RMSE": "N/A",
            "R2": "N/A"
        }
    }

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "details": str(type(exc).__name__)}
    )