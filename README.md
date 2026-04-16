# COMP3610_Assignment-4
Here’s a full **README.md** in Markdown format, with headings included, that you can drop straight into your assignment folder:

```markdown
# Taxi Tip Prediction API

A FastAPI service for predicting taxi trip tips using a trained MLflow model.

---

## Setup Instructions

### 1. Navigate to Project Folder
```bash
cd Downloads/COMP3610_Assignment-4
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
```

### 3. Activate Virtual Environment
- **Windows (PowerShell):**
  ```bash
  .venv\Scripts\Activate
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Run Locally

Start the FastAPI server:
```bash
uvicorn app:app --reload
```

Access the Swagger UI:
```
http://127.0.0.1:8000/docs
```

---

## Run Tests

Execute pytest:
```bash
pytest
```

All tests should pass (with possible deprecation warnings).

---

## Docker Instructions

### 1. Build Image
```bash
docker build -t taxi-tip-api .
```

### 2. Check Image Size
```bash
docker images
```

### 3. Run Container
```bash
docker run -d -p 8000:8000 taxi-tip-api
```

### 4. Verify API (outside container)
Using `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_hour": 14,
    "pickup_day_of_week": 3,
    "trip_duration_minutes": 12.5,
    "trip_distance": 3.2,
    "passenger_count": 2,
    "fare_amount": 15.0,
    "pickup_location": "Manhattan",
    "dropoff_location": "Queens"
  }'
```

Expected response:
```json
{
  "model_version": 2,
  "predictions": [
    {
      "prediction_id": "...",
      "tip_amount": 3.99
    }
  ]
}
```

---

## Troubleshooting

- Ensure **Docker Desktop** is running with **WSL2 integration** enabled before building images.
- If builds fail with `EOF` errors:
  - Restart Docker Desktop and WSL (`wsl --shutdown`).
  - Retry with `--progress=plain` or disable BuildKit (`setx DOCKER_BUILDKIT 0`).
- Check proxy settings in Docker Desktop if behind a corporate network.
- Update tests if your model version changes (e.g., from 2 → 1 since the cells were executed twice but version 2 was used).