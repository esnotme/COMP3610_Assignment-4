FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files
COPY requirements.txt .
COPY app.py .
COPY mlruns/ ./mlruns/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
