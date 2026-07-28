from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest
)


# --------------------------------------------------
# File locations
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model" / "model.pkl"


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------
app = FastAPI(
    title="Student Attendance Prediction API",
    description=(
        "Machine Learning API for predicting student attendance "
        "using a Random Forest Regressor."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# Prometheus metrics
# --------------------------------------------------

# Total number of API requests
API_REQUESTS = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"]
)

# API response-time distribution
API_RESPONSE_TIME = Histogram(
    "api_response_time_seconds",
    "API response time in seconds",
    ["method", "endpoint"]
)

# Total number of prediction requests
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["status_code"]
)


# --------------------------------------------------
# Load model
# --------------------------------------------------
if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_FILE}. "
        "Run 'python preprocess.py' and 'python train.py' first."
    )

model = joblib.load(MODEL_FILE)


# --------------------------------------------------
# Request monitoring middleware
# --------------------------------------------------
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = perf_counter()

    try:
        response = await call_next(request)
        status_code = response.status_code

    except Exception:
        status_code = 500

        # Record failed requests before re-raising the error
        if request.url.path != "/metrics":
            API_REQUESTS.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=str(status_code)
            ).inc()

            API_RESPONSE_TIME.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(perf_counter() - start_time)

            if request.url.path == "/predict":
                PREDICTION_REQUESTS.labels(
                    status_code=str(status_code)
                ).inc()

        raise

    # Do not count Prometheus scraping as an application request
    if request.url.path != "/metrics":
        API_REQUESTS.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=str(status_code)
        ).inc()

        API_RESPONSE_TIME.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(perf_counter() - start_time)

        if request.url.path == "/predict":
            PREDICTION_REQUESTS.labels(
                status_code=str(status_code)
            ).inc()

    return response


# --------------------------------------------------
# Request schema
# --------------------------------------------------
class AttendanceInput(BaseModel):
    previous_attendance: float = Field(
        ...,
        ge=0,
        le=100
    )

    timetable: int = Field(
        ...,
        ge=1
    )

    semester: int = Field(
        ...,
        ge=1,
        le=8
    )

    holidays: int = Field(
        ...,
        ge=0
    )

    internal_exams: int = Field(
        ...,
        ge=0
    )


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Student Attendance Prediction API"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def predict_attendance(data: AttendanceInput):
    try:
        input_data = pd.DataFrame([
            {
                "Previous_Attendance": data.previous_attendance,
                "Timetable": data.timetable,
                "Semester": data.semester,
                "Holidays": data.holidays,
                "Internal_Exams": data.internal_exams
            }
        ])

        prediction = model.predict(input_data)[0]
        prediction = max(0, min(100, float(prediction)))

        return {
            "Predicted Attendance": round(prediction, 2)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        ) from error


# --------------------------------------------------
# Prometheus metrics endpoint
# --------------------------------------------------
@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )