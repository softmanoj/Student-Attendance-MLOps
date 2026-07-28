import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from preprocess import PROPOSED_FEATURES


app = FastAPI(title="Student Attendance Prediction API", version="2.0.0")
model = joblib.load(Path("model/model.pkl"))

API_REQUESTS = Counter(
    "attendance_api_requests_total", "Total API requests", ["endpoint", "status"]
)
PREDICTIONS = Counter(
    "attendance_prediction_requests_total", "Total prediction requests"
)
ERRORS = Counter("attendance_prediction_errors_total", "Prediction errors")
LATENCY = Histogram(
    "attendance_prediction_latency_seconds", "Prediction response time"
)
LAST_PREDICTION = Gauge(
    "attendance_last_prediction_percent", "Most recent attendance prediction"
)

app.mount("/metrics", make_asgi_app())


class AttendanceInput(BaseModel):
    previous_attendance: float = Field(ge=0, le=100)
    timetable: int = Field(ge=1)
    semester: int = Field(ge=1, le=8)
    holidays: int = Field(ge=0)
    internal_exams: int = Field(ge=0)
    classes_conducted: int = Field(gt=0)
    classes_attended: int = Field(ge=0)
    approved_leave_days: int = Field(ge=0)
    assignment_completion_rate: float = Field(ge=0, le=100)
    study_hours_per_week: float = Field(ge=0, le=100)
    travel_distance_km: float = Field(ge=0)


@app.get("/")
def home():
    API_REQUESTS.labels(endpoint="/", status="success").inc()
    return {
        "message": "Student Attendance Prediction API",
        "model": "feature-enhanced-random-forest",
        "version": "2.0.0",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(payload: AttendanceInput):
    started = time.perf_counter()
    try:
        if payload.classes_attended > payload.classes_conducted:
            raise HTTPException(
                status_code=422,
                detail="classes_attended cannot exceed classes_conducted",
            )
        current_rate = (
            payload.classes_attended / payload.classes_conducted
        ) * 100
        values = [[
            payload.previous_attendance,
            payload.timetable,
            payload.semester,
            payload.holidays,
            payload.internal_exams,
            payload.classes_conducted,
            payload.classes_attended,
            current_rate,
            payload.approved_leave_days,
            payload.assignment_completion_rate,
            payload.study_hours_per_week,
            payload.travel_distance_km,
        ]]
        prediction = float(
            model.predict(pd.DataFrame(values, columns=PROPOSED_FEATURES))[0]
        )
        prediction = round(max(0, min(100, prediction)), 2)
        PREDICTIONS.inc()
        LAST_PREDICTION.set(prediction)
        API_REQUESTS.labels(endpoint="/predict", status="success").inc()
        return {
            "Predicted Attendance": prediction,
            "Current Attendance Rate": round(current_rate, 2),
        }
    except HTTPException:
        ERRORS.inc()
        API_REQUESTS.labels(endpoint="/predict", status="error").inc()
        raise
    except Exception as exc:
        ERRORS.inc()
        API_REQUESTS.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    finally:
        LATENCY.observe(time.perf_counter() - started)
