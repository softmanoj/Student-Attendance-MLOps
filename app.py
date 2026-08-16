import json
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from preprocess import PROPOSED_FEATURES


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
COMPARISON_REPORT_PATH = BASE_DIR / "reports" / "model_comparison.json"

app = FastAPI(
    title="Student Attendance Prediction API",
    version="2.2.0",
    description=(
        "Predict student attendance and compare the IEEE-inspired baseline "
        "with the proposed feature-enhanced model."
    ),
)
model = joblib.load(MODEL_PATH)

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
        "version": "2.2.0",
        "model_metrics": "/model-metrics",
        "swagger_ui": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get(
    "/model-metrics",
    tags=["Model Evaluation"],
    summary="Compare baseline and proposed model performance",
    description=(
        "Returns MAE, RMSE and R² values produced by train.py. Because this "
        "is a regression problem, R² percentage is reported instead of "
        "classification accuracy."
    ),
)
def model_metrics() -> dict[str, Any]:
    if not COMPARISON_REPORT_PATH.exists():
        API_REQUESTS.labels(endpoint="/model-metrics", status="error").inc()
        raise HTTPException(
            status_code=503,
            detail="Model comparison report not found. Run: python train.py",
        )

    try:
        report = json.loads(COMPARISON_REPORT_PATH.read_text(encoding="utf-8"))
        baseline = report["baseline"]
        proposed = report["proposed"]

        mae_improvement = (
            100 * (baseline["mae"] - proposed["mae"]) / baseline["mae"]
        )
        rmse_improvement = report.get(
            "rmse_improvement_percent",
            100 * (baseline["rmse"] - proposed["rmse"]) / baseline["rmse"],
        )

        result = {
            "problem_type": "Regression",
            "accuracy_explanation": (
                "Classification accuracy is not applicable. R² is shown as "
                "a percentage of attendance variation explained by the model."
            ),
            "baseline_model": {
                "name": "IEEE-inspired baseline Random Forest",
                "mae": round(baseline["mae"], 4),
                "rmse": round(baseline["rmse"], 4),
                "r2_score": round(baseline["r2"], 4),
                "r2_percentage": round(baseline["r2"] * 100, 2),
            },
            "proposed_model": {
                "name": "Feature-enhanced tuned Random Forest",
                "mae": round(proposed["mae"], 4),
                "rmse": round(proposed["rmse"], 4),
                "r2_score": round(proposed["r2"], 4),
                "r2_percentage": round(proposed["r2"] * 100, 2),
            },
            "improvement": {
                "mae_reduction_percentage": round(mae_improvement, 2),
                "rmse_reduction_percentage": round(rmse_improvement, 2),
                "r2_increase_percentage_points": round(
                    (proposed["r2"] - baseline["r2"]) * 100, 2
                ),
            },
            "best_parameters": report.get("best_parameters", {}),
            "examiner_summary": (
                f"The proposed model explains {proposed['r2'] * 100:.2f}% of "
                f"the variation in attendance, compared with "
                f"{baseline['r2'] * 100:.2f}% for the baseline. Its RMSE is "
                f"reduced by {rmse_improvement:.2f}%."
            ),
        }
        API_REQUESTS.labels(endpoint="/model-metrics", status="success").inc()
        return result
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        API_REQUESTS.labels(endpoint="/model-metrics", status="error").inc()
        raise HTTPException(
            status_code=500,
            detail="Model comparison report is invalid. Run: python train.py",
        ) from exc


@app.post(
    "/predict",
    tags=["Attendance Prediction"],
    summary="Predict final semester attendance",
    description=(
        "Uses the student's current and historical information to forecast "
        "final attendance and assign a shortage-risk category."
    ),
    responses={
        200: {
            "description": "Final-attendance prediction completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "Current Attendance Percentage": 82.0,
                        "Predicted Final Attendance Percentage": 83.12,
                        "Required Attendance Percentage": 75.0,
                        "Risk Status": "Low Risk",
                    }
                }
            },
        }
    },
)
def predict(payload: AttendanceInput) -> dict[str, float | str]:
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
        current_rate = round(current_rate, 2)
        required_attendance = 75.0

        if prediction < required_attendance:
            risk_status = "High Risk"
        elif prediction <= 80:
            risk_status = "Moderate Risk"
        else:
            risk_status = "Low Risk"

        PREDICTIONS.inc()
        LAST_PREDICTION.set(prediction)
        API_REQUESTS.labels(endpoint="/predict", status="success").inc()
        return {
            "Current Attendance Percentage": current_rate,
            "Predicted Final Attendance Percentage": prediction,
            "Required Attendance Percentage": required_attendance,
            "Risk Status": risk_status,
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
