from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


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
        "A Machine Learning API that predicts student attendance "
        "using a Random Forest Regressor."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------
if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_FILE}. "
        "Run 'python preprocess.py' and 'python train.py' first."
    )

model = joblib.load(MODEL_FILE)


# --------------------------------------------------
# Request schema
# --------------------------------------------------
class AttendanceInput(BaseModel):
    previous_attendance: float = Field(
        ...,
        ge=0,
        le=100,
        description="Previous attendance percentage"
    )

    timetable: int = Field(
        ...,
        ge=1,
        description="Number of timetable classes"
    )

    semester: int = Field(
        ...,
        ge=1,
        le=8,
        description="Current semester"
    )

    holidays: int = Field(
        ...,
        ge=0,
        description="Number of holidays"
    )

    internal_exams: int = Field(
        ...,
        ge=0,
        description="Number of internal examinations"
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
        # The column names and order must match the training dataset
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

        # Keep attendance within valid percentage limits
        prediction = max(0, min(100, float(prediction)))

        return {
            "Predicted Attendance": round(prediction, 2)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        ) from error