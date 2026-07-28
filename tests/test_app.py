from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Student Attendance Prediction API" in response.json()["message"]


def test_prediction_range():
    response = client.post(
        "/predict",
        json={
            "previous_attendance": 85,
            "timetable": 32,
            "semester": 5,
            "holidays": 6,
            "internal_exams": 3,
            "classes_conducted": 320,
            "classes_attended": 275,
            "approved_leave_days": 2,
            "assignment_completion_rate": 90,
            "study_hours_per_week": 16,
            "travel_distance_km": 7
        },
    )
    assert response.status_code == 200
    assert 0 <= response.json()["Predicted Attendance"] <= 100
