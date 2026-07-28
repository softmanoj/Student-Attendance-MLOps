from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Student Attendance Prediction API"
    }


def test_prediction_endpoint():
    test_input = {
        "previous_attendance": 85,
        "timetable": 32,
        "semester": 5,
        "holidays": 6,
        "internal_exams": 3
    }

    response = client.post(
        "/predict",
        json=test_input
    )

    assert response.status_code == 200

    response_data = response.json()

    assert "Predicted Attendance" in response_data
    assert isinstance(
        response_data["Predicted Attendance"],
        (int, float)
    )

    assert 0 <= response_data["Predicted Attendance"] <= 100


def test_invalid_attendance_input():
    invalid_input = {
        "previous_attendance": 150,
        "timetable": 32,
        "semester": 5,
        "holidays": 6,
        "internal_exams": 3
    }

    response = client.post(
        "/predict",
        json=invalid_input
    )

    assert response.status_code == 422