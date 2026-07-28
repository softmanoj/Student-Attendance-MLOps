"""Generate a reproducible synthetic dataset for the attendance MLOps demo.

This is prototype data, not real student data. The target is generated from
documented relationships plus random noise so the experiment is reproducible.
"""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
ROWS = 2000


def main() -> None:
    rng = np.random.default_rng(SEED)

    previous = np.clip(rng.normal(76, 11, ROWS), 40, 100)
    semester = rng.integers(1, 9, ROWS)
    timetable = rng.integers(28, 41, ROWS)
    holidays = rng.integers(3, 13, ROWS)
    internal_exams = rng.integers(2, 6, ROWS)

    classes_conducted = np.clip(
        timetable * rng.integers(11, 17, ROWS) - holidays - internal_exams * 2,
        180,
        600,
    ).astype(int)

    approved_leave = np.clip(
        rng.poisson(3.0, ROWS) + (previous < 65).astype(int), 0, 18
    )
    assignment_rate = np.clip(
        0.72 * previous + rng.normal(20, 11, ROWS), 35, 100
    )
    study_hours = np.clip(
        rng.normal(13, 5, ROWS) + (assignment_rate - 70) / 18, 2, 32
    )
    travel_distance = np.clip(rng.gamma(2.2, 4.2, ROWS), 0.5, 35)

    # Mid-semester attendance tendency. This is observed only up to a defined
    # prediction cut-off and therefore is not the final target.
    current_rate = np.clip(
        0.56 * previous
        + 0.23 * assignment_rate
        + 0.32 * study_hours
        - 0.75 * approved_leave
        - 0.14 * travel_distance
        + rng.normal(12, 6.5, ROWS),
        35,
        100,
    )
    classes_attended = np.minimum(
        classes_conducted,
        np.rint(classes_conducted * current_rate / 100),
    ).astype(int)
    current_rate = classes_attended / classes_conducted * 100

    # Final attendance is affected by historical, mid-semester, engagement,
    # leave and travel factors, with noise representing unobserved influences.
    final_attendance = np.clip(
        0.22 * previous
        + 0.52 * current_rate
        + 0.10 * assignment_rate
        + 0.22 * study_hours
        - 0.32 * approved_leave
        - 0.07 * travel_distance
        + 10
        + rng.normal(0, 3.2, ROWS),
        35,
        100,
    )

    frame = pd.DataFrame(
        {
            "Previous_Attendance": np.round(previous, 2),
            "Timetable": timetable,
            "Semester": semester,
            "Holidays": holidays,
            "Internal_Exams": internal_exams,
            "Classes_Conducted": classes_conducted,
            "Classes_Attended": classes_attended,
            "Current_Attendance_Rate": np.round(current_rate, 2),
            "Approved_Leave_Days": approved_leave,
            "Assignment_Completion_Rate": np.round(assignment_rate, 2),
            "Study_Hours_Per_Week": np.round(study_hours, 2),
            "Travel_Distance_Km": np.round(travel_distance, 2),
            "Attendance": np.round(final_attendance, 2),
        }
    )

    destination = Path(__file__).resolve().parents[1] / "data" / "student_attendance.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Generated {len(frame)} rows at {destination}")


if __name__ == "__main__":
    main()
