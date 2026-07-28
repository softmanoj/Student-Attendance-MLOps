from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/student_attendance.csv")
PROCESSED_DIR = Path("data/processed")
TARGET = "Attendance"
BASELINE_FEATURES = [
    "Previous_Attendance",
    "Timetable",
    "Semester",
    "Holidays",
    "Internal_Exams",
]
PROPOSED_FEATURES = BASELINE_FEATURES + [
    "Classes_Conducted",
    "Classes_Attended",
    "Current_Attendance_Rate",
    "Approved_Leave_Days",
    "Assignment_Completion_Rate",
    "Study_Hours_Per_Week",
    "Travel_Distance_Km",
]


def prepare_data():
    frame = pd.read_csv(DATA_PATH)
    required = set(PROPOSED_FEATURES + [TARGET])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    X = frame[PROPOSED_FEATURES].copy()
    y = frame[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=PROPOSED_FEATURES,
        index=X_train.index,
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=PROPOSED_FEATURES,
        index=X_test.index,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train_imputed.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test_imputed.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    joblib.dump(imputer, PROCESSED_DIR / "imputer.pkl")
    return X_train_imputed, X_test_imputed, y_train, y_test


if __name__ == "__main__":
    prepare_data()
    print("Processed train/test files saved in data/processed/")
