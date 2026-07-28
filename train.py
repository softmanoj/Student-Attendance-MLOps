from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# File locations
BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "model.pkl"

TARGET_COLUMN = "Attendance"


def train_model():
    # 1. Check processed datasets
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            "Processed datasets were not found. "
            "Run 'python preprocess.py' first."
        )

    # 2. Read processed datasets
    print("Reading processed datasets...")

    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)

    # Check target column
    if TARGET_COLUMN not in train_data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing from train.csv."
        )

    if TARGET_COLUMN not in test_data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing from test.csv."
        )

    # 3. Separate features and target
    X_train = train_data.drop(columns=[TARGET_COLUMN])
    y_train = train_data[TARGET_COLUMN]

    X_test = test_data.drop(columns=[TARGET_COLUMN])
    y_test = test_data[TARGET_COLUMN]

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Features: {list(X_train.columns)}")

    # 4. Create Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # 5. Train the model
    print("\nTraining Random Forest Regressor...")
    model.fit(X_train, y_train)

    # 6. Generate predictions
    predictions = model.predict(X_test)

    # 7. Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\nModel Evaluation")
    print("----------------")
    print(f"MAE       : {mae:.2f}")
    print(f"RMSE      : {rmse:.2f}")

    # R² requires at least two testing records
    if len(y_test) >= 2:
        r2 = r2_score(y_test, predictions)
        print(f"R² Score  : {r2:.4f}")
    else:
        print("R² Score  : Cannot be calculated with only one test record.")

    # Display actual and predicted values
    comparison = pd.DataFrame({
        "Actual_Attendance": y_test.to_numpy(),
        "Predicted_Attendance": predictions
    })

    print("\nActual and predicted values:")
    print(comparison)

    # 8. Save the trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved successfully at: {MODEL_FILE}")


if __name__ == "__main__":
    train_model()