from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# File locations
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"

MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "model.pkl"

TARGET_COLUMN = "Attendance"


# --------------------------------------------------
# Model configuration
# --------------------------------------------------
MODEL_NAME = "Student Attendance Random Forest"
NUMBER_OF_TREES = 100
RANDOM_STATE = 42


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

    # 3. Check target column
    if TARGET_COLUMN not in train_data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing from train.csv."
        )

    if TARGET_COLUMN not in test_data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing from test.csv."
        )

    # 4. Separate features and target
    X_train = train_data.drop(columns=[TARGET_COLUMN])
    y_train = train_data[TARGET_COLUMN]

    X_test = test_data.drop(columns=[TARGET_COLUMN])
    y_test = test_data[TARGET_COLUMN]

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Features: {list(X_train.columns)}")

    # 5. Create or select MLflow experiment
    mlflow.set_experiment("Student Attendance Prediction")

    # Every execution inside this block becomes one MLflow run
    with mlflow.start_run(
        run_name=f"RandomForest_{NUMBER_OF_TREES}_Trees"
    ) as run:

        # 6. Create Random Forest model
        model = RandomForestRegressor(
            n_estimators=NUMBER_OF_TREES,
            random_state=RANDOM_STATE
        )

        # 7. Train model
        print("\nTraining Random Forest Regressor...")
        model.fit(X_train, y_train)

        # 8. Generate predictions
        predictions = model.predict(X_test)

        # 9. Evaluate model
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )
        r2 = r2_score(y_test, predictions)

        print("\nModel Evaluation")
        print("----------------")
        print(f"MAE      : {mae:.4f}")
        print(f"RMSE     : {rmse:.4f}")
        print(f"R² Score : {r2:.4f}")

        # 10. Log model information and parameters
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "model_type": "RandomForestRegressor",
            "number_of_trees": NUMBER_OF_TREES,
            "random_state": RANDOM_STATE,
            "training_records": len(X_train),
            "testing_records": len(X_test)
        })

        # 11. Log evaluation metrics
        mlflow.log_metrics({
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2
        })

        # 12. Log the trained model in MLflow
        input_example = X_train.head(3)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="attendance_model",
            input_example=input_example,
            serialization_format="cloudpickle"
        )

        # 13. Save model.pkl locally
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_FILE)

        print(f"\nModel saved at: {MODEL_FILE}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print("Experiment stored successfully in MLflow.")


if __name__ == "__main__":
    train_model()