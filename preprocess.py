from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# File locations
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "student_attendance.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TARGET_COLUMN = "Attendance"


def preprocess_data():
    # 1. Read CSV dataset
    print("Reading dataset...")
    dataset = pd.read_csv(DATA_FILE)

    print("\nOriginal dataset:")
    print(dataset)

    # Check whether the target column exists
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found in the dataset."
        )

    # 2. Split features and target
    X = dataset.drop(columns=[TARGET_COLUMN])
    y = dataset[TARGET_COLUMN]

    print("\nFeature columns:")
    print(list(X.columns))

    print(f"\nTarget column: {TARGET_COLUMN}")

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # 4. Handle missing values
    # Fit only on training data to avoid data leakage
    imputer = SimpleImputer(strategy="median")

    X_train_processed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_processed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Restore original indexes for the target values
    y_train = y_train.loc[X_train_processed.index]
    y_test = y_test.loc[X_test_processed.index]

    # Combine features and target before saving
    train_data = X_train_processed.copy()
    train_data[TARGET_COLUMN] = y_train

    test_data = X_test_processed.copy()
    test_data[TARGET_COLUMN] = y_test

    # 5. Create processed-data directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 6. Save processed data
    train_file = PROCESSED_DIR / "train.csv"
    test_file = PROCESSED_DIR / "test.csv"

    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    print("\nData preprocessing completed successfully.")
    print(f"Training records: {len(train_data)}")
    print(f"Testing records: {len(test_data)}")
    print(f"Training data saved to: {train_file}")
    print(f"Testing data saved to: {test_file}")


if __name__ == "__main__":
    preprocess_data()