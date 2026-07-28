import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from preprocess import BASELINE_FEATURES, PROPOSED_FEATURES, prepare_data


MODEL_DIR = Path("model")
REPORT_DIR = Path("reports")


def metrics(model, X_test, y_test):
    prediction = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
        "r2": float(r2_score(y_test, prediction)),
    }


def log_run(name, model, feature_names, scores):
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_name", "RandomForestRegressor")
        mlflow.log_param("feature_count", len(feature_names))
        mlflow.log_param("features", ",".join(feature_names))
        mlflow.log_param("random_state", 42)
        for key, value in model.get_params().items():
            if key in {
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
            }:
                mlflow.log_param(key, value)
        mlflow.log_metrics(scores)
        mlflow.sklearn.log_model(model, artifact_path="model")


def main():
    X_train, X_test, y_train, y_test = prepare_data()
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    mlflow.set_experiment("student-attendance-comparison")

    baseline = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    baseline.fit(X_train[BASELINE_FEATURES], y_train)
    baseline_scores = metrics(baseline, X_test[BASELINE_FEATURES], y_test)
    log_run("ieee-inspired-baseline", baseline, BASELINE_FEATURES, baseline_scores)
    joblib.dump(baseline, MODEL_DIR / "baseline_model.pkl")

    parameter_grid = {
        "n_estimators": [150, 250],
        "max_depth": [10, 18, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.8],
    }
    search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        parameter_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train[PROPOSED_FEATURES], y_train)
    proposed = search.best_estimator_
    proposed_scores = metrics(proposed, X_test[PROPOSED_FEATURES], y_test)
    log_run("proposed-feature-enhanced", proposed, PROPOSED_FEATURES, proposed_scores)

    joblib.dump(proposed, MODEL_DIR / "model.pkl")
    comparison = {
        "baseline": baseline_scores,
        "proposed": proposed_scores,
        "best_parameters": search.best_params_,
        "rmse_improvement_percent": round(
            100 * (baseline_scores["rmse"] - proposed_scores["rmse"])
            / baseline_scores["rmse"],
            2,
        ),
    }
    (REPORT_DIR / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
