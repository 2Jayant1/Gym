"""Model evaluation — load trained models and run detailed metrics."""
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


ROOT = Path(__file__).resolve().parent.parent
CFG = load_config()
SEED = CFG["random_seed"]
ART_DIR = ROOT / CFG["models"]["artifacts_dir"]


def evaluate_classifier(model_file: str, features_file: str, target: str) -> None:
    model = joblib.load(ART_DIR / model_file)
    df = pd.read_parquet(ROOT / CFG["data"]["feature_store"] / features_file)

    exclude = {target, "member_id", "_source"}
    X = df[[c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]].fillna(0)
    y = df[target]

    _, X_test, _, y_test = train_test_split(X, y, test_size=CFG["test_size"], random_state=SEED)
    y_pred = model.predict(X_test)

    print(f"\n── {model_file} ──")
    print(classification_report(y_test, y_pred))


def evaluate_regressor(model_file: str, features_file: str, target: str) -> None:
    model = joblib.load(ART_DIR / model_file)
    df = pd.read_parquet(ROOT / CFG["data"]["feature_store"] / features_file)

    exclude = {target, "member_id", "_source"}
    X = df[[c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]].fillna(0)
    y = df[target].fillna(0)

    _, X_test, _, y_test = train_test_split(X, y, test_size=CFG["test_size"], random_state=SEED)
    y_pred = model.predict(X_test)

    print(f"\n── {model_file} ──")
    print(f"  MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"  R²:   {r2_score(y_test, y_pred):.4f}")


def evaluate_all() -> None:
    evaluate_classifier("workout_recommender.joblib", "gym_exercise_features.parquet", "workout_type")
    evaluate_regressor("calorie_predictor.joblib", "calories_exercise_features.parquet", "calories")
    evaluate_classifier("adherence_predictor.joblib", "daily_gym_features.parquet", "churn_risk")
    evaluate_regressor("progress_forecaster.joblib", "body_perf_features.parquet", "performance_score")


if __name__ == "__main__":
    evaluate_all()
