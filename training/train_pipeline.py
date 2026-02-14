"""Unified training pipeline — train all 4 models with metric tracking."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import yaml
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, mean_absolute_error, mean_squared_error,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


ROOT = Path(__file__).resolve().parent.parent
CFG = load_config()
SEED = CFG["random_seed"]
TEST_SIZE = CFG["test_size"]


def _numeric_only(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    exclude = set(exclude or [])
    exclude.update({"member_id", "_source"})
    cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    return df[cols]


def _safe_roc_auc(y_true, y_prob, multi_class="ovr"):
    try:
        if y_prob.ndim == 2:
            return roc_auc_score(y_true, y_prob, multi_class=multi_class, average="weighted")
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return 0.0


# ─── Model A: Workout Recommender (LightGBM Classifier) ─────────
def train_workout_recommender() -> dict:
    print("\n═══ Model A: Workout Recommender ═══")
    fs = ROOT / CFG["data"]["feature_store"] / "gym_exercise_features.parquet"
    if not fs.exists():
        print("[SKIP] gym_exercise features not found")
        return {}

    df = pd.read_parquet(fs)
    target = "workout_type"
    if target not in df.columns:
        print(f"[SKIP] Target '{target}' not in data")
        return {}

    features_df = _numeric_only(df, exclude=[target])
    features_df = features_df.fillna(0)
    y = df[target]
    feature_names = list(features_df.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    params = CFG["model_configs"]["workout_recommender"]["params"]
    model = lgb.LGBMClassifier(**params, random_state=SEED, verbose=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = _safe_roc_auc(y_test, y_prob)

    print(f"  F1 (weighted): {f1:.4f}")
    print(f"  ROC-AUC (OVR): {auc:.4f}")

    # Save
    art_dir = ROOT / CFG["models"]["artifacts_dir"]
    art_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, art_dir / "workout_recommender.joblib")

    return {
        "name": "workout_recommender",
        "algorithm": "LightGBM",
        "type": "classifier",
        "metrics": {"f1_weighted": round(f1, 4), "roc_auc_ovr": round(auc, 4)},
        "features": feature_names,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ─── Model B: Calorie Burn Predictor (LightGBM Regressor) ───────
def train_calorie_predictor() -> dict:
    print("\n═══ Model B: Calorie Burn Predictor ═══")
    fs = ROOT / CFG["data"]["feature_store"] / "calories_exercise_features.parquet"
    if not fs.exists():
        print("[SKIP] calories_exercise features not found")
        return {}

    df = pd.read_parquet(fs)
    target = "calories"
    if target not in df.columns:
        print(f"[SKIP] Target '{target}' not in data")
        return {}

    features_df = _numeric_only(df, exclude=[target])
    features_df = features_df.fillna(0)
    y = df[target].fillna(0)
    feature_names = list(features_df.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y, test_size=TEST_SIZE, random_state=SEED
    )

    params = CFG["model_configs"]["calorie_predictor"]["params"]
    model = lgb.LGBMRegressor(**params, random_state=SEED, verbose=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")

    art_dir = ROOT / CFG["models"]["artifacts_dir"]
    joblib.dump(model, art_dir / "calorie_predictor.joblib")

    return {
        "name": "calorie_predictor",
        "algorithm": "LightGBM",
        "type": "regressor",
        "metrics": {"mae": round(mae, 4), "rmse": round(rmse, 4)},
        "features": feature_names,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ─── Model C: Adherence / Churn Predictor (XGBoost Classifier) ──
def train_adherence_predictor() -> dict:
    print("\n═══ Model C: Adherence / Churn Predictor ═══")
    fs = ROOT / CFG["data"]["feature_store"] / "daily_gym_features.parquet"
    if not fs.exists():
        print("[SKIP] daily_gym features not found")
        return {}

    df = pd.read_parquet(fs)
    target = "churn_risk"
    if target not in df.columns:
        print(f"[SKIP] Target '{target}' not in data")
        return {}

    features_df = _numeric_only(df, exclude=[target])
    features_df = features_df.fillna(0)
    y = df[target].fillna(0).astype(int)
    feature_names = list(features_df.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    params = CFG["model_configs"]["adherence_predictor"]["params"]
    model = xgb.XGBClassifier(
        **params, random_state=SEED, eval_metric="logloss", use_label_encoder=False
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc = _safe_roc_auc(y_test, y_prob)

    print(f"  F1:      {f1:.4f}")
    print(f"  ROC-AUC: {auc:.4f}")

    art_dir = ROOT / CFG["models"]["artifacts_dir"]
    joblib.dump(model, art_dir / "adherence_predictor.joblib")

    return {
        "name": "adherence_predictor",
        "algorithm": "XGBoost",
        "type": "classifier",
        "metrics": {"f1": round(f1, 4), "roc_auc": round(auc, 4)},
        "features": feature_names,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ─── Model D: Progress Forecaster (XGBoost Regressor) ───────────
def train_progress_forecaster() -> dict:
    print("\n═══ Model D: Progress Forecaster ═══")
    fs = ROOT / CFG["data"]["feature_store"] / "body_perf_features.parquet"
    if not fs.exists():
        print("[SKIP] body_perf features not found")
        return {}

    df = pd.read_parquet(fs)
    target = "performance_score"
    if target not in df.columns:
        print(f"[SKIP] Target '{target}' not in data")
        return {}

    features_df = _numeric_only(df, exclude=[target])
    features_df = features_df.fillna(0)
    y = df[target].fillna(0)
    feature_names = list(features_df.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y, test_size=TEST_SIZE, random_state=SEED
    )

    params = CFG["model_configs"]["progress_forecaster"]["params"]
    model = xgb.XGBRegressor(**params, random_state=SEED, eval_metric="rmse")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse_val:.4f}")

    art_dir = ROOT / CFG["models"]["artifacts_dir"]
    joblib.dump(model, art_dir / "progress_forecaster.joblib")

    return {
        "name": "progress_forecaster",
        "algorithm": "XGBoost",
        "type": "regressor",
        "metrics": {"mae": round(mae, 4), "rmse": round(rmse_val, 4)},
        "features": feature_names,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ─── Registry update ────────────────────────────────────────────
def update_registry(results: list[dict]) -> None:
    reg_path = ROOT / CFG["models"]["registry_path"]
    reg_path.parent.mkdir(parents=True, exist_ok=True)

    registry = []
    if reg_path.exists():
        registry = json.loads(reg_path.read_text())

    ts = datetime.now(timezone.utc).isoformat()
    for r in results:
        if not r:
            continue
        entry = {
            "version": f"v{len(registry)+1}",
            "trained_at": ts,
            **r,
        }
        registry.append(entry)

    reg_path.write_text(json.dumps(registry, indent=2))
    print(f"\n[REGISTRY] Updated → {reg_path} ({len(registry)} entries)")


# ─── Run all ─────────────────────────────────────────────────────
def train_all() -> list[dict]:
    results = [
        train_workout_recommender(),
        train_calorie_predictor(),
        train_adherence_predictor(),
        train_progress_forecaster(),
    ]
    update_registry(results)
    return results


if __name__ == "__main__":
    train_all()
    print("\n[DONE] All models trained")
