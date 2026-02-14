"""Preprocessing — clean nulls, encode categoricals, scale numerics."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import yaml
import joblib


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for c in num_cols:
        if df[c].isnull().sum() > 0:
            df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        if df[c].isnull().sum() > 0:
            df[c] = df[c].fillna("unknown")

    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    if encoders is None:
        encoders = {}
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    skip = {"member_id", "_source"}

    for c in cat_cols:
        if c in skip:
            continue
        if c not in encoders:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le
        else:
            le = encoders[c]
            known = set(le.classes_)
            df[c] = df[c].astype(str).apply(lambda v: v if v in known else "unknown")
            if "unknown" not in known:
                le.classes_ = np.append(le.classes_, "unknown")
            df[c] = le.transform(df[c].astype(str))

    return df, encoders


def preprocess_gym_exercise(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_nulls(df)
    if "bmi" not in df.columns and "weight_kg" in df.columns and "height_m" in df.columns:
        df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2)
    return df


def preprocess_daily_gym(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_nulls(df)
    if "visit_date" in df.columns:
        df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
        df["day_of_week"] = df["visit_date"].dt.dayofweek
        df["month"] = df["visit_date"].dt.month
    return df


def preprocess_body_perf(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_nulls(df)
    if "body_fat" in df.columns:
        pass
    elif "body_fat_" in df.columns:
        df.rename(columns={"body_fat_": "body_fat_pct"}, inplace=True)
    return df


def preprocess_calories(df: pd.DataFrame, exercise_df: pd.DataFrame) -> pd.DataFrame:
    if "user_id" in df.columns and "user_id" in exercise_df.columns:
        merged = exercise_df.merge(df, on="user_id", how="left")
    else:
        merged = pd.concat([exercise_df, df[["calories"]]], axis=1)
    return clean_nulls(merged)


def preprocess_fitbit(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_nulls(df)
    if "activitydate" in df.columns:
        df["activitydate"] = pd.to_datetime(df["activitydate"], errors="coerce")
        df["day_of_week"] = df["activitydate"].dt.dayofweek
    return df


def run_preprocessing() -> dict[str, pd.DataFrame]:
    cfg = load_config()
    root = Path(__file__).resolve().parent.parent
    interim_dir = root / cfg["data"]["interim"]

    frames = {}
    for pq in interim_dir.glob("*.parquet"):
        frames[pq.stem] = pd.read_parquet(pq)

    if not frames:
        print("[ERROR] No interim data found. Run ingest.py first.")
        return {}

    processed = {}

    if "gym_exercise" in frames:
        processed["gym_exercise"] = preprocess_gym_exercise(frames["gym_exercise"])

    if "daily_gym" in frames:
        processed["daily_gym"] = preprocess_daily_gym(frames["daily_gym"])

    if "body_perf" in frames:
        processed["body_perf"] = preprocess_body_perf(frames["body_perf"])

    if "calories" in frames and "exercise" in frames:
        processed["calories_exercise"] = preprocess_calories(
            frames["calories"], frames["exercise"]
        )
    elif "exercise" in frames:
        processed["calories_exercise"] = clean_nulls(frames["exercise"])

    if "fitbit_daily" in frames:
        processed["fitbit_daily"] = preprocess_fitbit(frames["fitbit_daily"])

    # Encode and save encoders
    encoders = {}
    for name in processed:
        processed[name], encoders = encode_categoricals(processed[name], encoders)
        print(f"[PREPROCESS] {name}: {processed[name].shape}")

    artifacts_dir = root / cfg["models"]["artifacts_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, artifacts_dir / "label_encoders.joblib")
    print(f"[SAVE] Encoders → {artifacts_dir / 'label_encoders.joblib'}")

    return processed


if __name__ == "__main__":
    processed = run_preprocessing()
    print(f"[DONE] Preprocessed {len(processed)} datasets")
