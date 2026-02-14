"""Feature engineering — build ML-ready features from preprocessed data."""
import pandas as pd
import numpy as np
from pathlib import Path
import yaml


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ─── Gym exercise features ──────────────────────────────────────
def build_gym_exercise_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Intensity index: avg_bpm / resting_bpm
    if "avg_bpm" in out.columns and "resting_bpm" in out.columns:
        out["intensity_index"] = out["avg_bpm"] / out["resting_bpm"].replace(0, np.nan)
    # Cardio efficiency
    if "max_bpm" in out.columns and "resting_bpm" in out.columns:
        out["cardio_efficiency"] = (out["max_bpm"] - out["resting_bpm"]) / out["max_bpm"].replace(0, np.nan)
    # Session volume proxy
    if "session_duration_hours" in out.columns and "calories_burned" in out.columns:
        out["calorie_rate"] = out["calories_burned"] / out["session_duration_hours"].replace(0, np.nan)
    # BMI category
    if "bmi" in out.columns:
        out["bmi_category"] = pd.cut(out["bmi"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(float)
    # Hydration per kg
    if "water_intake_liters" in out.columns and "weight_kg" in out.columns:
        out["hydration_per_kg"] = out["water_intake_liters"] / out["weight_kg"].replace(0, np.nan)
    return out


# ─── Daily gym attendance features ──────────────────────────────
def build_daily_gym_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Attendance flag
    if "attendance_status" in out.columns:
        out["attended"] = (out["attendance_status"] == 0).astype(int)  # encoded
    # Session efficiency
    if "calories_burned" in out.columns and "workout_duration_minutes" in out.columns:
        out["calorie_rate_per_min"] = out["calories_burned"] / out["workout_duration_minutes"].replace(0, np.nan)

    # Adherence features per member
    if "member_id" in out.columns:
        grp = out.groupby("member_id")
        out["total_visits"] = grp["member_id"].transform("count")
        if "attended" in out.columns:
            out["adherence_score"] = grp["attended"].transform("mean")
            out["missed_session_rate"] = 1 - out["adherence_score"]

    return out


# ─── Body performance features ───────────────────────────────────
def build_body_perf_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "gripforce" in out.columns and "weight_kg" in out.columns:
        out["relative_grip"] = out["gripforce"] / out["weight_kg"].replace(0, np.nan)
    if "sit_ups_counts" in out.columns and "broad_jump_cm" in out.columns:
        out["power_endurance_ratio"] = out["broad_jump_cm"] / out["sit_ups_counts"].replace(0, np.nan)
    if "systolic" in out.columns and "diastolic" in out.columns:
        out["pulse_pressure"] = out["systolic"] - out["diastolic"]
    # Performance score (composite)
    perf_cols = [c for c in ["gripforce", "sit_ups_counts", "broad_jump_cm", "sit_and_bend_forward_cm"] if c in out.columns]
    if perf_cols:
        for c in perf_cols:
            std = out[c].std()
            std = std if std != 0 else 1
            out[f"{c}_z"] = (out[c] - out[c].mean()) / std
        out["performance_score"] = out[[f"{c}_z" for c in perf_cols]].mean(axis=1)
        out.drop(columns=[f"{c}_z" for c in perf_cols], inplace=True)
    return out


# ─── Calories exercise features ──────────────────────────────────
def build_calories_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "duration" in out.columns and "calories" in out.columns:
        out["calorie_rate"] = out["calories"] / out["duration"].replace(0, np.nan)
    if "heart_rate" in out.columns and "body_temp" in out.columns:
        out["exertion_index"] = out["heart_rate"] * out["body_temp"] / 100
    if "weight" in out.columns and "height" in out.columns:
        out["bmi"] = out["weight"] / ((out["height"] / 100) ** 2).replace(0, np.nan)
    return out


# ─── Fitbit daily features ───────────────────────────────────────
def build_fitbit_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    active_cols = [c for c in ["veryactiveminutes", "fairlyactiveminutes", "lightlyactiveminutes"] if c in out.columns]
    if active_cols:
        out["total_active_minutes"] = out[active_cols].sum(axis=1)
    if "totalsteps" in out.columns and "calories" in out.columns:
        out["steps_per_calorie"] = out["totalsteps"] / out["calories"].replace(0, np.nan)
    if "sedentaryminutes" in out.columns and "total_active_minutes" in out.columns:
        out["active_sedentary_ratio"] = out["total_active_minutes"] / out["sedentaryminutes"].replace(0, np.nan)
    # Engagement score
    if "totalsteps" in out.columns:
        out["engagement_score"] = (out["totalsteps"] / 10000).clip(0, 2)
    return out


# ─── Churn risk label synthesis ──────────────────────────────────
def synthesize_churn_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    risk = np.zeros(len(out))
    if "adherence_score" in out.columns:
        risk += (out["adherence_score"] < 0.5).astype(float) * 0.5
    if "total_visits" in out.columns:
        median_visits = out["total_visits"].median()
        risk += (out["total_visits"] < median_visits * 0.5).astype(float) * 0.3
    if "workout_duration_minutes" in out.columns:
        risk += (out["workout_duration_minutes"] < out["workout_duration_minutes"].quantile(0.25)).astype(float) * 0.2
    out["churn_risk"] = (risk >= 0.5).astype(int)
    return out


# ─── Master feature builder ─────────────────────────────────────
def build_all_features(processed: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    featured = {}

    if "gym_exercise" in processed:
        featured["gym_exercise"] = build_gym_exercise_features(processed["gym_exercise"])
        print(f"[FEATURES] gym_exercise: {featured['gym_exercise'].shape}")

    if "daily_gym" in processed:
        df = build_daily_gym_features(processed["daily_gym"])
        df = synthesize_churn_labels(df)
        featured["daily_gym"] = df
        print(f"[FEATURES] daily_gym: {featured['daily_gym'].shape}")

    if "body_perf" in processed:
        featured["body_perf"] = build_body_perf_features(processed["body_perf"])
        print(f"[FEATURES] body_perf: {featured['body_perf'].shape}")

    if "calories_exercise" in processed:
        featured["calories_exercise"] = build_calories_features(processed["calories_exercise"])
        print(f"[FEATURES] calories_exercise: {featured['calories_exercise'].shape}")

    if "fitbit_daily" in processed:
        featured["fitbit_daily"] = build_fitbit_features(processed["fitbit_daily"])
        print(f"[FEATURES] fitbit_daily: {featured['fitbit_daily'].shape}")

    return featured


def save_feature_store(featured: dict[str, pd.DataFrame]) -> Path:
    cfg = load_config()
    root = Path(__file__).resolve().parent.parent
    fs_dir = root / cfg["data"]["feature_store"]
    fs_dir.mkdir(parents=True, exist_ok=True)

    for name, df in featured.items():
        out = fs_dir / f"{name}_features.parquet"
        df.to_parquet(out, index=False)
        print(f"[SAVE] {name} → {out}")

    # Master features: concat all with source tag
    all_dfs = []
    for name, df in featured.items():
        d = df.copy()
        d["_source"] = name
        all_dfs.append(d)

    master = pd.concat(all_dfs, ignore_index=True, sort=False)
    master_path = root / cfg["data"]["master_features"]
    master.to_parquet(master_path, index=False)
    print(f"[SAVE] master → {master_path} ({master.shape})")

    return fs_dir


if __name__ == "__main__":
    from preprocess import run_preprocessing
    processed = run_preprocessing()
    featured = build_all_features(processed)
    save_feature_store(featured)
    print("[DONE] Feature engineering complete")
