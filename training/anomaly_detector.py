"""
Anomaly Detection — Isolation Forest on member behavior data.

Flags unusual patterns:
  - Members with abnormally high/low calorie burns
  - Extreme workout durations
  - Unusual vital signs during exercise
  - Outlier attendance patterns

Outputs saved to models/artifacts/anomaly_results.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

import yaml
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent


def _load_cfg():
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def detect_anomalies(contamination: float = 0.05) -> dict:
    """
    Run Isolation Forest anomaly detection across gym datasets.
    contamination: expected fraction of outliers (default 5%).
    """
    print("\n═══ Anomaly Detection (Isolation Forest) ═══")
    cfg = _load_cfg()
    fs_dir = ROOT / cfg["data"]["feature_store"]
    art_dir = ROOT / cfg["models"]["artifacts_dir"]
    art_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    total_anomalies = 0
    total_rows = 0

    datasets = [
        ("gym_exercise_features.parquet", "Gym Exercise"),
        ("daily_gym_features.parquet", "Daily Attendance"),
        ("body_perf_features.parquet", "Body Performance"),
        ("calories_exercise_features.parquet", "Calorie Tracking"),
    ]

    for fname, display_name in datasets:
        path = fs_dir / fname
        if not path.exists():
            continue

        df = pd.read_parquet(path)
        exclude = {"member_id", "_source", "_source_x", "_source_y", "workout_type"}
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
        X = df[numeric_cols].fillna(0)

        if len(X) < 20:
            continue

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Isolation Forest
        iso = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
            max_samples="auto",
            n_jobs=-1,
        )
        predictions = iso.fit_predict(X_scaled)
        scores = iso.decision_function(X_scaled)

        anomaly_mask = predictions == -1
        n_anomalies = int(anomaly_mask.sum())
        total_anomalies += n_anomalies
        total_rows += len(df)

        # Extract top anomalies with their most anomalous features
        anomaly_details = []
        if n_anomalies > 0:
            anomaly_indices = np.where(anomaly_mask)[0]
            anomaly_scores = scores[anomaly_mask]
            # Sort by most anomalous (lowest score)
            sorted_idx = np.argsort(anomaly_scores)[:10]  # top 10

            for i in sorted_idx:
                row_idx = anomaly_indices[i]
                row = df.iloc[row_idx]
                # Find which features deviate most from mean
                deviations = {}
                for col in numeric_cols:
                    col_mean = df[col].mean()
                    col_std = df[col].std()
                    if col_std > 0:
                        z = abs((row[col] - col_mean) / col_std)
                        if z > 2:
                            deviations[col] = {
                                "value": round(float(row[col]), 2),
                                "mean": round(float(col_mean), 2),
                                "z_score": round(float(z), 2),
                            }

                # Sort deviations by z_score
                top_deviations = dict(sorted(deviations.items(), key=lambda x: x[1]["z_score"], reverse=True)[:5])

                anomaly_details.append({
                    "row_index": int(row_idx),
                    "anomaly_score": round(float(anomaly_scores[i]), 4),
                    "top_deviations": top_deviations,
                    "description": _describe_anomaly(top_deviations),
                })

        result = {
            "dataset": display_name,
            "total_rows": len(df),
            "anomalies_found": n_anomalies,
            "anomaly_pct": round(n_anomalies / len(df) * 100, 1),
            "top_anomalies": anomaly_details,
        }
        all_results[fname.replace("_features.parquet", "")] = result
        print(f"  {display_name}: {n_anomalies}/{len(df)} anomalies ({result['anomaly_pct']}%)")

    # Save combined results
    output = {
        "total_anomalies": total_anomalies,
        "total_members": total_rows,
        "anomaly_rate": round(total_anomalies / max(total_rows, 1) * 100, 2),
        "datasets": all_results,
    }
    with open(art_dir / "anomaly_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"  ✓ Total: {total_anomalies} anomalies across {total_rows} records ({output['anomaly_rate']}%)")
    return output


def _describe_anomaly(deviations: dict) -> str:
    """Generate a human-readable description of why a data point is anomalous."""
    if not deviations:
        return "Unusual combination of feature values"

    parts = []
    for col, info in list(deviations.items())[:3]:
        col_readable = col.replace("_", " ").title()
        direction = "higher" if info["value"] > info["mean"] else "lower"
        parts.append(f"{col_readable} is {info['z_score']:.1f}σ {direction} than average "
                     f"({info['value']} vs {info['mean']} avg)")

    return ". ".join(parts) + "."


if __name__ == "__main__":
    detect_anomalies()
