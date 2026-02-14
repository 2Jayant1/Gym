"""
Member Clustering — discovers natural member segments via KMeans.

Outputs:
  - Cluster assignments per member
  - Cluster profiles (centroids → human-readable descriptions)
  - Saved to models/artifacts/cluster_*.joblib
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

import yaml
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parent.parent


def _load_cfg():
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def _auto_label(centroid: dict) -> str:
    """Generate a human-readable label for a cluster based on its centroid values."""
    labels = []

    age = centroid.get("age", 30)
    if age < 25:
        labels.append("Young")
    elif age > 45:
        labels.append("Senior")

    cal = centroid.get("calories_burned", 0) or centroid.get("calories", 0)
    if cal > 800:
        labels.append("High-Burner")
    elif cal < 300:
        labels.append("Light-Activity")

    bmi = centroid.get("bmi", 25)
    if bmi and bmi > 28:
        labels.append("High-BMI")
    elif bmi and bmi < 22:
        labels.append("Lean")

    exp = centroid.get("experience_level", 2)
    if exp and exp >= 2.5:
        labels.append("Experienced")
    elif exp and exp <= 1.5:
        labels.append("Newcomer")

    perf = centroid.get("performance_score", 0)
    if perf and perf > 0.5:
        labels.append("High-Performer")
    elif perf and perf < -0.5:
        labels.append("Developing")

    return " ".join(labels) if labels else "General Member"


def cluster_members(n_clusters: int = 4) -> dict:
    """
    Run KMeans clustering on gym member data.
    Returns cluster profiles and saves artifacts.
    """
    print("\n═══ Member Clustering (KMeans) ═══")
    cfg = _load_cfg()
    fs_dir = ROOT / cfg["data"]["feature_store"]
    art_dir = ROOT / cfg["models"]["artifacts_dir"]
    art_dir.mkdir(parents=True, exist_ok=True)

    # Collect features from all datasets
    all_dfs = []
    for fname in ["gym_exercise_features.parquet", "body_perf_features.parquet"]:
        path = fs_dir / fname
        if path.exists():
            df = pd.read_parquet(path)
            all_dfs.append(df)

    if not all_dfs:
        print("[SKIP] No feature data found for clustering")
        return {}

    # Use the largest dataset
    df = max(all_dfs, key=len)
    print(f"  Using dataset with {len(df)} rows, {len(df.columns)} columns")

    # Select numeric features only
    exclude = {"member_id", "_source", "workout_type"}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    X = df[numeric_cols].fillna(0)

    if len(X) < n_clusters:
        print(f"[SKIP] Not enough data ({len(X)} rows) for {n_clusters} clusters")
        return {}

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal k via silhouette score
    best_k = n_clusters
    best_sil = -1
    for k in range(2, min(8, len(X) // 10 + 1)):
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
        if sil > best_sil:
            best_sil = sil
            best_k = k

    print(f"  Optimal clusters: {best_k} (silhouette: {best_sil:.3f})")

    # Final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X_scaled)

    # Build cluster profiles
    df_labeled = df[numeric_cols].copy()
    df_labeled["cluster"] = labels
    profiles = []

    for cid in range(best_k):
        mask = df_labeled["cluster"] == cid
        cluster_df = df_labeled[mask]
        centroid = {}
        for col in numeric_cols:
            centroid[col] = round(float(cluster_df[col].mean()), 2)

        profile = {
            "cluster_id": cid,
            "size": int(mask.sum()),
            "pct": round(mask.sum() / len(df) * 100, 1),
            "label": _auto_label(centroid),
            "centroid": centroid,
        }
        profiles.append(profile)
        print(f"  Cluster {cid} ({profile['label']}): {profile['size']} members ({profile['pct']}%)")

    # Save artifacts
    joblib.dump(kmeans, art_dir / "kmeans_model.joblib")
    joblib.dump(scaler, art_dir / "kmeans_scaler.joblib")
    joblib.dump(numeric_cols, art_dir / "kmeans_features.joblib")

    with open(art_dir / "cluster_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"  ✓ Saved clustering artifacts to {art_dir}")
    return {"n_clusters": best_k, "silhouette": best_sil, "profiles": profiles}


if __name__ == "__main__":
    cluster_members()
