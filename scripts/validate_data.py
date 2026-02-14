"""Data validation — schema checks, null audits, distribution alerts."""
import pandas as pd
import numpy as np
from pathlib import Path
import yaml


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def check_nulls(df: pd.DataFrame, name: str, threshold: float = 0.3) -> list[str]:
    issues = []
    null_pct = df.isnull().mean()
    for col, pct in null_pct.items():
        if pct > threshold:
            issues.append(f"[WARN] {name}.{col}: {pct:.1%} nulls (threshold {threshold:.0%})")
    return issues


def check_duplicates(df: pd.DataFrame, name: str) -> list[str]:
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        return [f"[WARN] {name}: {dup_count} duplicate rows ({dup_count/len(df):.1%})"]
    return []


def check_numeric_ranges(df: pd.DataFrame, name: str) -> list[str]:
    issues = []
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df[c].std() == 0:
            issues.append(f"[WARN] {name}.{c}: zero variance (constant column)")
        if (df[c] < 0).any() and c in ("age", "weight", "height", "calories", "duration"):
            issues.append(f"[WARN] {name}.{c}: contains negative values")
    return issues


def validate_all() -> list[str]:
    cfg = load_config()
    root = Path(__file__).resolve().parent.parent
    fs_dir = root / cfg["data"]["feature_store"]

    all_issues = []
    for pq in sorted(fs_dir.glob("*_features.parquet")):
        df = pd.read_parquet(pq)
        name = pq.stem
        print(f"[VALIDATE] {name}: {df.shape}")

        all_issues.extend(check_nulls(df, name))
        all_issues.extend(check_duplicates(df, name))
        all_issues.extend(check_numeric_ranges(df, name))

    if not all_issues:
        print("[OK] All datasets passed validation")
    else:
        for issue in all_issues:
            print(issue)
        print(f"[SUMMARY] {len(all_issues)} issues found")

    return all_issues


if __name__ == "__main__":
    validate_all()
