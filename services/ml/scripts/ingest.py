"""Data ingestion — load raw CSVs, standardize columns, assign member IDs."""
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import hashlib


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def assign_member_id(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if "member_id" in df.columns:
        df["member_id"] = df["member_id"].astype(str)
        return df
    id_col = None
    for c in ["user_id", "id"]:
        if c in df.columns:
            id_col = c
            break
    if id_col:
        df["member_id"] = df[id_col].astype(str)
    else:
        df["member_id"] = [
            hashlib.md5(f"{source}_{i}".encode()).hexdigest()[:12]
            for i in range(len(df))
        ]
    return df


def ingest_all() -> dict[str, pd.DataFrame]:
    cfg = load_config()
    root = Path(__file__).resolve().parent.parent
    frames = {}

    for name, rel_path in cfg["datasets"].items():
        fp = root / rel_path
        if not fp.exists():
            print(f"[WARN] {name}: {fp} not found, skipping")
            continue
        df = pd.read_csv(fp)
        df = standardize_columns(df)
        df = assign_member_id(df, name)
        df["_source"] = name
        frames[name] = df
        print(f"[INGEST] {name}: {len(df)} rows, {len(df.columns)} cols")

    return frames


def save_interim(frames: dict[str, pd.DataFrame]) -> Path:
    cfg = load_config()
    root = Path(__file__).resolve().parent.parent
    interim_dir = root / cfg["data"]["interim"]
    interim_dir.mkdir(parents=True, exist_ok=True)

    for name, df in frames.items():
        out = interim_dir / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"[SAVE] {name} → {out}")

    return interim_dir


if __name__ == "__main__":
    frames = ingest_all()
    save_interim(frames)
    print("[DONE] Ingestion complete")
