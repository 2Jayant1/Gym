"""
Monitoring — inference logger + prediction distribution tracker.
Stores structured JSON logs for every prediction call.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "configs" / "config.yaml"


def _load_cfg() -> dict:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


CFG = _load_cfg()
LOG_DIR = ROOT / CFG.get("monitoring", {}).get("log_dir", "monitoring/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

PRED_LOG = LOG_DIR / "predictions.jsonl"
ERROR_LOG = LOG_DIR / "errors.jsonl"
PERF_LOG = LOG_DIR / "performance.jsonl"


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ─── Inference Logger ─────────────────────────────────────────
def log_prediction(
    model_name: str,
    inputs: dict[str, Any],
    output: dict[str, Any],
    inference_ms: float,
    member_id: Optional[str] = None,
) -> None:
    """Log a single prediction for drift / audit."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "member_id": member_id,
        "inputs": inputs,
        "output": output,
        "inference_ms": round(inference_ms, 3),
    }
    _append_jsonl(PRED_LOG, record)


# ─── Error Logger ────────────────────────────────────────────
def log_error(
    model_name: str,
    error: str,
    inputs: Optional[dict] = None,
) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "error": error,
        "inputs": inputs,
    }
    _append_jsonl(ERROR_LOG, record)


# ─── Performance Timer ───────────────────────────────────────
class InferenceTimer:
    """Context manager that times a block and writes to performance log."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.start: float = 0
        self.elapsed_ms: float = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": self.model_name,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "error": str(exc_val) if exc_val else None,
        }
        _append_jsonl(PERF_LOG, record)
        return False  # don't suppress exceptions


# ─── Summary / Stats ─────────────────────────────────────────
def prediction_summary(model_name: Optional[str] = None, last_n: int = 500) -> dict:
    """
    Return basic stats over the last N predictions (avg latency, count, etc.)
    Reads from the JSONL log — intentionally simple and file‑based.
    """
    if not PRED_LOG.exists():
        return {"total": 0}

    records = []
    with open(PRED_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if model_name and r.get("model") != model_name:
                    continue
                records.append(r)
            except json.JSONDecodeError:
                continue

    records = records[-last_n:]
    if not records:
        return {"total": 0, "model": model_name}

    latencies = [r["inference_ms"] for r in records if "inference_ms" in r]
    return {
        "model": model_name or "all",
        "total": len(records),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 2),
        "max_latency_ms": round(max(latencies), 2) if latencies else 0,
        "first": records[0].get("ts"),
        "last": records[-1].get("ts"),
    }
