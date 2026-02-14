"""
train.py — Single-command entry-point for the full ML + AI pipeline.

Usage:
    python train.py              # run everything
    python train.py --step ingest
    python train.py --step preprocess
    python train.py --step features
    python train.py --step validate
    python train.py --step train
    python train.py --step evaluate
    python train.py --step knowledge_base
    python train.py --step cluster
    python train.py --step anomaly
    python train.py --step serve   # start FastAPI server
"""
import argparse
import sys
import os
from pathlib import Path

# Ensure project root is on sys.path so relative imports work
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "serving"))


def step_ingest():
    print("\n╔═══════════════════════════════════════╗")
    print("║    Step 1 / 9 — Data Ingestion        ║")
    print("╚═══════════════════════════════════════╝\n")
    from scripts.ingest import ingest_all, save_interim
    frames = ingest_all()
    save_interim(frames)
    return frames


def step_preprocess():
    print("\n╔═══════════════════════════════════════╗")
    print("║    Step 2 / 9 — Preprocessing         ║")
    print("╚═══════════════════════════════════════╝\n")
    from scripts.preprocess import run_preprocessing
    return run_preprocessing()


def step_features(processed=None):
    print("\n╔═══════════════════════════════════════╗")
    print("║    Step 3 / 9 — Feature Engineering   ║")
    print("╚═══════════════════════════════════════╝\n")
    from scripts.build_features import build_all_features, save_feature_store
    if processed is None:
        from scripts.preprocess import run_preprocessing
        processed = run_preprocessing()
    featured = build_all_features(processed)
    save_feature_store(featured)
    return featured


def step_validate():
    print("\n╔═══════════════════════════════════════╗")
    print("║    Step 4 / 9 — Data Validation       ║")
    print("╚═══════════════════════════════════════╝\n")
    from scripts.validate_data import validate_all
    issues = validate_all()
    if issues:
        print(f"  ⚠ {len(issues)} issues found")
    else:
        print("  ✓ No issues")
    return issues


def step_train():
    print("\n╔═══════════════════════════════════════╗")
    print("║    Step 5 / 9 — Model Training        ║")
    print("╚═══════════════════════════════════════╝\n")
    from training.train_pipeline import train_all
    train_all()


def step_evaluate():
    print("\n╔═══════════════════════════════════════╗")
    print("║    Step 6 / 9 — Model Evaluation      ║")
    print("╚═══════════════════════════════════════╝\n")
    from training.evaluate import evaluate_all
    evaluate_all()


def step_knowledge_base():
    print("\n╔═══════════════════════════════════════╗")
    print("║    Step 7 / 9 — Knowledge Base (RAG)  ║")
    print("╚═══════════════════════════════════════╝\n")
    from serving.knowledge_base import build_knowledge_base
    kb = build_knowledge_base()
    print(f"  ✓ Built {len(kb.documents)} knowledge chunks for AI chatbot")


def step_cluster():
    print("\n╔═══════════════════════════════════════╗")
    print("║    Step 8 / 9 — Member Clustering     ║")
    print("╚═══════════════════════════════════════╝\n")
    from training.cluster_members import cluster_members
    result = cluster_members()
    print(f"  ✓ Found {result['n_clusters']} member segments (silhouette: {result['silhouette']:.3f})")


def step_anomaly():
    print("\n╔═══════════════════════════════════════╗")
    print("║    Step 9 / 9 — Anomaly Detection     ║")
    print("╚═══════════════════════════════════════╝\n")
    from training.anomaly_detector import detect_anomalies
    result = detect_anomalies()
    print(f"  ✓ Detected {result['total_anomalies']} anomalies across {len(result['datasets'])} datasets")


def step_serve():
    print("\n╔═══════════════════════════════════════╗")
    print("║    Starting ML Serving Layer (FastAPI) ║")
    print("╚═══════════════════════════════════════╝\n")
    import yaml
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    host = cfg.get("serving", {}).get("host", "0.0.0.0")
    port = cfg.get("serving", {}).get("port", 8000)
    # Change cwd so uvicorn discovers the module
    os.chdir(ROOT / "serving")
    import uvicorn
    uvicorn.run("api:app", host=host, port=port, reload=True)


STEPS = {
    "ingest": step_ingest,
    "preprocess": step_preprocess,
    "features": step_features,
    "validate": step_validate,
    "train": step_train,
    "evaluate": step_evaluate,
    "knowledge_base": step_knowledge_base,
    "cluster": step_cluster,
    "anomaly": step_anomaly,
    "serve": step_serve,
}


def run_all():
    """Execute the full pipeline: ingest → … → evaluate → knowledge base → cluster → anomaly."""
    step_ingest()
    processed = step_preprocess()
    step_features(processed)
    step_validate()
    step_train()
    step_evaluate()
    step_knowledge_base()
    step_cluster()
    step_anomaly()
    print("\n════════════════════════════════════════")
    print("  ✓ Full AI pipeline complete!")
    print("  Models saved to: models/artifacts/")
    print("  Knowledge base:  models/artifacts/knowledge_base/")
    print("  Clusters:        models/artifacts/cluster_profiles.json")
    print("  Anomalies:       models/artifacts/anomaly_results.json")
    print("  Registry at:     models/registry/model_registry.json")
    print("  Run `python train.py --step serve` to start the API.")
    print("════════════════════════════════════════\n")


def main():
    parser = argparse.ArgumentParser(description="FitFlex Gym Intelligence — ML Pipeline")
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()),
        default=None,
        help="Run a single pipeline step instead of the full pipeline",
    )
    args = parser.parse_args()

    if args.step:
        STEPS[args.step]()
    else:
        run_all()


if __name__ == "__main__":
    main()
