"""
FitFlex Gym Intelligence — FastAPI Serving Layer v2
AI Chatbot (RAG + LLM) · NLG Insights · Anomaly Detection · Clustering + ML Predictions.
"""
import json
import time
import os
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

import yaml
import joblib
import jwt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from serving.schemas import (
    WorkoutRecommendRequest, WorkoutRecommendResponse,
    CaloriePredictRequest, CaloriePredictResponse,
    AdherencePredictRequest, AdherencePredictResponse,
    ProgressPredictRequest, ProgressPredictResponse,
    HealthResponse,
    ChatRequest, ChatResponse,
    InsightsResponse, InsightItem,
    AnomalyResponse,
)

from serving.llm_client import LLMClient, create_client_from_config
from serving.knowledge_base import KnowledgeBase
from serving.chat_engine import ChatEngine, create_chat_engine
from serving.insights_engine import generate_dashboard_insights

# ─── Config ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "configs" / "config.yaml"

# Load environment variables for local development.
# In production (Docker/K8s), prefer real environment variables.
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "Backend" / ".env")


def _load_cfg() -> dict:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


CFG = _load_cfg()
ART_DIR = ROOT / CFG["models"]["artifacts_dir"]

# ─── Global model store ──────────────────────────────────────
_models: dict[str, Any] = {}
_label_encoders: dict[str, Any] = {}
_feature_names: dict[str, list[str]] = {}
_chat_engine: ChatEngine | None = None
_llm_client: LLMClient | None = None


def _try_load(name: str) -> bool:
    """Load a joblib model file if it exists. Return True on success."""
    path = ART_DIR / f"{name}.joblib"
    if path.exists():
        _models[name] = joblib.load(path)
        return True
    return False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load ML models, LLM client, and chat engine at startup."""
    global _chat_engine, _llm_client

    print("\n══ FitFlex AI — Loading models ══")

    # ── ML models ──
    names = [
        "workout_recommender",
        "calorie_predictor",
        "adherence_predictor",
        "progress_forecaster",
    ]
    for n in names:
        ok = _try_load(n)
        print(f"  {'✓' if ok else '✗'} {n}")

    # Optionally load label encoders (for the workout recommender)
    enc_path = ART_DIR / "label_encoders.joblib"
    if enc_path.exists():
        global _label_encoders
        _label_encoders = joblib.load(enc_path)
        print(f"  ✓ label_encoders ({len(_label_encoders)} columns)")

    # Load saved feature-name lists from the registry
    reg_path = ROOT / CFG["models"]["registry_path"]
    if reg_path.exists():
        registry = json.loads(reg_path.read_text())
        for entry in registry:
            name = entry.get("name")
            feats = entry.get("features")
            if name and feats:
                _feature_names[name] = feats

    # ── LLM + Chat Engine ──
    try:
        _llm_client = create_client_from_config(CFG)
        _chat_engine = create_chat_engine(CFG)
        llm_ok = _llm_client.is_available()
        print(f"  {'✓' if llm_ok else '✗'} LLM ({_llm_client.provider}: {_llm_client.model})")
        if _chat_engine and _chat_engine.kb:
            print(f"  ✓ Knowledge base: {len(_chat_engine.kb.documents)} chunks")
    except Exception as e:
        print(f"  ✗ LLM/Chat setup: {e}")
        _llm_client = None
        _chat_engine = None

    print("══ Ready ══\n")
    yield
    _models.clear()


app = FastAPI(
    title="FitFlex Gym Intelligence API",
    version="2.0.0",
    description="AI-powered gym management: chatbot, predictions, insights, anomaly detection",
    lifespan=lifespan,
)

# CORS — configurable; default to dev localhost
_cors_origins = os.getenv("CORS_ORIGIN", "http://localhost:3000").split(",")
_cors_origins = [o.strip() for o in _cors_origins if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Auth (JWT) ───────────────────────────────────────────────
_AUTH_SECRET = os.getenv("AUTH_SECRET", "")
_DISABLE_AUTH = os.getenv("DISABLE_AUTH", "false").lower() in {"1", "true", "yes"}


def _get_current_user(authorization: str | None = Header(default=None)) -> dict:
    """Validate JWT from Authorization: Bearer <token>."""
    if _DISABLE_AUTH:
        return {"id": "dev", "role": "admin"}

    if not _AUTH_SECRET or len(_AUTH_SECRET) < 8:
        raise HTTPException(status_code=503, detail="AUTH_SECRET not configured")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = authorization[7:]
    try:
        return jwt.decode(token, _AUTH_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def _require_role(roles: list[str]):
    allowed = set(roles)

    def _dep(user: dict = Depends(_get_current_user)):
        role = user.get("role")
        if role not in allowed:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user

    return _dep


# ─── Helpers ──────────────────────────────────────────────────
def _ensure_model(name: str):
    if name not in _models:
        raise HTTPException(status_code=503, detail=f"Model '{name}' not loaded. Run training first.")
    return _models[name]


def _align_features(features: dict, model_name: str) -> np.ndarray:
    """Build feature vector aligned to training column order, filling missing with 0."""
    if model_name in _feature_names:
        ordered = _feature_names[model_name]
    else:
        ordered = list(features.keys())
    row = [features.get(c, 0) for c in ordered]
    return np.array([row])


# ─── Health ───────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        models_loaded=list(_models.keys()),
        llm_available=_llm_client.is_available() if _llm_client else False,
        llm_provider=_llm_client.provider if _llm_client else "",
        llm_model=_llm_client.model if _llm_client else "",
        knowledge_base_chunks=len(_chat_engine.kb.documents) if _chat_engine and _chat_engine.kb else 0,
        version="2.0.0",
    )


# ══════════════════════════════════════════════════════════════
#  AI CHATBOT — RAG + LLM
# ══════════════════════════════════════════════════════════════
@app.post("/chat")
def chat(req: ChatRequest, _user: dict = Depends(_get_current_user)):
    """AI chatbot — RAG-powered, supports streaming (SSE) via stream=true."""
    if _chat_engine is None:
        raise HTTPException(503, "Chat engine not initialized. Ensure Ollama is running and knowledge base is built.")

    if req.stream:
        return StreamingResponse(
            _chat_engine.stream_chat(req.message, req.session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = _chat_engine.chat(req.message, req.session_id)
    return ChatResponse(**result)


@app.get("/chat/suggestions")
def chat_suggestions(_user: dict = Depends(_get_current_user)):
    """Contextual conversation starters."""
    if _chat_engine:
        return {"suggestions": _chat_engine.get_suggestions()}
    return {"suggestions": [
        "What workout should a beginner do?",
        "How many calories does a HIIT session burn?",
        "Tell me about this gym's member demographics",
    ]}


@app.delete("/chat/{session_id}")
def clear_chat(session_id: str, _user: dict = Depends(_get_current_user)):
    """Clear conversation history for a session."""
    if _chat_engine:
        _chat_engine.clear_history(session_id)
    return {"ok": True}


# ══════════════════════════════════════════════════════════════
#  NLG INSIGHTS
# ══════════════════════════════════════════════════════════════
@app.get("/insights")
def insights(_user: dict = Depends(_require_role(["admin", "trainer"]))):
    """Auto-generated natural language insights from gym data."""
    from datetime import datetime, timezone
    raw = generate_dashboard_insights()
    return InsightsResponse(
        insights=[InsightItem(**i) for i in raw],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ══════════════════════════════════════════════════════════════
#  CLUSTERS
# ══════════════════════════════════════════════════════════════
@app.get("/clusters")
def get_clusters(_user: dict = Depends(_require_role(["admin", "trainer"]))):
    """Member clustering / segmentation results."""
    path = ART_DIR / "cluster_profiles.json"
    if not path.exists():
        raise HTTPException(404, "Clustering not yet run. Execute `python train.py` first.")
    with open(path) as f:
        profiles = json.load(f)
    return {"clusters": profiles}


# ══════════════════════════════════════════════════════════════
#  ANOMALIES
# ══════════════════════════════════════════════════════════════
@app.get("/anomalies")
def get_anomalies(_user: dict = Depends(_require_role(["admin", "trainer"]))):
    """Anomaly detection results."""
    path = ART_DIR / "anomaly_results.json"
    if not path.exists():
        raise HTTPException(404, "Anomaly detection not yet run. Execute `python train.py` first.")
    with open(path) as f:
        data = json.load(f)
    return AnomalyResponse(**data)


# ─── Model A: Workout Recommender ────────────────────────────
@app.post("/recommend-workout", response_model=WorkoutRecommendResponse)
async def recommend_workout(req: WorkoutRecommendRequest, _user: dict = Depends(_get_current_user)):
    model = _ensure_model("workout_recommender")
    t0 = time.perf_counter()

    bmi = req.bmi if req.bmi is not None else req.weight_kg / (req.height_m ** 2)

    features = {
        "age": req.age,
        "weight_(kg)": req.weight_kg,
        "height_(m)": req.height_m,
        "max_bpm": req.heart_rate_avg + 30,
        "avg_bpm": req.heart_rate_avg,
        "resting_bpm": req.heart_rate_avg - 30,
        "session_duration_(hours)": req.session_duration_hr,
        "fat_percentage": req.fat_percentage or 20.0,
        "water_intake_(liters)": req.water_intake_liters or 2.0,
        "workout_frequency_(days/week)": req.workout_frequency_days_per_week or 3,
        "experience_level": req.experience_level,
        "bmi": bmi,
        "calories_burned": req.calories_burned or 0,
    }

    X = _align_features(features, "workout_recommender")
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    idx = int(np.argmax(proba))

    inference_ms = (time.perf_counter() - t0) * 1000

    return WorkoutRecommendResponse(
        recommended_workout=str(classes[idx]),
        confidence=round(float(proba[idx]), 4),
        all_probabilities={str(c): round(float(p), 4) for c, p in zip(classes, proba)},
        inference_ms=round(inference_ms, 2),
    )


# ─── Model B: Calorie Predictor ──────────────────────────────
@app.post("/predict-calories", response_model=CaloriePredictResponse)
async def predict_calories(req: CaloriePredictRequest, _user: dict = Depends(_get_current_user)):
    model = _ensure_model("calorie_predictor")
    t0 = time.perf_counter()

    features = {
        "age": req.age,
        "weight": req.weight_kg,
        "height": req.height_cm,
        "heart_rate": req.heart_rate,
        "duration": req.duration_min,
        "body_temp": req.body_temp or 37.0,
        "gender_encoded": req.gender_encoded or 0,
        "bmi": req.weight_kg / ((req.height_cm / 100) ** 2),
        "calorie_rate": 0,
        "intensity_index": req.heart_rate * (req.duration_min / 60),
    }

    X = _align_features(features, "calorie_predictor")
    pred = float(model.predict(X)[0])
    inference_ms = (time.perf_counter() - t0) * 1000

    return CaloriePredictResponse(
        predicted_calories=round(pred, 2),
        inference_ms=round(inference_ms, 2),
    )


# ─── Model C: Adherence / Churn Predictor ────────────────────
@app.post("/predict-adherence", response_model=AdherencePredictResponse)
async def predict_adherence(req: AdherencePredictRequest, _user: dict = Depends(_get_current_user)):
    model = _ensure_model("adherence_predictor")
    t0 = time.perf_counter()

    bmi = req.bmi if req.bmi is not None else req.weight_kg / 1.75 ** 2  # fallback if no height

    features = {
        "age": req.age,
        "weight_(kg)": req.weight_kg,
        "daily_calorie_intake": req.daily_calorie_intake,
        "workout_duration_(min)": req.workout_duration_min,
        "workout_intensity": req.workout_intensity,
        "sleep_hours": req.sleep_hours,
        "stress_level": req.stress_level,
        "bmi": bmi,
        "resting_heart_rate": req.resting_heart_rate or 70,
        "vo2_max": req.vo2_max or 40,
    }

    X = _align_features(features, "adherence_predictor")
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    inference_ms = (time.perf_counter() - t0) * 1000

    if proba < 0.3:
        risk = "low"
    elif proba < 0.6:
        risk = "medium"
    else:
        risk = "high"

    return AdherencePredictResponse(
        churn_risk=bool(pred),
        churn_probability=round(proba, 4),
        risk_label=risk,
        inference_ms=round(inference_ms, 2),
    )


# ─── Model D: Progress Forecaster ────────────────────────────
@app.post("/predict-progress", response_model=ProgressPredictResponse)
async def predict_progress(req: ProgressPredictRequest, _user: dict = Depends(_get_current_user)):
    model = _ensure_model("progress_forecaster")
    t0 = time.perf_counter()

    features = {
        "age": req.age,
        "weight(kg)": req.weight_kg,
        "height(cm)": req.height_cm,
        "body_fat_%": req.body_fat_pct or 20.0,
        "systolic": req.systolic_bp or 120,
        "diastolic": req.diastolic_bp or 80,
        "gripforce": req.grip_force or 40,
        "sit_and_bend_forward_cm": req.sit_and_bend_cm or 15,
        "sit-ups_counts": req.sit_ups_count or 30,
        "broad_jump_cm": req.broad_jump_cm or 200,
        "gender_encoded": req.gender_encoded or 0,
        "class_encoded": 0,
    }

    X = _align_features(features, "progress_forecaster")
    pred = float(model.predict(X)[0])
    inference_ms = (time.perf_counter() - t0) * 1000

    return ProgressPredictResponse(
        predicted_performance_score=round(pred, 4),
        inference_ms=round(inference_ms, 2),
    )


# ─── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    host = CFG.get("serving", {}).get("host", "0.0.0.0")
    port = CFG.get("serving", {}).get("port", 8001)
    uvicorn.run("api:app", host=host, port=port, reload=True)
