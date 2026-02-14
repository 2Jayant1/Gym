"""Pydantic schemas for the FastAPI ML serving layer."""
from pydantic import BaseModel, Field
from typing import Optional


# ─── Model A: Workout Recommender ────────────────────────────────
class WorkoutRecommendRequest(BaseModel):
    age: float = Field(..., ge=10, le=100, description="Member age")
    weight_kg: float = Field(..., gt=0, description="Body weight in kg")
    height_m: float = Field(..., gt=0, description="Height in metres")
    heart_rate_avg: float = Field(120.0, ge=40, le=220, description="Average heart rate during exercise")
    session_duration_hr: float = Field(1.0, gt=0, description="Planned session duration in hours")
    experience_level: int = Field(1, ge=1, le=3, description="1=Beginner, 2=Intermediate, 3=Advanced")
    bmi: Optional[float] = Field(None, description="BMI — auto-calculated if omitted")
    fat_percentage: Optional[float] = Field(None, ge=0, le=80)
    water_intake_liters: Optional[float] = Field(2.0, ge=0)
    workout_frequency_days_per_week: Optional[int] = Field(3, ge=0, le=7)
    calories_burned: Optional[float] = Field(None, ge=0)


class WorkoutRecommendResponse(BaseModel):
    recommended_workout: str
    confidence: float = Field(..., ge=0, le=1)
    all_probabilities: dict[str, float]
    inference_ms: float


# ─── Model B: Calorie Predictor ──────────────────────────────────
class CaloriePredictRequest(BaseModel):
    age: float = Field(..., ge=10, le=100)
    weight_kg: float = Field(..., gt=0)
    height_cm: float = Field(..., gt=0)
    heart_rate: float = Field(120.0, ge=40, le=220)
    duration_min: float = Field(30.0, gt=0, description="Exercise duration in minutes")
    body_temp: Optional[float] = Field(37.0, ge=34, le=42, description="Body temperature °C")
    gender_encoded: Optional[int] = Field(0, description="0=Male, 1=Female (label‑encoded)")


class CaloriePredictResponse(BaseModel):
    predicted_calories: float
    inference_ms: float


# ─── Model C: Adherence / Churn Predictor ────────────────────────
class AdherencePredictRequest(BaseModel):
    age: float = Field(..., ge=10, le=100)
    weight_kg: float = Field(..., gt=0)
    daily_calorie_intake: float = Field(2000.0, gt=0)
    workout_duration_min: float = Field(60.0, ge=0)
    workout_intensity: float = Field(5.0, ge=0, le=10, description="1‑10 scale")
    sleep_hours: float = Field(7.0, ge=0, le=24)
    stress_level: float = Field(5.0, ge=0, le=10)
    bmi: Optional[float] = Field(None)
    resting_heart_rate: Optional[float] = Field(None, ge=30, le=120)
    vo2_max: Optional[float] = Field(None, ge=10, le=90)


class AdherencePredictResponse(BaseModel):
    churn_risk: bool
    churn_probability: float = Field(..., ge=0, le=1)
    risk_label: str = Field(..., description="low / medium / high")
    inference_ms: float


# ─── Model D: Progress Forecaster ────────────────────────────────
class ProgressPredictRequest(BaseModel):
    age: float = Field(..., ge=10, le=100)
    weight_kg: float = Field(..., gt=0)
    height_cm: float = Field(..., gt=0)
    body_fat_pct: Optional[float] = Field(None, ge=0, le=80)
    systolic_bp: Optional[float] = Field(120, ge=70, le=200)
    diastolic_bp: Optional[float] = Field(80, ge=40, le=130)
    grip_force: Optional[float] = Field(None, ge=0)
    sit_and_bend_cm: Optional[float] = Field(None)
    sit_ups_count: Optional[float] = Field(None, ge=0)
    broad_jump_cm: Optional[float] = Field(None, ge=0)
    gender_encoded: Optional[int] = Field(0)


class ProgressPredictResponse(BaseModel):
    predicted_performance_score: float
    inference_ms: float


# ─── Generic health ──────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    llm_available: bool = False
    llm_provider: str = ""
    llm_model: str = ""
    knowledge_base_chunks: int = 0
    version: str


# ─── Chat ────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: str = Field("default", description="Conversation session ID")
    stream: bool = Field(False, description="Stream response via SSE")


class ChatResponse(BaseModel):
    response: str
    context_used: bool
    chunks_retrieved: int
    inference_ms: float
    model: str
    provider: str


# ─── Insights ────────────────────────────────────────────────────
class InsightItem(BaseModel):
    type: str
    title: str
    text: str
    priority: str
    category: str


class InsightsResponse(BaseModel):
    insights: list[InsightItem]
    generated_at: str


# ─── Clusters ────────────────────────────────────────────────────
class ClusterProfile(BaseModel):
    cluster_id: int
    size: int
    pct: float
    label: str
    centroid: dict[str, float]


# ─── Anomalies ───────────────────────────────────────────────────
class AnomalyResponse(BaseModel):
    total_anomalies: int
    total_members: int
    anomaly_rate: float
    datasets: dict

