"""
Insights Engine — generates natural-language analytics from gym data.

Instead of raw numbers, this produces human-readable narratives like:
  "Your gym's busiest day is Wednesday with 18% of visits.
   Members who train 4+ days/week burn 40% more calories on average."

Also runs live anomaly highlights and trend detection.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import yaml
import joblib


ROOT = Path(__file__).resolve().parent.parent


def _load_cfg():
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def _load_parquet(name: str) -> Optional[pd.DataFrame]:
    cfg = _load_cfg()
    path = ROOT / cfg["data"]["feature_store"] / f"{name}_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def _pct(part, total):
    return f"{part / total * 100:.1f}%" if total > 0 else "N/A"


# Label-encoded → human-readable mappings (from LabelEncoder at feature-eng time)
_WORKOUT_TYPE_MAP = {0: "Cardio", 1: "HIIT", 2: "Strength", 3: "Yoga"}
_GENDER_MAP = {0: "Female", 1: "Male"}
_EXP_MAP = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}


def _decode(val, mapping: dict):
    """Decode a label-encoded value back to its human-readable name."""
    return mapping.get(int(val), str(val)) if pd.notna(val) else str(val)


# ═══════════════════════════════════════════════════════════════
#  Dashboard Insights — top-level summary for admins
# ═══════════════════════════════════════════════════════════════
def generate_dashboard_insights() -> list[dict]:
    """Auto-generate narrative insights for the admin dashboard."""
    insights = []

    # ── Gym Exercise Insights ──
    df = _load_parquet("gym_exercise")
    if df is not None:
        n = len(df)
        age_col = [c for c in df.columns if "age" in c.lower()]
        cal_col = [c for c in df.columns if "calories" in c.lower() and "rate" not in c.lower()]
        wtype_col = [c for c in df.columns if "workout_type" in c.lower()]
        exp_col = [c for c in df.columns if "experience" in c.lower()]

        if wtype_col:
            top_workout = df[wtype_col[0]].value_counts()
            top_name = _decode(top_workout.index[0], _WORKOUT_TYPE_MAP)
            top_pct = top_workout.iloc[0] / n * 100
            insights.append({
                "type": "trend",
                "title": "Most Popular Workout",
                "text": f"**{top_name}** leads with {top_pct:.0f}% of all sessions ({top_workout.iloc[0]}/{n}). "
                        f"Consider adding more {top_name} slots during peak hours.",
                "priority": "medium",
                "category": "workouts",
            })

        if cal_col:
            avg_cal = df[cal_col[0]].mean()
            high_burners = df[df[cal_col[0]] > avg_cal * 1.5]
            insights.append({
                "type": "stat",
                "title": "Calorie Performance",
                "text": f"Average calorie burn is **{avg_cal:.0f} kcal** per session. "
                        f"{len(high_burners)} members ({_pct(len(high_burners), n)}) consistently burn 50%+ above average — "
                        f"these are your power users.",
                "priority": "low",
                "category": "performance",
            })

        if exp_col:
            beginners = df[df[exp_col[0]] == 1] if 1 in df[exp_col[0]].values else pd.DataFrame()
            if len(beginners) > 0:
                insights.append({
                    "type": "action",
                    "title": "Beginner Retention Opportunity",
                    "text": f"**{len(beginners)} beginners** ({_pct(len(beginners), n)}) in your gym. "
                            f"Research shows 50% of beginners quit within 6 months. "
                            f"Consider a guided onboarding program or buddy system.",
                    "priority": "high",
                    "category": "retention",
                })

    # ── Attendance Insights ──
    df = _load_parquet("daily_gym")
    if df is not None:
        n = len(df)
        churn_col = [c for c in df.columns if "churn" in c.lower()]
        adh_col = [c for c in df.columns if "adherence" in c.lower()]
        dow_col = [c for c in df.columns if "day_of_week" in c.lower()]

        if churn_col:
            at_risk = df[df[churn_col[0]] == 1]
            risk_pct = len(at_risk) / n * 100
            severity = "critical" if risk_pct > 30 else "high" if risk_pct > 15 else "medium"
            insights.append({
                "type": "alert",
                "title": "Churn Risk Alert",
                "text": f"**{len(at_risk)} members ({risk_pct:.0f}%)** are flagged as churn risks. "
                        f"These members show declining attendance patterns. "
                        f"Proactive outreach (personal check-in, discount offer) could save "
                        f"~${len(at_risk) * 50}/month in lost revenue.",
                "priority": severity,
                "category": "retention",
            })

        if dow_col:
            day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                         4: "Friday", 5: "Saturday", 6: "Sunday"}
            day_counts = df[dow_col[0]].value_counts().sort_index()
            busiest = day_counts.idxmax()
            quietest = day_counts.idxmin()
            insights.append({
                "type": "trend",
                "title": "Attendance Patterns",
                "text": f"Busiest day: **{day_names.get(int(busiest), busiest)}** ({day_counts.max()} visits). "
                        f"Quietest: **{day_names.get(int(quietest), quietest)}** ({day_counts.min()} visits). "
                        f"Consider off-peak promotions for {day_names.get(int(quietest), quietest)} "
                        f"to balance load.",
                "priority": "low",
                "category": "operations",
            })

    # ── Body Performance Insights ──
    df = _load_parquet("body_perf")
    if df is not None:
        perf_col = [c for c in df.columns if "performance_score" in c.lower()]
        if perf_col:
            avg_perf = df[perf_col[0]].mean()
            top_10 = df[perf_col[0]].quantile(0.9)
            bottom_10 = df[perf_col[0]].quantile(0.1)
            insights.append({
                "type": "stat",
                "title": "Fitness Level Distribution",
                "text": f"Average performance score: **{avg_perf:.2f}**. "
                        f"Top 10% threshold: {top_10:.2f}. Bottom 10%: {bottom_10:.2f}. "
                        f"The gap between top and bottom performers is {top_10 - bottom_10:.2f} points — "
                        f"targeted group classes could help close this gap.",
                "priority": "medium",
                "category": "performance",
            })

    # ── Model Performance ──
    reg_path = ROOT / _load_cfg()["models"]["registry_path"]
    if reg_path.exists():
        with open(reg_path) as f:
            registry = json.load(f)
        model_summaries = []
        for m in registry:
            name = m.get("name", "").replace("_", " ").title()
            metrics = m.get("metrics", {})
            top_metric = max(metrics.items(), key=lambda x: x[1]) if metrics else ("N/A", 0)
            model_summaries.append(f"{name}: {top_metric[0]}={top_metric[1]}")
        if model_summaries:
            insights.append({
                "type": "info",
                "title": "AI Model Performance",
                "text": f"**{len(registry)} ML models** active. " + " | ".join(model_summaries) + ". "
                        f"Models are trained on your gym's real data for maximum accuracy.",
                "priority": "low",
                "category": "ai",
            })

    # ── Anomaly highlights from saved results ──
    anomaly_path = ROOT / "models" / "artifacts" / "anomaly_results.json"
    if anomaly_path.exists():
        with open(anomaly_path) as f:
            anomalies = json.load(f)
        n_anomalies = anomalies.get("total_anomalies", 0)
        n_total = anomalies.get("total_members", 0)
        if n_anomalies > 0:
            insights.append({
                "type": "alert",
                "title": "Behavioral Anomalies Detected",
                "text": f"**{n_anomalies} unusual patterns** detected out of {n_total} members. "
                        f"These could indicate data issues, exceptional members, or concerning behavior changes. "
                        f"Review the anomaly report in the AI Insights panel.",
                "priority": "high" if n_anomalies > n_total * 0.1 else "medium",
                "category": "anomaly",
            })

    # ── Cluster insights from saved results ──
    cluster_path = ROOT / "models" / "artifacts" / "cluster_profiles.json"
    if cluster_path.exists():
        with open(cluster_path) as f:
            clusters = json.load(f)
        if isinstance(clusters, list) and clusters:
            n_clusters = len(clusters)
            largest = max(clusters, key=lambda c: c.get("size", 0))
            insights.append({
                "type": "info",
                "title": "Member Segments Identified",
                "text": f"AI discovered **{n_clusters} distinct member segments** through clustering. "
                        f"Largest segment: \"{largest.get('label', 'Unlabeled')}\" ({largest.get('size', 0)} members). "
                        f"Use these segments for targeted marketing and personalized programming.",
                "priority": "medium",
                "category": "segmentation",
            })

    return insights


# ═══════════════════════════════════════════════════════════════
#  Member-level Insights
# ═══════════════════════════════════════════════════════════════
def generate_member_insight(member_data: dict) -> str:
    """Generate a personalized natural-language insight for one member."""
    parts = []

    age = member_data.get("age")
    weight = member_data.get("weight_kg") or member_data.get("weight")
    bmi = member_data.get("bmi")
    cal = member_data.get("calories_burned")
    exp = member_data.get("experience_level")
    churn = member_data.get("churn_risk")
    perf = member_data.get("performance_score")

    if age:
        parts.append(f"At **{age} years old**, ")
    if exp:
        level = {1: "a beginner", 2: "an intermediate", 3: "an advanced"}.get(int(exp), f"level {exp}")
        parts.append(f"you're {level} gym-goer. ")
    if bmi:
        if bmi < 18.5:
            parts.append(f"Your BMI of {bmi:.1f} is underweight — focus on strength training and calorie surplus. ")
        elif bmi < 25:
            parts.append(f"Your BMI of {bmi:.1f} is in the healthy range — great foundation for any goal. ")
        elif bmi < 30:
            parts.append(f"Your BMI of {bmi:.1f} is slightly elevated — combining cardio with strength training can help. ")
        else:
            parts.append(f"With a BMI of {bmi:.1f}, focus on sustainable fat loss through Zone 2 cardio and nutrition fundamentals. ")

    if cal:
        parts.append(f"You're burning an average of **{cal:.0f} kcal** per session. ")

    if churn is not None:
        if churn:
            parts.append("⚠️ Your attendance has been declining — consistency is key! Even 2 sessions/week maintains progress. ")
        else:
            parts.append("✅ Your attendance is solid — keep up the momentum! ")

    if perf is not None:
        parts.append(f"Your performance score is **{perf:.2f}**. ")

    return "".join(parts) if parts else "Keep showing up — every workout counts! 💪"
