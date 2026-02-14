"""
Knowledge Base — builds a TF-IDF vector store from all gym datasets.

This is the "memory" the AI chatbot draws from. Instead of hallucinating,
the LLM retrieves real statistics and context from this store (RAG).

No GPU / PyTorch needed. Pure scikit-learn.
"""
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import yaml
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "models" / "artifacts" / "knowledge_base"


# ═══════════════════════════════════════════════════════════════
#  Vector Store
# ═══════════════════════════════════════════════════════════════
class KnowledgeBase:
    """TF-IDF document store with cosine-similarity retrieval."""

    def __init__(self):
        self.documents: list[str] = []
        self.metadata: list[dict] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

    def add(self, text: str, category: str = "general", source: str = ""):
        self.documents.append(text)
        self.metadata.append({"category": category, "source": source})

    def build_index(self):
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def query(self, question: str, top_k: int = 8) -> list[dict]:
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idx:
            if scores[i] > 0.02:
                results.append({
                    "text": self.documents[i],
                    "score": round(float(scores[i]), 4),
                    "metadata": self.metadata[i],
                })
        return results

    def save(self, path: Path = KB_DIR):
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path / "tfidf_vectorizer.joblib")
        joblib.dump(self.tfidf_matrix, path / "tfidf_matrix.joblib")
        with open(path / "documents.json", "w", encoding="utf-8") as f:
            json.dump({"documents": self.documents, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, path: Path = KB_DIR) -> "KnowledgeBase":
        kb = cls()
        kb.vectorizer = joblib.load(path / "tfidf_vectorizer.joblib")
        kb.tfidf_matrix = joblib.load(path / "tfidf_matrix.joblib")
        with open(path / "documents.json", encoding="utf-8") as f:
            data = json.load(f)
            kb.documents = data["documents"]
            kb.metadata = data["metadata"]
        return kb


# ═══════════════════════════════════════════════════════════════
#  Chunk generators — turn raw data into searchable text
# ═══════════════════════════════════════════════════════════════
def _col(df, *candidates):
    """Return the first matching column name (handles naming variations)."""
    for c in candidates:
        if c in df.columns:
            return c
        # fuzzy: ignore underscores and parens
        clean = re.sub(r"[^a-z0-9]", "", c.lower())
        for col in df.columns:
            if re.sub(r"[^a-z0-9]", "", col.lower()) == clean:
                return col
    return None


def _safe_stat(series, stat="mean"):
    try:
        v = getattr(series.dropna(), stat)()
        return f"{v:.1f}"
    except Exception:
        return "N/A"


def _add_gym_exercise_chunks(kb: KnowledgeBase, df: pd.DataFrame):
    """Generate rich text chunks from the gym exercise dataset."""
    n = len(df)
    cat = "gym_exercise"

    # Overall summary
    age_col = _col(df, "age")
    wt_col = _col(df, "weight_(kg)", "weight_kg", "weight")
    ht_col = _col(df, "height_(m)", "height_m", "height")
    cal_col = _col(df, "calories_burned")
    dur_col = _col(df, "session_duration_(hours)", "session_duration_hours")
    wtype_col = _col(df, "workout_type")
    bmi_col = _col(df, "bmi")
    fat_col = _col(df, "fat_percentage")
    exp_col = _col(df, "experience_level")
    hr_col = _col(df, "avg_bpm")
    water_col = _col(df, "water_intake_(liters)", "water_intake_liters")
    freq_col = _col(df, "workout_frequency_(days/week)", "workout_frequency_days_week")

    kb.add(
        f"The gym exercise tracking dataset contains {n} recorded workout sessions from gym members. "
        f"This data captures demographics, biometrics, and workout performance for each session. "
        f"{'Member ages range from ' + _safe_stat(df[age_col], 'min') + ' to ' + _safe_stat(df[age_col], 'max') + ' years (average ' + _safe_stat(df[age_col]) + ').' if age_col else ''} "
        f"{'Average body weight is ' + _safe_stat(df[wt_col]) + ' kg.' if wt_col else ''} "
        f"{'Average BMI is ' + _safe_stat(df[bmi_col]) + '.' if bmi_col else ''}",
        cat, "overview",
    )

    # Per-workout-type breakdown
    if wtype_col:
        for wtype, grp in df.groupby(wtype_col):
            parts = [f"Workout type '{wtype}': {len(grp)} sessions ({len(grp)/n*100:.1f}% of total)."]
            if cal_col:
                parts.append(f"Average calories burned: {_safe_stat(grp[cal_col])} kcal (range {_safe_stat(grp[cal_col],'min')}-{_safe_stat(grp[cal_col],'max')}).")
            if dur_col:
                parts.append(f"Average session duration: {_safe_stat(grp[dur_col])} hours.")
            if hr_col:
                parts.append(f"Average heart rate: {_safe_stat(grp[hr_col])} BPM.")
            if age_col:
                parts.append(f"Average member age: {_safe_stat(grp[age_col])} years.")
            if exp_col:
                parts.append(f"Average experience level: {_safe_stat(grp[exp_col])} (1=beginner, 2=intermediate, 3=advanced).")
            kb.add(" ".join(parts), cat, f"workout_{wtype}")

    # Calorie insights
    if cal_col:
        kb.add(
            f"Calorie burn statistics across all workouts: average {_safe_stat(df[cal_col])} kcal per session, "
            f"minimum {_safe_stat(df[cal_col],'min')} kcal, maximum {_safe_stat(df[cal_col],'max')} kcal, "
            f"median {_safe_stat(df[cal_col],'median')} kcal. "
            f"{'Higher calorie burns correlate with longer session durations and higher heart rates.' if dur_col and hr_col else ''}",
            cat, "calories",
        )

    # Experience level distribution
    if exp_col:
        for lvl in sorted(df[exp_col].dropna().unique()):
            grp = df[df[exp_col] == lvl]
            label = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}.get(int(lvl), f"Level {lvl}")
            parts = [f"{label} members ({len(grp)} sessions, {len(grp)/n*100:.1f}%):"]
            if cal_col:
                parts.append(f"avg calories {_safe_stat(grp[cal_col])},")
            if fat_col:
                parts.append(f"avg body fat {_safe_stat(grp[fat_col])}%,")
            if water_col:
                parts.append(f"avg water intake {_safe_stat(grp[water_col])} liters.")
            kb.add(" ".join(parts), cat, f"experience_{lvl}")

    # BMI insights
    if bmi_col:
        kb.add(
            f"BMI distribution: average {_safe_stat(df[bmi_col])}, "
            f"min {_safe_stat(df[bmi_col],'min')}, max {_safe_stat(df[bmi_col],'max')}. "
            f"Underweight (<18.5): {len(df[df[bmi_col]<18.5])} members, "
            f"Normal (18.5-25): {len(df[(df[bmi_col]>=18.5)&(df[bmi_col]<25)])} members, "
            f"Overweight (25-30): {len(df[(df[bmi_col]>=25)&(df[bmi_col]<30)])} members, "
            f"Obese (30+): {len(df[df[bmi_col]>=30])} members.",
            cat, "bmi",
        )

    # Hydration
    if water_col:
        kb.add(
            f"Water intake across members: average {_safe_stat(df[water_col])} liters per day. "
            f"Range: {_safe_stat(df[water_col],'min')} to {_safe_stat(df[water_col],'max')} liters. "
            f"Recommended hydration is 2.5-3.5 liters for active gym members.",
            cat, "hydration",
        )


def _add_daily_gym_chunks(kb: KnowledgeBase, df: pd.DataFrame):
    """Chunks from daily gym attendance dataset."""
    n = len(df)
    cat = "daily_attendance"

    age_col = _col(df, "age")
    mem_col = _col(df, "membership_type")
    wtype_col = _col(df, "workout_type")
    dur_col = _col(df, "workout_duration_minutes", "workout_duration_(min)")
    cal_col = _col(df, "calories_burned")
    attend_col = _col(df, "attendance_status")
    adh_col = _col(df, "adherence_score")
    churn_col = _col(df, "churn_risk")
    dow_col = _col(df, "day_of_week")

    kb.add(
        f"The daily gym attendance dataset has {n} visit records capturing member check-ins, workout details, and attendance patterns. "
        f"{'Member ages range from ' + _safe_stat(df[age_col], 'min') + ' to ' + _safe_stat(df[age_col], 'max') + '.' if age_col else ''} "
        f"{'Average workout duration is ' + _safe_stat(df[dur_col]) + ' minutes.' if dur_col else ''}",
        cat, "overview",
    )

    # Membership type breakdown
    if mem_col:
        for mtype, grp in df.groupby(mem_col):
            parts = [f"Membership type '{mtype}': {len(grp)} visits ({len(grp)/n*100:.1f}%)."]
            if dur_col:
                parts.append(f"Avg workout duration: {_safe_stat(grp[dur_col])} min.")
            if cal_col:
                parts.append(f"Avg calories burned: {_safe_stat(grp[cal_col])}.")
            if adh_col:
                parts.append(f"Avg adherence score: {_safe_stat(grp[adh_col])}.")
            kb.add(" ".join(parts), cat, f"membership_{mtype}")

    # Day-of-week patterns
    if dow_col:
        day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                     4: "Friday", 5: "Saturday", 6: "Sunday"}
        parts = ["Gym attendance by day of week:"]
        for dow in sorted(df[dow_col].dropna().unique()):
            cnt = len(df[df[dow_col] == dow])
            name = day_names.get(int(dow), str(dow))
            parts.append(f"{name}: {cnt} visits ({cnt/n*100:.1f}%)")
        kb.add(" ".join(parts) + ".", cat, "day_patterns")

    # Churn risk
    if churn_col:
        try:
            at_risk = df[df[churn_col] == 1]
            safe = df[df[churn_col] == 0]
            kb.add(
                f"Churn risk analysis: {len(at_risk)} members ({len(at_risk)/n*100:.1f}%) flagged as at-risk for leaving the gym. "
                f"{len(safe)} members ({len(safe)/n*100:.1f}%) have healthy attendance patterns. "
                f"{'At-risk members have an average adherence score of ' + _safe_stat(at_risk[adh_col]) + ' vs ' + _safe_stat(safe[adh_col]) + ' for stable members.' if adh_col else ''}",
                cat, "churn",
            )
        except Exception:
            pass


def _add_body_perf_chunks(kb: KnowledgeBase, df: pd.DataFrame):
    """Chunks from body performance dataset."""
    n = len(df)
    cat = "body_performance"

    age_col = _col(df, "age")
    class_col = _col(df, "class")
    grip_col = _col(df, "gripforce")
    situp_col = _col(df, "sit_ups_counts", "sit-ups_counts")
    jump_col = _col(df, "broad_jump_cm")
    bend_col = _col(df, "sit_and_bend_forward_cm")
    fat_col = _col(df, "body_fat_", "body_fat_%", "body fat_%")
    perf_col = _col(df, "performance_score")
    bp_sys = _col(df, "systolic")
    bp_dia = _col(df, "diastolic")

    kb.add(
        f"The body performance dataset contains {n} physical fitness assessments. "
        f"{'Age range: ' + _safe_stat(df[age_col],'min') + '-' + _safe_stat(df[age_col],'max') + ' years.' if age_col else ''} "
        f"{'Metrics include grip force, sit-ups, broad jump, flexibility, blood pressure, and body fat percentage.' } "
        f"{'Performance is classified into classes (A=excellent through D=needs improvement).' if class_col else ''}",
        cat, "overview",
    )

    # Per-class breakdown
    if class_col:
        for cls_val, grp in df.groupby(class_col):
            parts = [f"Performance class '{cls_val}' ({len(grp)} members, {len(grp)/n*100:.1f}%):"]
            if grip_col:
                parts.append(f"avg grip force {_safe_stat(grp[grip_col])} kg,")
            if situp_col:
                parts.append(f"avg sit-ups {_safe_stat(grp[situp_col])},")
            if jump_col:
                parts.append(f"avg broad jump {_safe_stat(grp[jump_col])} cm,")
            if bend_col:
                parts.append(f"avg flexibility {_safe_stat(grp[bend_col])} cm,")
            if fat_col:
                parts.append(f"avg body fat {_safe_stat(grp[fat_col])}%,")
            if bp_sys and bp_dia:
                parts.append(f"avg blood pressure {_safe_stat(grp[bp_sys])}/{_safe_stat(grp[bp_dia])} mmHg.")
            kb.add(" ".join(parts), cat, f"class_{cls_val}")

    # Performance score distribution
    if perf_col:
        q25 = df[perf_col].quantile(0.25)
        q50 = df[perf_col].quantile(0.50)
        q75 = df[perf_col].quantile(0.75)
        kb.add(
            f"Performance score distribution: 25th percentile = {q25:.2f}, "
            f"median = {q50:.2f}, 75th percentile = {q75:.2f}. "
            f"Scores above {q75:.2f} indicate elite fitness. Below {q25:.2f} suggests significant room for improvement.",
            cat, "performance_dist",
        )

    # Blood pressure insights
    if bp_sys and bp_dia:
        kb.add(
            f"Blood pressure across members: average systolic {_safe_stat(df[bp_sys])} mmHg, "
            f"average diastolic {_safe_stat(df[bp_dia])} mmHg. "
            f"Normal: < 120/80. Elevated: 120-129/<80. High Stage 1: 130-139/80-89. High Stage 2: ≥140/≥90.",
            cat, "blood_pressure",
        )


def _add_calories_chunks(kb: KnowledgeBase, df: pd.DataFrame):
    """Chunks from the calories + exercise merged dataset."""
    n = len(df)
    cat = "calories"

    age_col = _col(df, "age")
    dur_col = _col(df, "duration")
    hr_col = _col(df, "heart_rate")
    cal_col = _col(df, "calories")
    temp_col = _col(df, "body_temp")
    gender_col = _col(df, "gender", "gender_encoded")
    bmi_col = _col(df, "bmi")

    kb.add(
        f"The calorie burning dataset contains {n} exercise records with heart rate, duration, body temperature, and calories burned. "
        f"{'Duration ranges from ' + _safe_stat(df[dur_col],'min') + ' to ' + _safe_stat(df[dur_col],'max') + ' minutes (avg ' + _safe_stat(df[dur_col]) + ').' if dur_col else ''} "
        f"{'Average calories burned: ' + _safe_stat(df[cal_col]) + ' kcal.' if cal_col else ''} "
        f"{'Average heart rate during exercise: ' + _safe_stat(df[hr_col]) + ' BPM.' if hr_col else ''}",
        cat, "overview",
    )

    # Calorie ranges by duration bucket
    if dur_col and cal_col:
        for lo, hi, label in [(0, 10, "short (0-10 min)"), (10, 20, "medium (10-20 min)"), (20, 30, "long (20-30 min)")]:
            grp = df[(df[dur_col] >= lo) & (df[dur_col] < hi)]
            if len(grp) > 0:
                kb.add(
                    f"For {label} exercises ({len(grp)} records): "
                    f"average calorie burn is {_safe_stat(grp[cal_col])} kcal, "
                    f"average heart rate {_safe_stat(grp[hr_col]) if hr_col else 'N/A'} BPM.",
                    cat, f"duration_{label}",
                )

    # Heart rate zones
    if hr_col and cal_col:
        for lo, hi, zone in [(60, 100, "light"), (100, 140, "moderate"), (140, 200, "vigorous")]:
            grp = df[(df[hr_col] >= lo) & (df[hr_col] < hi)]
            if len(grp) > 0:
                kb.add(
                    f"Heart rate zone '{zone}' ({lo}-{hi} BPM, {len(grp)} records): "
                    f"average calorie burn {_safe_stat(grp[cal_col])} kcal over {_safe_stat(grp[dur_col]) if dur_col else 'N/A'} min average duration.",
                    cat, f"hr_zone_{zone}",
                )


def _add_fitbit_chunks(kb: KnowledgeBase, df: pd.DataFrame):
    """Chunks from Fitbit daily activity."""
    n = len(df)
    cat = "fitbit"

    steps_col = _col(df, "totalsteps")
    dist_col = _col(df, "totaldistance")
    cal_col = _col(df, "calories")
    active_col = _col(df, "total_active_minutes", "veryactiveminutes")
    sed_col = _col(df, "sedentaryminutes")
    engage_col = _col(df, "engagement_score")

    kb.add(
        f"Fitbit wearable data: {n} daily activity records from gym members' fitness trackers. "
        f"{'Average daily steps: ' + _safe_stat(df[steps_col]) + '.' if steps_col else ''} "
        f"{'Average daily calories: ' + _safe_stat(df[cal_col]) + ' kcal.' if cal_col else ''} "
        f"{'Average sedentary minutes: ' + _safe_stat(df[sed_col]) + ' min/day.' if sed_col else ''}",
        cat, "overview",
    )

    # Activity level insights
    if steps_col:
        kb.add(
            f"Daily step count: average {_safe_stat(df[steps_col])}, "
            f"min {_safe_stat(df[steps_col],'min')}, max {_safe_stat(df[steps_col],'max')}. "
            f"WHO recommends 8000-10000 steps/day. "
            f"Members above 10000 steps: {len(df[df[steps_col] > 10000])} ({len(df[df[steps_col] > 10000])/n*100:.1f}%). "
            f"Members below 5000 steps (sedentary): {len(df[df[steps_col] < 5000])} ({len(df[df[steps_col] < 5000])/n*100:.1f}%).",
            cat, "steps",
        )


def _add_model_info_chunks(kb: KnowledgeBase, registry_path: Path):
    """Chunks from trained model performance."""
    cat = "model_info"
    with open(registry_path) as f:
        registry = json.load(f)

    for entry in registry:
        name = entry.get("name", "unknown")
        algo = entry.get("algorithm", "unknown")
        mtype = entry.get("type", "unknown")
        metrics = entry.get("metrics", {})
        n_train = entry.get("n_train", "?")
        n_test = entry.get("n_test", "?")
        features = entry.get("features", [])

        metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        kb.add(
            f"ML Model '{name}' ({algo} {mtype}): trained on {n_train} samples, tested on {n_test} samples. "
            f"Performance metrics: {metric_str}. "
            f"Uses {len(features)} features: {', '.join(features[:10])}{'…' if len(features) > 10 else ''}.",
            cat, name,
        )


def _add_fitness_knowledge(kb: KnowledgeBase):
    """General fitness domain knowledge — ensures the chatbot can answer common questions."""
    cat = "fitness_knowledge"
    chunks = [
        # Workout types
        "HIIT (High-Intensity Interval Training) alternates between intense bursts and rest periods. "
        "Burns 400-600 calories in 30 minutes. Best for fat loss, cardiovascular fitness, and metabolic boost. "
        "Recommended 2-3 times per week with rest days between sessions.",

        "Strength training builds muscle mass, increases bone density, and boosts metabolism. "
        "For beginners: 3 sets of 8-12 reps. Intermediate: 4 sets of 6-10 reps. Advanced: 5+ sets with progressive overload. "
        "Major compound exercises: squat, deadlift, bench press, overhead press, barbell row.",

        "Yoga improves flexibility, balance, and mental well-being. Burns 200-400 calories per hour. "
        "Types: Hatha (beginner-friendly), Vinyasa (flow-based), Ashtanga (intense), Yin (passive stretching). "
        "Recommended 2-4 times per week as active recovery or primary workout.",

        "Cardio exercise includes running, cycling, swimming, and elliptical training. "
        "Zone 2 cardio (60-70% max heart rate) is optimal for fat burning and endurance building. "
        "Max heart rate estimate: 220 minus age. Target 150-300 minutes of moderate cardio per week.",

        # Nutrition
        "For muscle building: consume 1.6-2.2g protein per kg bodyweight daily. "
        "Caloric surplus of 200-500 kcal above maintenance for lean bulk. "
        "Good protein sources: chicken breast (31g/100g), eggs (13g/100g), Greek yogurt (10g/100g), whey protein (80g/100g).",

        "For fat loss: create a caloric deficit of 300-500 kcal below maintenance. "
        "Maintenance calories ≈ bodyweight (kg) × 30-33. "
        "Prioritize protein (prevents muscle loss), eat fiber-rich foods for satiety, drink 2-3 liters of water daily.",

        # Recovery
        "Rest and recovery are critical for progress. Muscles grow during rest, not during workouts. "
        "Sleep 7-9 hours per night. Sleep deprivation reduces strength by 10-15% and increases injury risk. "
        "Active recovery (light walking, stretching) is better than complete rest on off days.",

        # Progressive overload
        "Progressive overload is the fundamental principle of fitness improvement. "
        "Increase weight by 2-5% when you can complete all target reps with good form. "
        "Alternatively: increase reps, sets, time under tension, or decrease rest periods. "
        "Track your workouts to ensure progressive overload is happening.",

        # Heart rate training
        "Heart rate training zones: Zone 1 (50-60% max HR) = very light warmup. "
        "Zone 2 (60-70%) = fat burning, endurance. Zone 3 (70-80%) = aerobic capacity. "
        "Zone 4 (80-90%) = anaerobic threshold, lactate. Zone 5 (90-100%) = maximum effort, sprint intervals.",

        # Common mistakes
        "Common gym mistakes: not warming up (5-10 min dynamic stretching recommended), "
        "skipping compound movements for isolation exercises, not tracking workouts, "
        "overtraining without adequate rest, poor hydration, inconsistent attendance. "
        "Consistency beats intensity — 3 good sessions per week outperforms 6 mediocre ones.",

        # Supplements
        "Evidence-based supplements: creatine monohydrate (5g daily, well-researched for strength and power), "
        "caffeine (3-6mg/kg bodyweight pre-workout for performance), whey protein (convenient protein source), "
        "vitamin D (if deficient, common in indoor exercisers), omega-3 (anti-inflammatory, heart health).",

        # Body composition
        "Body fat percentage categories — Men: Essential (2-5%), Athletes (6-13%), Fitness (14-17%), "
        "Average (18-24%), Obese (25%+). Women: Essential (10-13%), Athletes (14-20%), "
        "Fitness (21-24%), Average (25-31%), Obese (32%+).",
    ]
    for i, chunk in enumerate(chunks):
        kb.add(chunk, cat, f"knowledge_{i}")


# ═══════════════════════════════════════════════════════════════
#  Main builder
# ═══════════════════════════════════════════════════════════════
def build_knowledge_base() -> KnowledgeBase:
    """Build the full knowledge base from all available gym data."""
    cfg_path = ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    kb = KnowledgeBase()
    fs_dir = ROOT / cfg["data"]["feature_store"]

    # Load and chunk each dataset
    loaders = [
        ("gym_exercise_features.parquet", _add_gym_exercise_chunks),
        ("daily_gym_features.parquet", _add_daily_gym_chunks),
        ("body_perf_features.parquet", _add_body_perf_chunks),
        ("calories_exercise_features.parquet", _add_calories_chunks),
        ("fitbit_daily_features.parquet", _add_fitbit_chunks),
    ]
    for fname, fn in loaders:
        path = fs_dir / fname
        if path.exists():
            df = pd.read_parquet(path)
            fn(kb, df)
            print(f"[KB] Loaded {fname}: {len(df)} rows")
        else:
            print(f"[KB] Skipping {fname} (not found)")

    # Model registry
    reg_path = ROOT / cfg["models"]["registry_path"]
    if reg_path.exists():
        _add_model_info_chunks(kb, reg_path)
        print("[KB] Loaded model registry")

    # General fitness knowledge
    _add_fitness_knowledge(kb)
    print("[KB] Added fitness domain knowledge")

    kb.build_index()
    kb.save()
    print(f"[KB] ✓ Knowledge base built: {len(kb.documents)} chunks indexed")
    return kb


if __name__ == "__main__":
    build_knowledge_base()
