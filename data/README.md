# Data Layout

- `data/raw/` — original downloads (never modify in-place).
- `data/processed/` — extracted/cleaned data ready for ML pipelines.

Rule: raw stays immutable; derive everything into processed.
