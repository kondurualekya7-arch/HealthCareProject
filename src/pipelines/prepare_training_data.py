from __future__ import annotations

import os

from src.data.cleaning import clean_appointments
from src.data.contracts import PathConfig
from src.data.load_data import ensure_extended_columns, load_raw_appointments
from src.data.validation import validate_quality, validate_schema
from src.features.build_features import build_features
from src.nlp.sentiment import enrich_with_sentiment


def run_prepare_pipeline(config: PathConfig | None = None) -> str:
    cfg = config or PathConfig()

    df = load_raw_appointments(cfg)
    df = ensure_extended_columns(df)
    df = clean_appointments(df)
    validate_schema(df)
    validate_quality(df)

    df = build_features(df)
    df = enrich_with_sentiment(df)

    os.makedirs(os.path.dirname(cfg.training_output), exist_ok=True)
    df.to_csv(cfg.training_output, index=False)
    return cfg.training_output


if __name__ == "__main__":
    output_path = run_prepare_pipeline()
    print(f"Prepared dataset saved to: {output_path}")
