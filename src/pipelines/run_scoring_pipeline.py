from __future__ import annotations

import json
import os

import joblib
import pandas as pd

from src.agent.workflow import run_agent_decision_flow
from src.data.contracts import PathConfig
from src.models.feature_config import FEATURE_COLUMNS
from src.pipelines.prepare_training_data import run_prepare_pipeline
from src.pipelines.generate_business_insights import generate_business_insights


def run_scoring_pipeline(config: PathConfig | None = None) -> tuple[str, str]:
    cfg = config or PathConfig()

    if not os.path.exists(cfg.model_output):
        raise FileNotFoundError("Model file not found. Run training pipeline first.")

    training_path = run_prepare_pipeline(cfg)
    dataset = pd.read_csv(training_path)

    model = joblib.load(cfg.model_output)
    raw_probability = model.predict_proba(dataset[FEATURE_COLUMNS])[:, 1]

    decision_threshold = 0.5
    if os.path.exists(cfg.threshold_output):
        with open(cfg.threshold_output, "r", encoding="utf-8") as f:
            threshold_cfg = json.load(f)
            decision_threshold = float(threshold_cfg.get("threshold", 0.5))

    calibrated_probability = raw_probability
    if os.path.exists(cfg.calibrator_output):
        calibrator = joblib.load(cfg.calibrator_output)
        try:
            calibrated_probability = calibrator.predict(raw_probability)
        except Exception:
            calibrated_probability = raw_probability

    dataset["no_show_probability"] = calibrated_probability
    dataset["decision_threshold"] = decision_threshold

    dataset["no_show_prediction"] = (dataset["no_show_probability"] >= decision_threshold).astype(int)
    scored = run_agent_decision_flow(dataset)

    os.makedirs(os.path.dirname(cfg.scoring_output), exist_ok=True)
    scored.to_csv(cfg.scoring_output, index=False)

    actions = scored[[
        "patientid",
        "appointmentid",
        "no_show_probability",
        "risk_band",
        "behavior_segment",
        "transport_need",
        "recommended_action",
        "reschedule_recommended",
        "schedule_status",
        "suggested_appointmentday",
        "suggested_slot",
        "schedule_priority",
        "schedule_reason",
        "decision_trace",
    ]].copy()
    actions.to_csv(cfg.actions_output, index=False)

    generate_business_insights(cfg)

    return cfg.scoring_output, cfg.actions_output


if __name__ == "__main__":
    scored_path, actions_path = run_scoring_pipeline()
    print(f"Scored data: {scored_path}")
    print(f"Action recommendations: {actions_path}")
