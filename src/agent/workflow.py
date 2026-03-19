from __future__ import annotations

import pandas as pd

from src.behavior.segments import assign_behavior_segments
from src.optimization.action_recommender import generate_actions
from src.optimization.scheduling import apply_rescheduling_plan
from src.optimization.transport_rules import apply_transport_rules


RISK_BANDS = {
    "high": 0.7,
    "medium": 0.3,
}


def risk_band(probability: float) -> str:
    if probability >= RISK_BANDS["high"]:
        return "high"
    if probability >= RISK_BANDS["medium"]:
        return "medium"
    return "low"


def run_agent_decision_flow(scored_df: pd.DataFrame) -> pd.DataFrame:
    output = scored_df.copy()
    output["risk_band"] = output["no_show_probability"].apply(risk_band)
    output = apply_transport_rules(output)
    output = assign_behavior_segments(output)
    output = generate_actions(output)
    output = apply_rescheduling_plan(output)

    output["decision_trace"] = (
        "risk="
        + output["risk_band"].astype(str)
        + "|sentiment="
        + output["sentiment_label"].astype(str)
        + "|transport="
        + output["transport_need"].astype(str)
        + "|schedule="
        + output["schedule_status"].astype(str)
    )

    return output
