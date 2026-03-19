from __future__ import annotations

import pandas as pd


def recommend_action(row: pd.Series) -> str:
    if row["no_show_probability"] >= 0.7:
        if row["sentiment_label"] == "negative":
            return "Call + counseling + reminder"
        if row["transport_need"] == "urgent_support":
            return "Arrange transport support"
        return "Priority reminder escalation"

    if row["transport_need"] in {"urgent_support", "consider_support"}:
        return "Offer transport or closer clinic"

    if row["waiting_days"] > 14:
        return "Offer earlier reschedule slot"

    return "Standard SMS reminder"


def generate_actions(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["recommended_action"] = output.apply(recommend_action, axis=1)
    return output
