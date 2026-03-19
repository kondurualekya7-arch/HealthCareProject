from __future__ import annotations

import pandas as pd


def assign_behavior_segments(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    conditions = [
        (output["historical_no_show_rate"] >= 0.5) | (output["no_show_probability"] >= 0.7),
        (output["travel_time_minutes"] >= 45) | (output["distance_category"] == "far"),
        output["sentiment_label"] == "negative",
        output["age_group"] == "senior",
    ]
    labels = [
        "high_risk_patient",
        "transport_dependent",
        "negative_sentiment",
        "elderly_needs_support",
    ]

    output["behavior_segment"] = "regular"
    for condition, label in zip(conditions, labels):
        output.loc[condition, "behavior_segment"] = label

    return output
