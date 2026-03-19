from __future__ import annotations

import pandas as pd


def transport_priority_score(df: pd.DataFrame) -> pd.Series:
    score = (
        (df["transport_distance_km"] >= 15).astype(int) * 0.35
        + (df["travel_time_minutes"] >= 45).astype(int) * 0.25
        + (df["age_group"] == "senior").astype(int) * 0.20
        + (df["sentiment_label"] == "negative").astype(int) * 0.20
    )
    return score.round(2)


def classify_transport_need(score: float) -> str:
    if score >= 0.7:
        return "urgent_support"
    if score >= 0.4:
        return "consider_support"
    return "standard"


def apply_transport_rules(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["transport_priority_score"] = transport_priority_score(output)
    output["transport_need"] = output["transport_priority_score"].apply(classify_transport_need)
    return output
