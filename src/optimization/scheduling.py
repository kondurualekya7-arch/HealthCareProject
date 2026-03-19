from __future__ import annotations

from typing import Tuple

import pandas as pd


def _next_business_day(date_value: pd.Timestamp) -> pd.Timestamp:
    adjusted = date_value
    while adjusted.weekday() >= 5:
        adjusted += pd.Timedelta(days=1)
    return adjusted


def _suggest_slot(row: pd.Series) -> str:
    if row.get("transport_need") == "urgent_support":
        return "10:30"
    if row.get("risk_band") == "high" and row.get("sentiment_label") == "negative":
        return "09:00"
    if row.get("risk_band") == "high":
        return "11:00"
    return "14:00"


def _reschedule_reason(row: pd.Series) -> str:
    if row.get("recommended_action") == "Offer earlier reschedule slot":
        return "Long waiting time; suggest earlier slot"
    if row.get("transport_need") == "urgent_support":
        return "High transport burden; align to support slot"
    if row.get("risk_band") == "high" and row.get("sentiment_label") == "negative":
        return "High risk and negative sentiment; early follow-up"
    if row.get("risk_band") == "high":
        return "High no-show risk; proactive reschedule"
    return "No change needed"


def _days_to_move_earlier(row: pd.Series) -> int:
    waiting_days = int(max(row.get("waiting_days", 0), 0))

    if row.get("recommended_action") == "Offer earlier reschedule slot":
        return min(max(waiting_days - 3, 1), 10)

    if row.get("risk_band") == "high" and row.get("sentiment_label") == "negative":
        return min(max(waiting_days // 2, 1), 7)

    if row.get("transport_need") == "urgent_support":
        return min(max(waiting_days // 3, 1), 5)

    if row.get("risk_band") == "high":
        return min(max(waiting_days // 4, 1), 4)

    return 0


def _should_reschedule(row: pd.Series) -> bool:
    if row.get("recommended_action") in {
        "Offer earlier reschedule slot",
        "Offer transport or closer clinic",
        "Call + counseling + reminder",
    }:
        return True

    if row.get("risk_band") == "high" and float(row.get("no_show_probability", 0.0)) >= 0.8:
        return True

    return False


def _schedule_priority(row: pd.Series) -> float:
    probability = float(row.get("no_show_probability", 0.0))
    transport_score = float(row.get("transport_priority_score", 0.0))
    sentiment_penalty = 0.1 if row.get("sentiment_label") == "negative" else 0.0
    return round(min(probability + 0.3 * transport_score + sentiment_penalty, 1.0), 3)


def _build_schedule_decision(row: pd.Series) -> Tuple[str, str, str, float]:
    appointment_day = pd.to_datetime(row.get("appointmentday"), errors="coerce", utc=True)
    scheduled_day = pd.to_datetime(row.get("scheduledday"), errors="coerce", utc=True)

    if pd.isna(appointment_day):
        return "keep_schedule", "", "", 0.0

    reschedule_needed = _should_reschedule(row)
    days_to_move = _days_to_move_earlier(row)

    if not reschedule_needed or days_to_move <= 0:
        return "keep_schedule", appointment_day.strftime("%Y-%m-%d"), "", _schedule_priority(row)

    proposed = appointment_day - pd.Timedelta(days=days_to_move)

    if pd.notna(scheduled_day):
        min_allowed = scheduled_day.floor("D") + pd.Timedelta(days=1)
        if proposed < min_allowed:
            proposed = min_allowed

    proposed = _next_business_day(proposed)

    if proposed.floor("D") >= appointment_day.floor("D"):
        return "keep_schedule", appointment_day.strftime("%Y-%m-%d"), "", _schedule_priority(row)

    return (
        "reschedule_recommended",
        proposed.strftime("%Y-%m-%d"),
        _suggest_slot(row),
        _schedule_priority(row),
    )


def apply_rescheduling_plan(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    decisions = output.apply(_build_schedule_decision, axis=1, result_type="expand")
    decisions.columns = [
        "schedule_status",
        "suggested_appointmentday",
        "suggested_slot",
        "schedule_priority",
    ]

    output = pd.concat([output, decisions], axis=1)
    output["schedule_reason"] = output.apply(_reschedule_reason, axis=1)
    output["reschedule_recommended"] = (output["schedule_status"] == "reschedule_recommended").astype(int)

    return output
