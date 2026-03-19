from __future__ import annotations

import numpy as np
import pandas as pd


NO_SHOW_TRUE_VALUES = {"yes", "y", "1", 1, True}


def _normalize_no_show(value) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() in NO_SHOW_TRUE_VALUES else 0
    return 1 if value in NO_SHOW_TRUE_VALUES else 0


def clean_appointments(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    output = output.drop_duplicates(subset=["patientid", "appointmentid"], keep="last")

    for col in ["scheduledday", "appointmentday"]:
        output[col] = pd.to_datetime(output[col], errors="coerce", utc=True)

    output["age"] = pd.to_numeric(output["age"], errors="coerce")
    output["age"] = output["age"].fillna(output["age"].median())
    output["age"] = np.where(output["age"] < 0, 0, output["age"])

    for binary_col in ["scholarship", "hipertension", "diabetes", "alcoholism", "handcap", "sms_received"]:
        if binary_col in output.columns:
            output[binary_col] = pd.to_numeric(output[binary_col], errors="coerce").fillna(0).astype(int)

    output["no_show"] = output["no_show"].apply(_normalize_no_show).astype(int)

    output["patient_feedback_text"] = output["patient_feedback_text"].fillna("No feedback provided")
    output["weather_condition"] = output["weather_condition"].fillna("unknown")
    output["booking_channel"] = output["booking_channel"].fillna("unknown")

    output["transport_distance_km"] = pd.to_numeric(output["transport_distance_km"], errors="coerce")
    output["transport_distance_km"] = output["transport_distance_km"].fillna(output["transport_distance_km"].median())

    output["travel_time_minutes"] = pd.to_numeric(output["travel_time_minutes"], errors="coerce")
    output["travel_time_minutes"] = output["travel_time_minutes"].fillna((output["transport_distance_km"] * 4.5).round())

    output = output.dropna(subset=["scheduledday", "appointmentday"])
    return output
