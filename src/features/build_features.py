from __future__ import annotations

import pandas as pd
import numpy as np


def _age_band(age: float) -> str:
    if age < 18:
        return "child"
    if age < 35:
        return "young_adult"
    if age < 60:
        return "adult"
    return "senior"


def _distance_band(distance: float) -> str:
    if distance < 5:
        return "near"
    if distance < 15:
        return "medium"
    return "far"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    output = output.sort_values(["patientid", "appointmentday", "scheduledday"]).reset_index(drop=True)

    output["waiting_days"] = (
        (output["appointmentday"].dt.floor("D") - output["scheduledday"].dt.floor("D")).dt.days
    ).clip(lower=0)

    output["appointment_weekday"] = output["appointmentday"].dt.day_name()
    output["is_weekend"] = output["appointmentday"].dt.dayofweek.isin([5, 6]).astype(int)
    output["age_group"] = output["age"].apply(_age_band)
    output["sms_received_flag"] = output["sms_received"].astype(int)
    output["distance_category"] = output["transport_distance_km"].apply(_distance_band)
    output["chronic_illness_flag"] = ((output["hipertension"] == 1) | (output["diabetes"] == 1)).astype(int)

    appt_hour = output["appointmentday"].dt.hour
    fallback_hour = output["scheduledday"].dt.hour
    output["appointment_hour"] = appt_hour.where(appt_hour > 0, fallback_hour)
    output["appointment_period"] = np.where(output["appointment_hour"] < 12, "morning", "afternoon")

    output["num_reminders"] = np.where(
        output["sms_received_flag"] == 1,
        1 + (output["waiting_days"] >= 7).astype(int) + (output["waiting_days"] >= 14).astype(int),
        0,
    )

    output["past_visits_count"] = output.groupby("patientid").cumcount()
    output["past_no_show_count"] = (
        output.groupby("patientid")["no_show"].cumsum() - output["no_show"]
    ).clip(lower=0)

    output["historical_no_show_rate"] = np.where(
        output["past_visits_count"] > 0,
        output["past_no_show_count"] / output["past_visits_count"].replace(0, np.nan),
        output["no_show"].mean(),
    )
    output["historical_no_show_rate"] = output["historical_no_show_rate"].fillna(output["no_show"].mean())

    prev_appointment = output.groupby("patientid")["appointmentday"].shift(1)
    output["last_visit_gap_days"] = (
        (output["appointmentday"].dt.floor("D") - prev_appointment.dt.floor("D")).dt.days
    )
    output["last_visit_gap_days"] = output["last_visit_gap_days"].fillna(output["last_visit_gap_days"].median())
    output["last_visit_gap_days"] = output["last_visit_gap_days"].clip(lower=0)

    return output
