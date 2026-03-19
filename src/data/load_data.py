from __future__ import annotations

import os
import pandas as pd

from src.data.contracts import PathConfig


COLUMN_RENAME_MAP = {
    "PatientId": "patientid",
    "AppointmentID": "appointmentid",
    "Gender": "gender",
    "ScheduledDay": "scheduledday",
    "AppointmentDay": "appointmentday",
    "Age": "age",
    "Neighbourhood": "neighbourhood",
    "Scholarship": "scholarship",
    "Hipertension": "hipertension",
    "Hypertension": "hipertension",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholism",
    "Handcap": "handcap",
    "Handicap": "handcap",
    "SMS_received": "sms_received",
    "No-show": "no_show",
}


DEFAULT_FEEDBACK = [
    "I will be there on time.",
    "I might miss because of travel delay.",
    "Please reschedule, transport is difficult.",
    "Confirmed, thank you.",
    "Not sure I can attend this week.",
]


DEFAULT_WEATHER = ["clear", "rain", "storm", "cloudy"]
DEFAULT_CHANNELS = ["app", "call", "walk-in"]


def load_raw_appointments(path_config: PathConfig | None = None) -> pd.DataFrame:
    config = path_config or PathConfig()
    if not os.path.exists(config.raw_input):
        raise FileNotFoundError(
            f"Raw dataset not found at {config.raw_input}. Place Kaggle CSV there first."
        )

    df = pd.read_csv(config.raw_input)
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def ensure_extended_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    n_rows = len(output)

    if "patient_feedback_text" not in output.columns:
        output["patient_feedback_text"] = [DEFAULT_FEEDBACK[i % len(DEFAULT_FEEDBACK)] for i in range(n_rows)]

    if "transport_distance_km" not in output.columns:
        output["transport_distance_km"] = ((output.index % 25) + 1).astype(float)

    if "travel_time_minutes" not in output.columns:
        output["travel_time_minutes"] = (output["transport_distance_km"] * 4.5).round().astype(int)

    if "weather_condition" not in output.columns:
        output["weather_condition"] = [DEFAULT_WEATHER[i % len(DEFAULT_WEATHER)] for i in range(n_rows)]

    if "booking_channel" not in output.columns:
        output["booking_channel"] = [DEFAULT_CHANNELS[i % len(DEFAULT_CHANNELS)] for i in range(n_rows)]

    return output
