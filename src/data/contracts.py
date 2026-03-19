from dataclasses import dataclass
from typing import List

BASE_REQUIRED_COLUMNS = [
    "patientid",
    "appointmentid",
    "gender",
    "scheduledday",
    "appointmentday",
    "age",
    "neighbourhood",
    "scholarship",
    "hipertension",
    "diabetes",
    "alcoholism",
    "handcap",
    "sms_received",
    "no_show",
]

EXTENDED_COLUMNS = [
    "patient_feedback_text",
    "transport_distance_km",
    "travel_time_minutes",
    "weather_condition",
    "booking_channel",
]

TARGET_COLUMN = "no_show"


@dataclass(frozen=True)
class PathConfig:
    raw_input: str = "data/raw/KaggleV2-May-2016.csv"
    processed_output: str = "data/processed/appointments_processed.csv"
    training_output: str = "data/processed/training_dataset.csv"
    scoring_output: str = "data/curated/scored_appointments.csv"
    actions_output: str = "data/curated/action_recommendations.csv"
    model_output: str = "models/no_show_best_model.pkl"
    metrics_output: str = "models/no_show_metrics.json"
    model_comparison_output: str = "models/model_comparison.csv"
    model_comparison_json_output: str = "models/model_comparison.json"
    feature_importance_output: str = "models/random_forest_feature_importance.csv"
    shap_summary_output: str = "models/random_forest_shap_summary.csv"
    calibrator_output: str = "models/probability_calibrator.pkl"
    threshold_output: str = "models/decision_threshold.json"
    calibration_report_output: str = "models/calibration_report.json"
    business_insights_output: str = "reports/business_insights.md"
    business_insights_json_output: str = "reports/business_insights.json"


def all_expected_columns() -> List[str]:
    return BASE_REQUIRED_COLUMNS + EXTENDED_COLUMNS
