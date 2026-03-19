from __future__ import annotations

import json
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.data.contracts import PathConfig


FEATURE_COLUMNS = [
    "age",
    "waiting_days",
    "sms_received_flag",
    "is_weekend",
    "age_group",
    "distance_category",
    "transport_distance_km",
    "travel_time_minutes",
    "appointment_weekday",
    "booking_channel",
    "weather_condition",
    "sentiment_score",
    "sentiment_label",
    "historical_no_show_rate",
]

TARGET_COLUMN = "no_show"


NUMERIC_FEATURES = [
    "age",
    "waiting_days",
    "sms_received_flag",
    "is_weekend",
    "transport_distance_km",
    "travel_time_minutes",
    "sentiment_score",
    "historical_no_show_rate",
]

CATEGORICAL_FEATURES = [
    "age_group",
    "distance_category",
    "appointment_weekday",
    "booking_channel",
    "weather_condition",
    "sentiment_label",
]


def build_training_pipeline(random_state: int = 42) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=30, random_state=random_state)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def train_model(dataset: pd.DataFrame, config: PathConfig | None = None) -> Tuple[Pipeline, Dict[str, float]]:
    cfg = config or PathConfig()

    data = dataset.copy()
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = build_training_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_pred, y_proba)

    joblib.dump(pipeline, cfg.model_output)
    with open(cfg.metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return pipeline, metrics


def score_dataset(model: Pipeline, dataset: pd.DataFrame) -> pd.DataFrame:
    output = dataset.copy()
    output["no_show_probability"] = model.predict_proba(output[FEATURE_COLUMNS])[:, 1]
    output["no_show_prediction"] = (output["no_show_probability"] >= 0.5).astype(int)
    return output
