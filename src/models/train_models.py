from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data.contracts import PathConfig
from src.models.feature_config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:  # pragma: no cover
    SMOTE = None
    ImbPipeline = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


TARGET_COLUMN = "no_show"


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
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


def _build_models(scale_pos_weight: float) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=600, class_weight="balanced", solver="liblinear"),
        "decision_tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=30, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=140,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=4,
        )

    return models


def _build_pipeline(model: Any) -> Any:
    preprocessor = _build_preprocessor()

    if SMOTE is not None and ImbPipeline is not None:
        return ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def _evaluate(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def _find_best_threshold(y_true: pd.Series, y_prob: np.ndarray, min_recall: float = 0.60) -> dict[str, float]:
    thresholds = np.linspace(0.10, 0.90, 81)
    best: dict[str, float] | None = None
    fallback: dict[str, float] | None = None

    y_true_np = np.array(y_true)
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision = float(precision_score(y_true_np, y_pred, zero_division=0))
        recall = float(recall_score(y_true_np, y_pred, zero_division=0))
        f1 = float(f1_score(y_true_np, y_pred, zero_division=0))
        record = {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        if fallback is None or record["f1"] > fallback["f1"]:
            fallback = record

        if recall >= min_recall and (best is None or record["f1"] > best["f1"]):
            best = record

    return best if best is not None else (fallback or {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0})


def _split(dataset: pd.DataFrame) -> SplitData:
    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return SplitData(X_train, X_test, y_train, y_test)


def _extract_preprocessor_and_model(pipeline: Any) -> tuple[Any, Any]:
    return pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]


def _export_feature_importance(pipeline: Any, output_file: Path) -> None:
    preprocessor, model = _extract_preprocessor_and_model(pipeline)
    if not hasattr(model, "feature_importances_"):
        return

    feature_names = preprocessor.get_feature_names_out().tolist()
    importance = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    imp_df = imp_df.sort_values("importance", ascending=False)
    imp_df.to_csv(output_file, index=False)


def _export_shap_summary_if_available(pipeline: Any, X_sample: pd.DataFrame, output_file: Path) -> None:
    try:
        import shap  # type: ignore
    except Exception:
        return

    preprocessor, model = _extract_preprocessor_and_model(pipeline)
    if not hasattr(model, "feature_importances_"):
        return

    X_t = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out().tolist()

    try:
        explainer = shap.TreeExplainer(model)
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()

        shap_values = explainer.shap_values(X_t)

        if isinstance(shap_values, list):
            shap_array = np.array(shap_values[1])
        else:
            shap_array = np.array(shap_values)

        mean_abs = np.abs(shap_array).mean(axis=0)
        shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(output_file, index=False)
    except Exception:
        return


def train_and_compare_models(dataset: pd.DataFrame, config: PathConfig | None = None) -> tuple[Any, dict[str, float], str]:
    cfg = config or PathConfig()
    split = _split(dataset)

    positive_rate = float(split.y_train.mean())
    scale_pos_weight = float((1 - positive_rate) / max(positive_rate, 1e-6))

    models = _build_models(scale_pos_weight=scale_pos_weight)

    comparison_records: list[dict[str, Any]] = []
    fitted_pipelines: dict[str, Any] = {}
    best_name = ""
    best_pipeline = None
    best_metrics: dict[str, float] = {}
    best_score = -1.0

    for model_name, model in models.items():
        pipeline = _build_pipeline(model)
        pipeline.fit(split.X_train, split.y_train)
        fitted_pipelines[model_name] = pipeline

        y_pred = pipeline.predict(split.X_test)
        y_prob = pipeline.predict_proba(split.X_test)[:, 1]
        metrics = _evaluate(split.y_test, y_pred, y_prob)

        comparison_records.append({"model": model_name, **metrics})

        score = metrics["roc_auc"] + 0.1 * metrics["f1"]
        if score > best_score:
            best_score = score
            best_name = model_name
            best_pipeline = pipeline
            best_metrics = metrics

    comparison_df = pd.DataFrame(comparison_records).sort_values("roc_auc", ascending=False)
    comparison_df.to_csv(cfg.model_comparison_output, index=False)

    with open(cfg.metrics_output, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    with open(cfg.model_comparison_json_output, "w", encoding="utf-8") as f:
        json.dump(comparison_records, f, indent=2)

    if best_pipeline is None:
        raise RuntimeError("No model trained.")

    raw_probs = best_pipeline.predict_proba(split.X_test)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrated_probs = raw_probs
    calibration_used = False

    try:
        calibrator.fit(raw_probs, split.y_test)
        calibrated_probs = calibrator.predict(raw_probs)
        calibration_used = True
        joblib.dump(calibrator, cfg.calibrator_output)
    except Exception:
        calibration_used = False

    best_threshold_info = _find_best_threshold(split.y_test, calibrated_probs, min_recall=0.60)
    threshold = float(best_threshold_info["threshold"])
    tuned_pred = (calibrated_probs >= threshold).astype(int)
    tuned_metrics = _evaluate(split.y_test, tuned_pred, calibrated_probs)
    tuned_metrics["threshold"] = threshold
    tuned_metrics["threshold_precision"] = float(best_threshold_info["precision"])
    tuned_metrics["threshold_recall"] = float(best_threshold_info["recall"])
    tuned_metrics["threshold_f1"] = float(best_threshold_info["f1"])
    tuned_metrics["brier_raw"] = float(brier_score_loss(split.y_test, raw_probs))
    tuned_metrics["brier_calibrated"] = float(brier_score_loss(split.y_test, calibrated_probs))
    tuned_metrics["calibration_used"] = calibration_used

    with open(cfg.threshold_output, "w", encoding="utf-8") as f:
        json.dump(best_threshold_info, f, indent=2)

    calibration_report = {
        "brier_raw": tuned_metrics["brier_raw"],
        "brier_calibrated": tuned_metrics["brier_calibrated"],
        "calibration_used": calibration_used,
        "chosen_threshold": threshold,
    }
    with open(cfg.calibration_report_output, "w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2)

    joblib.dump(best_pipeline, cfg.model_output)

    rf_model = fitted_pipelines.get("random_forest")
    if rf_model is not None:
        _export_feature_importance(rf_model, Path(cfg.feature_importance_output))

    if best_pipeline is not None:
        _export_shap_summary_if_available(
            best_pipeline,
            split.X_test.head(500),
            Path(cfg.shap_summary_output),
        )

    return best_pipeline, tuned_metrics, best_name
