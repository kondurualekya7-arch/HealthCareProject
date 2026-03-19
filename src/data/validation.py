from __future__ import annotations

from typing import List

import pandas as pd

from src.data.contracts import BASE_REQUIRED_COLUMNS, EXTENDED_COLUMNS


class ValidationError(Exception):
    pass


def validate_schema(df: pd.DataFrame) -> None:
    missing: List[str] = [c for c in BASE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required base columns: {missing}")


    missing_extended: List[str] = [c for c in EXTENDED_COLUMNS if c not in df.columns]
    if missing_extended:
        raise ValidationError(f"Missing extended columns: {missing_extended}")


def validate_quality(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValidationError("Dataset is empty after cleaning.")

    if df["no_show"].nunique() < 2:
        raise ValidationError("Target no_show has less than two classes.")

    if df["scheduledday"].isna().mean() > 0.02 or df["appointmentday"].isna().mean() > 0.02:
        raise ValidationError("Date quality below threshold.")
