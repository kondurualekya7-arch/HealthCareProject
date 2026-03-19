from __future__ import annotations

import pandas as pd

from src.data.contracts import PathConfig
from src.models.train_models import train_and_compare_models
from src.pipelines.prepare_training_data import run_prepare_pipeline


def run_training_pipeline(config: PathConfig | None = None) -> dict:
    cfg = config or PathConfig()
    training_path = run_prepare_pipeline(cfg)
    dataset = pd.read_csv(training_path)

    _, metrics, best_model_name = train_and_compare_models(dataset, cfg)
    metrics["best_model"] = best_model_name
    return metrics


if __name__ == "__main__":
    results = run_training_pipeline()
    print("Training complete. Metrics:")
    for k, v in results.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
