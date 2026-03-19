# Modeling and Explainability Notes

## Model Comparison
This project trains and compares:
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- XGBoost (if package is available)

Comparison artifacts are saved in:
- `models/model_comparison.csv`
- `models/model_comparison.json`

## Imbalance Handling
No-show labels are imbalanced. Mitigation used:
- `class_weight='balanced'` for linear/tree models
- `SMOTE` oversampling in training pipeline when `imbalanced-learn` is available
- `scale_pos_weight` for XGBoost when enabled

## Explainability
- Feature importance export for Random Forest:
  - `models/random_forest_feature_importance.csv`
- SHAP summary export (optional if `shap` is installed):
  - `models/random_forest_shap_summary.csv`

## Threshold Tuning and Calibration
- Probability calibration with isotonic regression (when fit succeeds)
- Decision threshold tuned on validation/test probabilities with recall floor
- Artifacts:
  - `models/probability_calibrator.pkl`
  - `models/decision_threshold.json`
  - `models/calibration_report.json`
