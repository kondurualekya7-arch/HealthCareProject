# Demo Script (Interview / Portfolio)

## Duration
8-12 minutes

## 1) Problem Framing (1 minute)
- "Hospitals lose capacity and continuity of care due to missed appointments."
- "Goal: predict no-shows early and trigger practical interventions: reminders, transport, and smart rescheduling."

## 2) Data + Feature Engineering (2 minutes)
- Show raw schema and mention extension columns (sentiment + transport + channel).
- Highlight high-impact features:
  - lead time (`waiting_days`)
  - historical behavior (`past_visits_count`, `past_no_show_count`, `historical_no_show_rate`)
  - communication (`sms_received_flag`, `num_reminders`)
  - operational context (`appointment_period`, `appointment_weekday`)

## 3) Modeling Strategy (2 minutes)
- Explain model benchmark stack: Logistic Regression, Decision Tree, Random Forest, XGBoost.
- Explain imbalance handling: class weights + SMOTE.
- Explain calibration + tuned threshold and why this matters operationally.

## 4) Explainability + Business Insights (2 minutes)
- Show model comparison and best model artifacts.
- Show feature importance / SHAP summary.
- Show business insight report and call out 2-3 decisions it drives.

## 5) Operations UI Walkthrough (2-3 minutes)
- Open Executive Summary tab: key KPIs.
- Open Operations tab: queue triage + bulk actions.
- Open Schedule Management: approve and export reschedules.
- Open Analytics tab: what-if threshold simulation and operational impact.

## 6) Close with Impact (1 minute)
- "This is not only prediction—it is a closed-loop operational system."
- "Outputs are ready for both dashboard decision-making and frontline workflows."

---

## Suggested Live Commands
```powershell
# Train and compare models
python -m src.pipelines.run_training_pipeline

# Score and generate decisions
python -m src.pipelines.run_scoring_pipeline

# Run operations UI
streamlit run frontend/app.py
```
