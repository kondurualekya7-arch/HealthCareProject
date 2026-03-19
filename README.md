# Healthcare Appointment No-Show Prediction (AI-Agent Ready MVP)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![ML](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost-green)
![Frontend](https://img.shields.io/badge/Frontend-Streamlit-red)
![BI](https://img.shields.io/badge/BI-Power%20BI-yellow)
![Status](https://img.shields.io/badge/Status-End--to--End%20Working-brightgreen)

## Objective
Predict appointment no-shows and produce operational recommendations for reminder escalation, transport coordination, and schedule optimization.

## Why This Project Matters
- Reduces missed appointments through proactive intervention
- Improves operational efficiency via risk-aware scheduling
- Connects ML prediction to real workflow actions (transport, reminders, reschedule)

## Portfolio Quick Links
- Architecture: `docs/architecture.md`
- Demo script: `docs/demo_script.md`
- Power BI storyboard: `dashboard/powerbi_storyboard.md`
- Modeling notes: `reports/modeling_notes.md`
- Screenshot kit: `docs/screenshots/README.md`

## MVP Components
- Data ingestion and cleaning for Kaggle Brazil no-show dataset
- Feature engineering for waiting days, weekday effects, appointment period, reminder intensity, age and transport features
- Sentiment enrichment from patient feedback text
- Multi-model comparison (Logistic Regression, Decision Tree, Random Forest, XGBoost if available)
- Behavioral segmentation and transport prioritization
- Agent-style rule workflow for recommended actions
- Power BI-ready curated outputs and dashboard specification

## Expected Input
Place Kaggle CSV at:
- `data/raw/KaggleV2-May-2016.csv`

Synthetic extension columns are auto-generated if missing:
- `patient_feedback_text`
- `transport_distance_km`
- `travel_time_minutes`
- `weather_condition`
- `booking_channel`

## Run
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Prepare and train:
   - `python -m src.pipelines.run_training_pipeline`
3. Score and generate actions:
   - `python -m src.pipelines.run_scoring_pipeline`
4. Launch management frontend:
   - `streamlit run frontend/app.py`

## Architecture
See system design and Mermaid diagram in:
- `docs/architecture.md`

## Demo Flow (Interview Ready)
Use the guided script in:
- `docs/demo_script.md`

## Outputs
- Best model: `models/no_show_best_model.pkl`
- Metrics: `models/no_show_metrics.json`
- Model comparison: `models/model_comparison.csv`, `models/model_comparison.json`
- Explainability: `models/random_forest_feature_importance.csv`, `models/random_forest_shap_summary.csv` (if SHAP is available)
- Calibration + threshold: `models/probability_calibrator.pkl`, `models/decision_threshold.json`, `models/calibration_report.json`
- Training data: `data/processed/training_dataset.csv`
- Scored appointments: `data/curated/scored_appointments.csv`
- Action recommendations: `data/curated/action_recommendations.csv`
- Scheduling recommendations: included in curated outputs (`schedule_status`, `suggested_appointmentday`, `suggested_slot`, `schedule_priority`)
- Finalized approved schedule (from frontend): `data/curated/finalized_schedule.csv`
- Business insights report: `reports/business_insights.md`, `reports/business_insights.json`

## Power BI
Use curated outputs in `data/curated/` and dashboard guidance in `dashboard/powerbi_spec.md`.
See enhanced storyboard in `dashboard/powerbi_storyboard.md`.

## Frontend Management UI
Use `frontend/app.py` for a lightweight operations view with:
- Risk and sentiment filtering
- Transport-priority triage
- Recommended action queue export
- Schedule management queue with reschedule suggestions and downloadable plan

## AI-Agent Expansion
See `docs/agent_blueprint.md` for automation flow and extension points.

## Sample Screenshots
After capturing images into `docs/screenshots/`, add these to showcase the system:

![Executive Summary](docs/screenshots/01-executive-summary.png)
![Operations Queue](docs/screenshots/02-operations-queue.png)
![Schedule Management](docs/screenshots/03-schedule-management.png)
![Analytics Story](docs/screenshots/04-analytics-story.png)
![Model Explainability](docs/screenshots/05-model-explainability.png)
