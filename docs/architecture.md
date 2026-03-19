# System Architecture

This project implements an end-to-end healthcare no-show intelligence workflow with prediction, decision support, scheduling optimization, and operations UI.

## Architecture Diagram (Mermaid)

```mermaid
flowchart LR
    A[Raw Appointment Data\nKaggle + Extended Fields] --> B[Data Ingestion & Validation\nsrc/data]
    B --> C[Feature Engineering\nsrc/features]
    C --> D[Sentiment Enrichment\nsrc/nlp]
    D --> E[Model Training\nMulti-Model + Imbalance + Calibration\nsrc/models]
    E --> F[Best Model Artifacts\nmodels/*.pkl, metrics, threshold]
    D --> G[Scoring Pipeline\nsrc/pipelines/run_scoring_pipeline.py]
    F --> G
    G --> H[Agent Decision Flow\nrisk + transport + actions + scheduling\nsrc/agent]
    H --> I[Curated Outputs\ndata/curated/*.csv]
    I --> J[Power BI Dashboard]
    I --> K[Streamlit Operations Frontend\nfrontend/app.py]

    K --> L[Schedule Approval\nfinalized_schedule.csv]
    K --> M[Bulk Action Plan\nbulk_action_plan.csv]

    I --> N[Business Insights\nreports/business_insights.md]
    E --> O[Explainability\nFeature Importance + SHAP]
```

## Component Summary
- `src/data`: schema contracts, ingestion, cleaning, validation
- `src/features`: engineered behavioral + operational predictors
- `src/models`: model comparison, imbalance handling, calibration, threshold tuning
- `src/agent`: risk banding, recommendation logic, scheduling decisions
- `src/pipelines`: prepare, train, score, and insight generation orchestration
- `frontend`: role-aware operations interface
- `dashboard`: Power BI model, DAX, and storytelling specification
