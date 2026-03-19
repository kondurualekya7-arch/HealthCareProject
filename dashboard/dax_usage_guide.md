# DAX Usage Guide

## 1) Import tables
- Import `data/curated/scored_appointments.csv` as table name: `scored_appointments`
- (Optional) Import `data/curated/action_recommendations.csv` as `action_recommendations`

## 2) Create measures
- Open `dashboard/no_show_measures.dax`
- Copy all measures into Power BI (Model view -> New measure)

## 3) Visual mapping by page

### Overview
- Cards: `Total Appointments`, `No-Show Count`, `No-Show Rate`, `Avg No-Show Probability`
- Trend line: Date axis + `No-Show Rate`

### Patient Insights
- Bar: `age_group` vs `No-Show Rate`
- Donut: `sentiment_label` by `Total Appointments`
- Card: `Negative Sentiment Share`

### Operational Insights
- Bar: `appointment_weekday` vs `No-Show Rate`
- Clustered column: `No-Show Rate (SMS Sent)` and `No-Show Rate (No SMS)`
- Card: `SMS Impact (No-Show Rate Delta)`

### Prediction
- Cards: `High Risk Count`, `High Risk Share`, `Predicted No-Show Rate`, `Precision Proxy`
- Table: patient-level list with `no_show_probability`, `risk_band`, `behavior_segment`

### Transport Optimization
- Cards: `Urgent Transport Count`, `Urgent Transport Share`, `High-Risk Urgent Transport Count`
- Bar: `transport_need` vs `Total Appointments`
- Table: `recommended_action`, `transport_priority_score`, `travel_time_minutes`

## 4) Recommended slicers
- `appointment_weekday`
- `risk_band`
- `sentiment_label`
- `booking_channel`
- `age_group`
