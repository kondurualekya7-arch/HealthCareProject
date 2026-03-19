# Power BI Dashboard Specification

## Data Sources
- `data/curated/scored_appointments.csv`
- `data/curated/action_recommendations.csv`

## Semantic Model
Create relationships on:
- `appointmentid`
- `patientid`

## Required Pages
1. Overview
   - Total appointments
   - No-show rate
   - No-show trend by appointment date
2. Patient Insights
   - No-show by age_group
   - No-show by gender
   - Sentiment distribution
3. Operational Insights
   - No-show by weekday
   - No-show by appointment period (morning/afternoon)
   - SMS effect on attendance
   - Number of reminders vs no-show
   - Distance/travel-time vs no-show
   - Heatmap: weekday vs appointment period
4. Prediction View
   - High-risk patient table
   - Probability distribution
5. Transport Optimization
   - Transport_need distribution
   - Action type counts
   - High-distance high-risk queue

## Suggested Measures (DAX)
- `Total Appointments = COUNTROWS(scored_appointments)`
- `NoShow Count = CALCULATE(COUNTROWS(scored_appointments), scored_appointments[no_show] = 1)`
- `NoShow Rate = DIVIDE([NoShow Count], [Total Appointments])`
- `High Risk Count = CALCULATE(COUNTROWS(scored_appointments), scored_appointments[risk_band] = "high")`
- `Transport Urgent Count = CALCULATE(COUNTROWS(scored_appointments), scored_appointments[transport_need] = "urgent_support")`

## Filters
- Date range
- Risk band
- Sentiment label
- Booking channel
- Weekday
- Neighbourhood (clinic proxy)
- Appointment period
