# AI Agent Blueprint

## Goal
Automate intervention decisions for high-risk appointments.

## Deterministic Workflow
1. Ingest booked appointments
2. Score no-show probability
3. Compute sentiment bracket (negative / neutral / positive)
4. Compute transport priority
5. Assign behavior segment
6. Recommend intervention action
7. Log decision trace for audit

## Suggested Triggers
- New appointment booked
- Appointment updated
- Daily batch pre-visit check

## Suggested Actions by Rule
- High risk + negative sentiment: call + counseling + reminder
- High risk + urgent transport need: arrange transport support
- Medium risk + long waiting days: offer earlier slot
- Low risk: standard reminder

## Agent-Ready Interfaces
- `run_prepare_pipeline()`
- `run_training_pipeline()`
- `run_scoring_pipeline()`
- `run_agent_decision_flow()`

## Phase-2 Extension
- Replace synthetic sentiment with transcript ingestion
- Add constrained optimization for transport capacity and cost
- Add event-driven API endpoint and audit store
- Add monitoring for drift and intervention effectiveness
