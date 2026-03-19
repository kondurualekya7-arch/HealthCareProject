# Optimization Recommendations

## Scheduling
- Prioritize early slots for patients with waiting_days > 14 and medium/high risk.
- Hold contingency slots to absorb predicted no-show volatility.

## Reminder Strategy
- Low risk: standard SMS reminder.
- Medium risk: SMS + confirmation link.
- High risk: call-based escalation and counseling support.

## Transport Coordination
- Assign transport support first to: high risk + urgent_support transport_need.
- Offer closer-clinic rescheduling for far-distance patients when transport inventory is limited.
- Maintain a daily transport queue sorted by transport_priority_score and appointment time.

## Sentiment-Driven Intervention
- Negative sentiment should trigger proactive human outreach.
- Neutral sentiment with high travel burden should trigger transport offer.
- Positive sentiment with low risk can remain on standard workflow.

## KPI Tracking
- No-show rate reduction (overall and by segment)
- Intervention uptake rate
- Transport utilization efficiency
- Cost per prevented no-show
