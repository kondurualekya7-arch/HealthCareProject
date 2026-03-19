from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.contracts import PathConfig


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def generate_business_insights(config: PathConfig | None = None) -> dict:
    cfg = config or PathConfig()
    scored = pd.read_csv(cfg.scoring_output)

    no_show_rate = float(scored["no_show"].mean())

    sms_sent = scored[scored["sms_received"] == 1]
    sms_not_sent = scored[scored["sms_received"] == 0]
    sms_sent_no_show = float(sms_sent["no_show"].mean()) if not sms_sent.empty else 0.0
    sms_not_sent_no_show = float(sms_not_sent["no_show"].mean()) if not sms_not_sent.empty else 0.0

    by_weekday = scored.groupby("appointment_weekday")["no_show"].mean().sort_values(ascending=False)
    peak_day = by_weekday.index[0] if not by_weekday.empty else "N/A"
    peak_day_rate = float(by_weekday.iloc[0]) if not by_weekday.empty else 0.0

    lead_time_bins = pd.cut(scored["waiting_days"], bins=[-1, 0, 3, 7, 14, 999], labels=["0", "1-3", "4-7", "8-14", "15+"])
    lead_time_no_show = scored.groupby(lead_time_bins, observed=False)["no_show"].mean().to_dict()

    high_risk_share = float((scored["risk_band"] == "high").mean())
    reschedule_recommended = int((scored.get("schedule_status", pd.Series(dtype=str)) == "reschedule_recommended").sum())

    insights = {
        "overall_no_show_rate": no_show_rate,
        "sms_sent_no_show_rate": sms_sent_no_show,
        "sms_not_sent_no_show_rate": sms_not_sent_no_show,
        "peak_no_show_weekday": peak_day,
        "peak_no_show_weekday_rate": peak_day_rate,
        "lead_time_no_show_rate": {k: float(v) for k, v in lead_time_no_show.items()},
        "high_risk_share": high_risk_share,
        "reschedule_recommended_count": reschedule_recommended,
    }

    md_lines = [
        "# Business Insights",
        "",
        f"- Overall no-show rate: {_pct(no_show_rate)}",
        f"- No-show rate with SMS: {_pct(sms_sent_no_show)}",
        f"- No-show rate without SMS: {_pct(sms_not_sent_no_show)}",
        f"- Peak no-show weekday: {peak_day} ({_pct(peak_day_rate)})",
        f"- High-risk share of appointments: {_pct(high_risk_share)}",
        f"- Reschedule recommendations generated: {reschedule_recommended:,}",
        "",
        "## Lead Time Effect",
    ]

    for label, value in insights["lead_time_no_show_rate"].items():
        md_lines.append(f"- Waiting days {label}: {_pct(value)} no-show")

    md_lines.extend([
        "",
        "## Recommendations",
        "- Send SMS reminders at least 24 hours before appointment and add second reminder for high-risk patients.",
        "- Prioritize high-risk and transport-dependent patients in earlier operational slots.",
        "- Use targeted overbooking only in high-risk windows with operational safeguards.",
        "- Trigger proactive rescheduling for long lead-time appointments.",
    ])

    Path(cfg.business_insights_output).write_text("\n".join(md_lines), encoding="utf-8")
    with open(cfg.business_insights_json_output, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)

    return insights
