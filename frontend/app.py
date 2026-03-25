from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SCORED_PATH = ROOT / "data" / "curated" / "scored_appointments.csv"
ACTIONS_PATH = ROOT / "data" / "curated" / "action_recommendations.csv"
FINALIZED_SCHEDULE_PATH = ROOT / "data" / "curated" / "finalized_schedule.csv"
BULK_ACTIONS_PATH = ROOT / "data" / "curated" / "bulk_action_plan.csv"
MODEL_COMPARISON_PATH = ROOT / "models" / "model_comparison.csv"
METRICS_PATH = ROOT / "models" / "no_show_metrics.json"
THRESHOLD_PATH = ROOT / "models" / "decision_threshold.json"
CALIBRATION_PATH = ROOT / "models" / "calibration_report.json"
FEATURE_IMPORTANCE_PATH = ROOT / "models" / "random_forest_feature_importance.csv"
SHAP_SUMMARY_PATH = ROOT / "models" / "random_forest_shap_summary.csv"
BUSINESS_INSIGHTS_MD = ROOT / "reports" / "business_insights.md"
BUSINESS_INSIGHTS_JSON = ROOT / "reports" / "business_insights.json"


def apply_professional_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1400px;}
        .dashboard-title {font-size: 2.0rem; font-weight: 700; margin-bottom: 0.1rem;}
        .dashboard-subtitle {color: #6b7280; margin-bottom: 1rem;}
        .metric-chip {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            background: #eef2ff;
            color: #3730a3;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_altair_chart(chart: alt.Chart, title: str | None = None, height: int = 300) -> alt.Chart:
    if title:
        chart = chart.properties(title=title)
    return (
        chart.properties(height=height)
        .configure_axis(
            labelFontSize=12,
            titleFontSize=13,
            gridColor="#e5e7eb",
        )
        .configure_view(strokeOpacity=0)
        .configure_legend(labelFontSize=12, titleFontSize=12, orient="top")
        .configure_title(fontSize=16, anchor="start", color="#111827")
        .configure_range(
            category=[
                "#4f46e5",
                "#0ea5e9",
                "#f59e0b",
                "#10b981",
                "#ef4444",
                "#8b5cf6",
                "#14b8a6",
            ]
        )
    )


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    scored = pd.read_csv(SCORED_PATH)
    if "appointmentday" in scored.columns:
        scored["appointmentday"] = pd.to_datetime(scored["appointmentday"], errors="coerce", utc=True)
        scored["appointment_date"] = scored["appointmentday"].dt.date

    if ACTIONS_PATH.exists():
        actions = pd.read_csv(ACTIONS_PATH)
        action_merge_columns = [
            "appointmentid",
            "recommended_action",
            "reschedule_recommended",
            "schedule_status",
            "suggested_appointmentday",
            "suggested_slot",
            "schedule_priority",
            "schedule_reason",
        ]
        action_merge_columns = [c for c in action_merge_columns if c in actions.columns]
        merged = scored.merge(
            actions[action_merge_columns],
            on="appointmentid",
            how="left",
            suffixes=("", "_from_actions"),
        )
        if "recommended_action_from_actions" in merged.columns:
            merged["recommended_action"] = merged["recommended_action_from_actions"].fillna(
                merged["recommended_action"]
            )
            merged = merged.drop(columns=["recommended_action_from_actions"])
        return merged

    return scored


@st.cache_data(show_spinner=False)
def load_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_optional_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _compute_binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true_i = y_true.astype(int)
    y_pred_i = y_pred.astype(int)

    tp = int(((y_true_i == 1) & (y_pred_i == 1)).sum())
    fp = int(((y_true_i == 0) & (y_pred_i == 1)).sum())
    fn = int(((y_true_i == 1) & (y_pred_i == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    st.sidebar.header("Filters")

    role = st.sidebar.selectbox(
        "Role View",
        ["Ops Lead", "Scheduler", "Care Coordinator"],
        index=0,
    )

    risk_options = sorted(df["risk_band"].dropna().unique().tolist())
    sentiment_options = sorted(df["sentiment_label"].dropna().unique().tolist())
    transport_options = sorted(df["transport_need"].dropna().unique().tolist())
    action_options = sorted(df["recommended_action"].dropna().unique().tolist())
    weekday_options = sorted(df["appointment_weekday"].dropna().unique().tolist())
    period_options = sorted(df["appointment_period"].dropna().unique().tolist()) if "appointment_period" in df.columns else []
    neighbourhood_options = sorted(df["neighbourhood"].dropna().unique().tolist()) if "neighbourhood" in df.columns else []
    channel_options = sorted(df["booking_channel"].dropna().unique().tolist()) if "booking_channel" in df.columns else []
    age_group_options = sorted(df["age_group"].dropna().unique().tolist()) if "age_group" in df.columns else []
    gender_options = sorted(df["gender"].dropna().unique().tolist()) if "gender" in df.columns else []

    max_reminders = int(df["num_reminders"].max()) if "num_reminders" in df.columns else 0

    min_date = df["appointment_date"].min() if "appointment_date" in df.columns else None
    max_date = df["appointment_date"].max() if "appointment_date" in df.columns else None

    selected_risk = st.sidebar.multiselect("Risk band", risk_options, default=risk_options)
    selected_sentiment = st.sidebar.multiselect("Sentiment", sentiment_options, default=sentiment_options)
    selected_transport = st.sidebar.multiselect("Transport need", transport_options, default=transport_options)
    selected_action = st.sidebar.multiselect("Recommended action", action_options, default=action_options)
    selected_weekday = st.sidebar.multiselect("Weekday", weekday_options, default=weekday_options)
    selected_period = st.sidebar.multiselect("Appointment period", period_options, default=period_options) if period_options else []
    selected_neighbourhood = st.sidebar.multiselect("Clinic / Neighbourhood", neighbourhood_options, default=neighbourhood_options) if neighbourhood_options else []
    selected_channel = st.sidebar.multiselect("Booking channel", channel_options, default=channel_options) if channel_options else []
    selected_age_group = st.sidebar.multiselect("Age group", age_group_options, default=age_group_options) if age_group_options else []
    selected_gender = st.sidebar.multiselect("Gender", gender_options, default=gender_options) if gender_options else []
    reminder_range = st.sidebar.slider("Number of reminders", min_value=0, max_value=max_reminders, value=(0, max_reminders)) if "num_reminders" in df.columns else (0, 0)
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date)) if min_date is not None and max_date is not None else None

    default_threshold = float(df["decision_threshold"].iloc[0]) if "decision_threshold" in df.columns and not df.empty else 0.5
    what_if_threshold = st.sidebar.slider(
        "What-if policy threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(default_threshold),
        step=0.01,
    )

    filtered = df[
        df["risk_band"].isin(selected_risk)
        & df["sentiment_label"].isin(selected_sentiment)
        & df["transport_need"].isin(selected_transport)
        & df["recommended_action"].isin(selected_action)
        & df["appointment_weekday"].isin(selected_weekday)
    ].copy()

    if period_options:
        filtered = filtered[filtered["appointment_period"].isin(selected_period)]

    if neighbourhood_options:
        filtered = filtered[filtered["neighbourhood"].isin(selected_neighbourhood)]

    if channel_options:
        filtered = filtered[filtered["booking_channel"].isin(selected_channel)]

    if age_group_options:
        filtered = filtered[filtered["age_group"].isin(selected_age_group)]

    if gender_options:
        filtered = filtered[filtered["gender"].isin(selected_gender)]

    if "num_reminders" in filtered.columns:
        filtered = filtered[
            (filtered["num_reminders"] >= reminder_range[0])
            & (filtered["num_reminders"] <= reminder_range[1])
        ]

    if date_range and isinstance(date_range, tuple) and len(date_range) == 2 and "appointment_date" in filtered.columns:
        start_date, end_date = date_range
        filtered = filtered[(filtered["appointment_date"] >= start_date) & (filtered["appointment_date"] <= end_date)]

    if "no_show_probability" in filtered.columns:
        filtered["what_if_prediction"] = (filtered["no_show_probability"] >= what_if_threshold).astype(int)

    ui_context = {
        "role": role,
        "what_if_threshold": what_if_threshold,
    }
    return filtered, ui_context


def show_kpis(df: pd.DataFrame, ui_context: dict) -> None:
    total = len(df)
    no_show_rate = float(df["no_show"].mean()) if total else 0.0
    predicted_rate = float(df["no_show_prediction"].mean()) if total and "no_show_prediction" in df.columns else 0.0
    what_if_predicted_rate = float(df["what_if_prediction"].mean()) if total and "what_if_prediction" in df.columns else predicted_rate
    high_risk = int((df["risk_band"] == "high").sum())
    urgent_transport = int((df["transport_need"] == "urgent_support").sum())
    reschedule_count = int((df.get("schedule_status", pd.Series(dtype=str)) == "reschedule_recommended").sum())
    threshold_used = float(df["decision_threshold"].iloc[0]) if total and "decision_threshold" in df.columns else 0.5
    what_if_threshold = float(ui_context.get("what_if_threshold", threshold_used))

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Appointments", f"{total:,}")
    col2.metric("No-show rate", f"{no_show_rate:.1%}")
    col3.metric("Predicted no-show", f"{predicted_rate:.1%}")
    col4.metric("High-risk patients", f"{high_risk:,}")
    col5.metric("Urgent transport", f"{urgent_transport:,}")
    col6.metric("Threshold", f"{threshold_used:.2f}")
    col7.metric("What-if predicted", f"{what_if_predicted_rate:.1%}", delta=f"{(what_if_predicted_rate - predicted_rate):+.1%}")
    st.caption(f"Reschedule recommendations in filtered view: {reschedule_count:,}")
    st.caption(f"Active role: {ui_context.get('role', 'Ops Lead')} | What-if threshold: {what_if_threshold:.2f}")
    st.markdown(
        f"""
        <div>
            <span class='metric-chip'>Risk Bands: Low 0–30%</span>
            <span class='metric-chip'>Medium 30–70%</span>
            <span class='metric-chip'>High &gt;70%</span>
            <span class='metric-chip'>Role: {ui_context.get('role', 'Ops Lead')}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_management_queue(df: pd.DataFrame, role: str) -> None:
    st.subheader("Management Queue")

    queue_columns = [
        "patientid",
        "appointmentid",
        "appointmentday",
        "risk_band",
        "no_show_probability",
        "sentiment_label",
        "transport_need",
        "behavior_segment",
        "recommended_action",
        "travel_time_minutes",
        "transport_distance_km",
    ]

    queue_columns = [col for col in queue_columns if col in df.columns]

    if role == "Scheduler":
        scheduler_cols = [
            "patientid",
            "appointmentid",
            "appointmentday",
            "schedule_status",
            "suggested_appointmentday",
            "suggested_slot",
            "schedule_priority",
            "recommended_action",
        ]
        queue_columns = [col for col in scheduler_cols if col in df.columns]
    elif role == "Care Coordinator":
        coordinator_cols = [
            "patientid",
            "appointmentid",
            "risk_band",
            "no_show_probability",
            "sentiment_label",
            "behavior_segment",
            "recommended_action",
            "schedule_reason",
        ]
        queue_columns = [col for col in coordinator_cols if col in df.columns]

    queue = df[queue_columns].sort_values(by="no_show_probability", ascending=False)

    editor_df = queue.head(1000).copy()
    editor_df.insert(0, "select", False)

    edited = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=True,
        height=440,
        key="bulk_action_editor",
    )

    selected = edited[edited["select"]].drop(columns=["select"], errors="ignore")
    bulk_action = st.selectbox(
        "Bulk action to apply",
        [
            "Send reminder 24h",
            "Send reminder 24h + 2h",
            "Call patient",
            "Arrange transport",
            "Escalate to care coordinator",
            "Confirm reschedule",
        ],
    )

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="Download selected bulk plan",
            data=(selected.assign(bulk_action=bulk_action).to_csv(index=False).encode("utf-8")),
            file_name="bulk_action_plan_selected.csv",
            mime="text/csv",
            disabled=selected.empty,
        )
    with c2:
        if st.button("Save selected bulk plan"):
            if selected.empty:
                st.warning("Select at least one row to create a bulk action plan.")
            else:
                plan = selected.copy()
                plan["bulk_action"] = bulk_action
                plan["plan_timestamp_utc"] = pd.Timestamp.utcnow().isoformat()
                BULK_ACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
                plan.to_csv(BULK_ACTIONS_PATH, index=False)
                st.success(f"Bulk action plan saved: {BULK_ACTIONS_PATH}")

    st.download_button(
        label="Download filtered queue (CSV)",
        data=queue.to_csv(index=False).encode("utf-8"),
        file_name="management_queue_filtered.csv",
        mime="text/csv",
    )


def show_schedule_management(df: pd.DataFrame) -> None:
    st.subheader("Schedule Management")

    if "schedule_status" not in df.columns:
        st.info("No schedule columns found. Re-run scoring pipeline to generate schedule recommendations.")
        return

    schedule_df = df[df["schedule_status"] == "reschedule_recommended"].copy()
    if schedule_df.empty:
        st.success("No appointments currently need rescheduling for selected filters.")
        return

    schedule_columns = [
        "patientid",
        "appointmentid",
        "appointmentday",
        "suggested_appointmentday",
        "suggested_slot",
        "schedule_priority",
        "schedule_reason",
        "risk_band",
        "transport_need",
        "recommended_action",
    ]
    schedule_columns = [col for col in schedule_columns if col in schedule_df.columns]

    schedule_queue = schedule_df[schedule_columns].sort_values(
        by=["schedule_priority", "appointmentday"], ascending=[False, True]
    )

    editor_df = schedule_queue.head(1000).copy()
    editor_df.insert(0, "approve", False)

    edited = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=True,
        height=420,
        key="schedule_approval_editor",
    )

    approved = edited[edited["approve"]].copy()
    approved = approved.drop(columns=["approve"], errors="ignore")

    if not approved.empty:
        st.caption(f"Approved selections: {len(approved):,}")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download approved reschedules (CSV)",
            data=approved.to_csv(index=False).encode("utf-8"),
            file_name="approved_reschedules.csv",
            mime="text/csv",
            disabled=approved.empty,
        )

    with col2:
        if st.button("Save approved schedule to file", type="primary"):
            if approved.empty:
                st.warning("Select at least one row in the approval table before saving.")
            else:
                finalized = approved.copy()
                finalized["approval_timestamp_utc"] = pd.Timestamp.utcnow().isoformat()
                finalized = finalized.drop_duplicates(subset=["appointmentid"], keep="first")
                FINALIZED_SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
                finalized.to_csv(FINALIZED_SCHEDULE_PATH, index=False)
                st.success(f"Finalized schedule saved: {FINALIZED_SCHEDULE_PATH}")

    st.dataframe(schedule_queue.head(1000), use_container_width=True, height=260)
    st.download_button(
        label="Download reschedule queue (CSV)",
        data=schedule_queue.to_csv(index=False).encode("utf-8"),
        file_name="reschedule_queue.csv",
        mime="text/csv",
    )


def show_distributions(df: pd.DataFrame) -> None:
    left, right = st.columns(2)

    with left:
        st.subheader("Risk Distribution")
        risk_counts = df["risk_band"].value_counts().sort_index()
        st.bar_chart(risk_counts)

    with right:
        st.subheader("Recommended Actions")
        action_counts = df["recommended_action"].value_counts().head(10)
        st.bar_chart(action_counts)


def show_analytics_story(df: pd.DataFrame) -> None:
    st.subheader("Operational Analytics Story")

    c1, c2 = st.columns(2)
    with c1:
        if "age_group" in df.columns:
            st.caption("No-show by Age Group")
            age_rate = df.groupby("age_group")["no_show"].mean().sort_values(ascending=False)
            st.bar_chart(age_rate)

        if "gender" in df.columns:
            st.caption("No-show by Gender")
            gender_rate = df.groupby("gender")["no_show"].mean().sort_values(ascending=False)
            st.bar_chart(gender_rate)

        if "sms_received" in df.columns:
            st.caption("No-show by SMS Sent")
            sms_rate = df.groupby("sms_received")["no_show"].mean()
            sms_rate.index = sms_rate.index.map({0: "No SMS", 1: "SMS Sent"})
            st.bar_chart(sms_rate)

    with c2:
        st.caption("No-show by Weekday")
        weekday_rate = df.groupby("appointment_weekday")["no_show"].mean().sort_values(ascending=False)
        st.bar_chart(weekday_rate)

        if "appointment_period" in df.columns:
            st.caption("No-show by Appointment Period")
            period_rate = df.groupby("appointment_period")["no_show"].mean().sort_values(ascending=False)
            st.bar_chart(period_rate)

        if "num_reminders" in df.columns:
            st.caption("No-show by Number of Reminders")
            reminder_rate = df.groupby("num_reminders")["no_show"].mean().sort_index()
            st.line_chart(reminder_rate)

    if "appointment_date" in df.columns:
        st.caption("No-show Trend Over Time")
        trend = df.dropna(subset=["appointment_date"]).groupby("appointment_date")["no_show"].mean()
        st.line_chart(trend)

    if "appointment_period" in df.columns:
        st.caption("Heatmap Table: Weekday vs Appointment Period (No-show Rate)")
        heatmap = pd.pivot_table(
            df,
            values="no_show",
            index="appointment_weekday",
            columns="appointment_period",
            aggfunc="mean",
        )
        st.dataframe(heatmap.style.format("{:.1%}"), use_container_width=True)


def show_policy_simulation(df: pd.DataFrame, ui_context: dict) -> None:
    st.subheader("Policy Simulation")

    if df.empty or "what_if_prediction" not in df.columns:
        st.info("Not enough data for simulation.")
        return

    current_threshold = float(df["decision_threshold"].iloc[0]) if "decision_threshold" in df.columns else 0.5
    what_if_threshold = float(ui_context.get("what_if_threshold", current_threshold))

    baseline_pred = df["no_show_prediction"].astype(int) if "no_show_prediction" in df.columns else pd.Series([0] * len(df))
    what_if_pred = df["what_if_prediction"].astype(int)
    y_true = df["no_show"].astype(int)

    baseline_metrics = _compute_binary_metrics(y_true, baseline_pred)
    what_if_metrics = _compute_binary_metrics(y_true, what_if_pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("Threshold (current)", f"{current_threshold:.2f}")
    c2.metric("Threshold (what-if)", f"{what_if_threshold:.2f}")
    c3.metric("Predicted no-show delta", f"{what_if_pred.mean() - baseline_pred.mean():+.1%}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Precision Δ", f"{(what_if_metrics['precision'] - baseline_metrics['precision']):+.3f}")
    m2.metric("Recall Δ", f"{(what_if_metrics['recall'] - baseline_metrics['recall']):+.3f}")
    m3.metric("F1 Δ", f"{(what_if_metrics['f1'] - baseline_metrics['f1']):+.3f}")

    comparison = pd.DataFrame(
        [
            {
                "policy": "current",
                "threshold": current_threshold,
                "precision": baseline_metrics["precision"],
                "recall": baseline_metrics["recall"],
                "f1": baseline_metrics["f1"],
            },
            {
                "policy": "what_if",
                "threshold": what_if_threshold,
                "precision": what_if_metrics["precision"],
                "recall": what_if_metrics["recall"],
                "f1": what_if_metrics["f1"],
            },
        ]
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)


def show_professional_visualizations(df: pd.DataFrame) -> None:
    st.subheader("Professional Visualizations")

    if df.empty:
        st.info("No data available for visualization.")
        return

    vis_tab1, vis_tab2, vis_tab3 = st.tabs([
        "Trend & Risk",
        "Behavior & Communication",
        "Operations & Actions",
    ])

    with vis_tab1:
        left, right = st.columns(2)

        with left:
            if "appointment_date" in df.columns:
                trend_df = (
                    df.dropna(subset=["appointment_date"])
                    .groupby("appointment_date", as_index=False)["no_show"]
                    .mean()
                    .rename(columns={"no_show": "no_show_rate"})
                )
                if not trend_df.empty:
                    trend_df["appointment_date"] = pd.to_datetime(trend_df["appointment_date"])
                    trend_df = trend_df.sort_values("appointment_date")
                    trend_df["rolling_7d"] = trend_df["no_show_rate"].rolling(7, min_periods=1).mean()

                    line_actual = alt.Chart(trend_df).mark_line(point=True).encode(
                        x=alt.X("appointment_date:T", title="Date"),
                        y=alt.Y("no_show_rate:Q", title="No-show rate"),
                        tooltip=["appointment_date:T", alt.Tooltip("no_show_rate:Q", format=".2%")],
                    ).properties(title="No-show Trend")

                    line_roll = alt.Chart(trend_df).mark_line(strokeDash=[4, 3], color="#ff7f0e").encode(
                        x="appointment_date:T",
                        y=alt.Y("rolling_7d:Q", title="No-show rate"),
                        tooltip=["appointment_date:T", alt.Tooltip("rolling_7d:Q", format=".2%")],
                    )

                    chart = style_altair_chart((line_actual + line_roll).interactive(), height=300)
                    st.altair_chart(chart, use_container_width=True)

        with right:
            if "no_show_probability" in df.columns:
                prob_df = df[["no_show_probability", "no_show"]].copy()
                prob_df["actual"] = prob_df["no_show"].map({0: "Show", 1: "No-Show"})
                hist = alt.Chart(prob_df).mark_bar(opacity=0.65).encode(
                    x=alt.X("no_show_probability:Q", bin=alt.Bin(maxbins=30), title="Predicted probability"),
                    y=alt.Y("count():Q", title="Appointments"),
                    color=alt.Color("actual:N", title="Actual"),
                    tooltip=[alt.Tooltip("count():Q", title="Count")],
                ).properties(title="Prediction Probability Distribution")
                st.altair_chart(style_altair_chart(hist, height=300), use_container_width=True)

        if "appointment_period" in df.columns and "appointment_weekday" in df.columns:
            heat = (
                df.groupby(["appointment_weekday", "appointment_period"], as_index=False)["no_show"]
                .mean()
                .rename(columns={"no_show": "no_show_rate"})
            )
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heatmap = alt.Chart(heat).mark_rect().encode(
                x=alt.X("appointment_period:N", title="Appointment period"),
                y=alt.Y("appointment_weekday:N", sort=weekday_order, title="Weekday"),
                color=alt.Color("no_show_rate:Q", title="No-show rate"),
                tooltip=["appointment_weekday:N", "appointment_period:N", alt.Tooltip("no_show_rate:Q", format=".2%")],
            ).properties(title="Weekday × Appointment Period Heatmap")
            st.altair_chart(style_altair_chart(heatmap, height=260), use_container_width=True)

    with vis_tab2:
        left, right = st.columns(2)

        with left:
            if "waiting_days" in df.columns:
                lead_df = df.copy()
                lead_df["lead_time_bucket"] = pd.cut(
                    lead_df["waiting_days"],
                    bins=[-1, 0, 3, 7, 14, 30, 999],
                    labels=["0", "1-3", "4-7", "8-14", "15-30", "31+"],
                )
                lead_rate = (
                    lead_df.groupby("lead_time_bucket", as_index=False, observed=False)["no_show"]
                    .mean()
                    .rename(columns={"no_show": "no_show_rate"})
                )
                lead_chart = alt.Chart(lead_rate).mark_bar().encode(
                    x=alt.X("lead_time_bucket:N", title="Lead time (days)"),
                    y=alt.Y("no_show_rate:Q", title="No-show rate"),
                    tooltip=["lead_time_bucket:N", alt.Tooltip("no_show_rate:Q", format=".2%")],
                ).properties(title="Lead Time Impact")
                st.altair_chart(style_altair_chart(lead_chart, height=280), use_container_width=True)

        with right:
            if "num_reminders" in df.columns:
                reminder_rate = (
                    df.groupby("num_reminders", as_index=False)["no_show"]
                    .mean()
                    .rename(columns={"no_show": "no_show_rate"})
                )
                reminder_chart = alt.Chart(reminder_rate).mark_line(point=True).encode(
                    x=alt.X("num_reminders:Q", title="Number of reminders"),
                    y=alt.Y("no_show_rate:Q", title="No-show rate"),
                    tooltip=["num_reminders:Q", alt.Tooltip("no_show_rate:Q", format=".2%")],
                ).properties(title="Reminder Effectiveness")
                st.altair_chart(style_altair_chart(reminder_chart, height=280), use_container_width=True)

        if "age_group" in df.columns and "risk_band" in df.columns:
            cohort = df.groupby(["age_group", "risk_band"], as_index=False).size().rename(columns={"size": "count"})
            cohort_chart = alt.Chart(cohort).mark_rect().encode(
                x=alt.X("risk_band:N", title="Risk band"),
                y=alt.Y("age_group:N", title="Age group"),
                color=alt.Color("count:Q", title="Appointments"),
                tooltip=["age_group:N", "risk_band:N", "count:Q"],
            ).properties(title="Age Group × Risk Band Cohort Matrix")
            st.altair_chart(style_altair_chart(cohort_chart, height=260), use_container_width=True)

    with vis_tab3:
        left, right = st.columns(2)

        with left:
            if "recommended_action" in df.columns and "risk_band" in df.columns:
                action_risk = df.groupby(["recommended_action", "risk_band"], as_index=False).size().rename(columns={"size": "count"})
                stacked = alt.Chart(action_risk).mark_bar().encode(
                    x=alt.X("recommended_action:N", title="Recommended action", sort="-y"),
                    y=alt.Y("count:Q", title="Count"),
                    color=alt.Color("risk_band:N", title="Risk band"),
                    tooltip=["recommended_action:N", "risk_band:N", "count:Q"],
                ).properties(title="Action Mix by Risk Band")
                st.altair_chart(style_altair_chart(stacked, height=320), use_container_width=True)

        with right:
            if "transport_distance_km" in df.columns and "no_show_probability" in df.columns:
                scatter_df = df[["transport_distance_km", "no_show_probability", "risk_band"]].dropna().copy()
                if len(scatter_df) > 5000:
                    scatter_df = scatter_df.sample(5000, random_state=42)
                scatter = alt.Chart(scatter_df).mark_circle(size=55, opacity=0.45).encode(
                    x=alt.X("transport_distance_km:Q", title="Transport distance (km)"),
                    y=alt.Y("no_show_probability:Q", title="No-show probability"),
                    color=alt.Color("risk_band:N", title="Risk band"),
                    tooltip=["transport_distance_km:Q", alt.Tooltip("no_show_probability:Q", format=".2f"), "risk_band:N"],
                ).properties(title="Distance vs Predicted Risk")
                st.altair_chart(style_altair_chart(scatter, height=320), use_container_width=True)

        if "schedule_status" in df.columns:
            funnel_source = pd.DataFrame(
                {
                    "stage": [
                        "Total appointments",
                        "Predicted no-show",
                        "High risk",
                        "Reschedule recommended",
                    ],
                    "count": [
                        int(len(df)),
                        int((df.get("no_show_prediction", pd.Series(dtype=int)) == 1).sum()),
                        int((df.get("risk_band", pd.Series(dtype=str)) == "high").sum()),
                        int((df.get("schedule_status", pd.Series(dtype=str)) == "reschedule_recommended").sum()),
                    ],
                }
            )
            funnel = alt.Chart(funnel_source).mark_bar().encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("stage:N", title="Journey stage", sort="-x"),
                tooltip=["stage:N", "count:Q"],
            ).properties(title="Intervention Funnel")
            st.altair_chart(style_altair_chart(funnel, height=250), use_container_width=True)


def show_model_and_explainability() -> None:
    st.subheader("Model Performance, Calibration, and Explainability")

    model_comparison = load_optional_csv(MODEL_COMPARISON_PATH)
    metrics = load_optional_json(METRICS_PATH)
    threshold_info = load_optional_json(THRESHOLD_PATH)
    calibration = load_optional_json(CALIBRATION_PATH)
    importance = load_optional_csv(FEATURE_IMPORTANCE_PATH)
    shap_summary = load_optional_csv(SHAP_SUMMARY_PATH)

    if not model_comparison.empty:
        st.caption("Multi-model comparison")
        st.dataframe(model_comparison, use_container_width=True)

    if metrics:
        cols = st.columns(6)
        cols[0].metric("Best model", str(metrics.get("best_model", "n/a")))
        cols[1].metric("Accuracy", f"{float(metrics.get('accuracy', 0.0)):.3f}")
        cols[2].metric("Precision", f"{float(metrics.get('precision', 0.0)):.3f}")
        cols[3].metric("Recall", f"{float(metrics.get('recall', 0.0)):.3f}")
        cols[4].metric("F1", f"{float(metrics.get('f1', 0.0)):.3f}")
        cols[5].metric("ROC-AUC", f"{float(metrics.get('roc_auc', 0.0)):.3f}")

    t1, t2, t3 = st.columns(3)
    t1.metric("Risk brackets", "Low 0–30 / Med 30–70 / High >70")
    t2.metric("Chosen threshold", f"{float(threshold_info.get('threshold', 0.5)):.2f}")
    t3.metric("Calibration used", "Yes" if calibration.get("calibration_used", False) else "No")

    if calibration:
        st.caption(
            f"Calibration (Brier): raw={float(calibration.get('brier_raw', 0.0)):.4f}, calibrated={float(calibration.get('brier_calibrated', 0.0)):.4f}"
        )

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Feature Importance (Random Forest)")
        if not importance.empty:
            st.dataframe(importance.head(25), use_container_width=True)
        else:
            st.info("Feature importance file not found.")

    with c2:
        st.caption("SHAP Summary (Tree-based model)")
        if not shap_summary.empty:
            st.dataframe(shap_summary.head(25), use_container_width=True)
        else:
            st.info("SHAP summary file not found.")


def show_business_insights() -> None:
    st.subheader("Business Insights and Recommendations")
    insights = load_optional_json(BUSINESS_INSIGHTS_JSON)

    if BUSINESS_INSIGHTS_MD.exists():
        st.markdown(BUSINESS_INSIGHTS_MD.read_text(encoding="utf-8"))

    if insights:
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Overall no-show", f"{float(insights.get('overall_no_show_rate', 0.0)):.1%}")
        i2.metric("SMS no-show", f"{float(insights.get('sms_sent_no_show_rate', 0.0)):.1%}")
        i3.metric("No SMS no-show", f"{float(insights.get('sms_not_sent_no_show_rate', 0.0)):.1%}")
        i4.metric("High-risk share", f"{float(insights.get('high_risk_share', 0.0)):.1%}")

        lead_time = insights.get("lead_time_no_show_rate", {})
        if isinstance(lead_time, dict) and lead_time:
            lead_df = pd.DataFrame(
                {
                    "lead_time_bucket": list(lead_time.keys()),
                    "no_show_rate": [float(v) for v in lead_time.values()],
                }
            )
            st.caption("Lead time impact")
            st.bar_chart(lead_df.set_index("lead_time_bucket"))

    st.caption("Optimization actions")
    st.markdown(
        "- Send reminder 24h before and add second reminder for high-risk appointments.\n"
        "- Overbook selectively in high-risk windows with safeguards.\n"
        "- Prioritize high-risk patients earlier in the day and align transport support."
    )


def show_executive_summary(df: pd.DataFrame) -> None:
    st.subheader("Executive Summary")

    if df.empty:
        st.info("No data available for selected filters.")
        return

    no_show_rate = float(df["no_show"].mean())
    high_risk_count = int((df["risk_band"] == "high").sum()) if "risk_band" in df.columns else 0
    transport_urgent = int((df["transport_need"] == "urgent_support").sum()) if "transport_need" in df.columns else 0
    reschedule_count = int((df.get("schedule_status", pd.Series(dtype=str)) == "reschedule_recommended").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Appointments", f"{len(df):,}")
    c2.metric("No-show rate", f"{no_show_rate:.1%}")
    c3.metric("High-risk", f"{high_risk_count:,}")
    c4.metric("Urgent transport", f"{transport_urgent:,}")

    st.caption(f"Reschedule recommendations: {reschedule_count:,}")

    left, right = st.columns(2)
    with left:
        st.caption("Top 5 High-Risk Appointments")
        high_risk_cols = [
            "appointmentid",
            "patientid",
            "appointmentday",
            "no_show_probability",
            "risk_band",
            "recommended_action",
            "transport_need",
        ]
        high_risk_cols = [col for col in high_risk_cols if col in df.columns]
        high_risk_table = (
            df.sort_values("no_show_probability", ascending=False)[high_risk_cols].head(5)
            if "no_show_probability" in df.columns
            else df[high_risk_cols].head(5)
        )
        st.dataframe(high_risk_table, use_container_width=True, hide_index=True)

    with right:
        st.caption("Top 5 Operational Actions")
        if "recommended_action" in df.columns:
            action_counts = (
                df["recommended_action"]
                .value_counts()
                .head(5)
                .rename_axis("recommended_action")
                .reset_index(name="count")
            )
            st.dataframe(action_counts, use_container_width=True, hide_index=True)
        else:
            st.info("No action data available.")

    if "appointment_date" in df.columns:
        st.caption("Today-focused queue")
        today = pd.Timestamp.utcnow().date()
        today_df = df[df["appointment_date"] == today].copy()
        if today_df.empty:
            st.info("No appointments found for today in current filter scope.")
        else:
            today_cols = [
                "appointmentid",
                "patientid",
                "appointmentday",
                "risk_band",
                "no_show_probability",
                "recommended_action",
                "schedule_status",
            ]
            today_cols = [col for col in today_cols if col in today_df.columns]
            st.dataframe(
                today_df.sort_values("no_show_probability", ascending=False)[today_cols].head(20),
                use_container_width=True,
                hide_index=True,
            )


def main() -> None:
    st.set_page_config(page_title="No-Show Management", layout="wide")
    apply_professional_theme()
    st.markdown("<div class='dashboard-title'>Healthcare No-Show Management Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dashboard-subtitle'>Operational command center for prediction, intervention, and schedule optimization.</div>",
        unsafe_allow_html=True,
    )

    if not SCORED_PATH.exists():
        st.error("Missing scored dataset. Run scoring pipeline first: python -m src.pipelines.run_scoring_pipeline")
        st.stop()

    df = load_data()
    filtered, ui_context = apply_filters(df)

    show_kpis(filtered, ui_context)

    tab_exec, tab_ops, tab_analytics, tab_model, tab_insights = st.tabs(
        ["Executive Summary", "Operations", "Analytics", "Model & Explainability", "Insights"]
    )

    with tab_exec:
        show_executive_summary(filtered)

    with tab_ops:
        show_distributions(filtered)
        show_management_queue(filtered, role=ui_context.get("role", "Ops Lead"))
        show_schedule_management(filtered)

    with tab_analytics:
        show_analytics_story(filtered)
        show_professional_visualizations(filtered)
        show_policy_simulation(filtered, ui_context)

    with tab_model:
        if ui_context.get("role") == "Ops Lead":
            show_model_and_explainability()
        else:
            st.info("Model and explainability view is available for Ops Lead role.")

    with tab_insights:
        show_business_insights()


if __name__ == "__main__":
    main()
