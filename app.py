"""
Fitness Tracker Dashboard (Streamlit)

Reads the small, pre-aggregated files produced by the notebook:
    data/daily_master_clean.csv   - one row per user per day
    data/data_quality_report.csv  - rows removed per table and reason
    data/hourly_profile.csv       - per-user, per-hour sums/counts (steps, calories, heart rate)

The raw Fitbit CSVs (millions of heart-rate rows) are NOT needed here, which keeps
the repo small enough for GitHub and Streamlit Community Cloud.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Fitness Tracker Dashboard", page_icon="🏃", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LEVELS = ["Sedentary", "Low active", "Somewhat active", "Active", "Highly active"]
STEP_GOAL = 10_000
SLEEP_GOAL = 7


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_daily(source) -> pd.DataFrame:
    df = pd.read_csv(source, dtype={"Id": str}, parse_dates=["Date"])
    df["Weekday"] = pd.Categorical(df["Date"].dt.day_name(), categories=WEEKDAYS, ordered=True)
    if "ActivityLevel" in df.columns:
        df["ActivityLevel"] = pd.Categorical(df["ActivityLevel"], categories=LEVELS, ordered=True)
    return df


@st.cache_data(show_spinner=False)
def load_optional_csv(source, **kwargs):
    try:
        return pd.read_csv(source, **kwargs)
    except (FileNotFoundError, ValueError):
        return None


def resolve(upload, filename):
    """Prefer an uploaded file, otherwise the file in the repo's data/ folder."""
    if upload is not None:
        return upload
    path = DATA_DIR / filename
    return path if path.exists() else None


# --------------------------------------------------------------------------- #
# Sidebar: optional uploads + filters
# --------------------------------------------------------------------------- #
st.sidebar.title("🏃 Fitness Dashboard")

with st.sidebar.expander("Use your own files (optional)"):
    up_daily = st.file_uploader("daily_master_clean.csv", type="csv", key="daily")
    up_quality = st.file_uploader("data_quality_report.csv", type="csv", key="quality")
    up_hourly = st.file_uploader("hourly_profile.csv", type="csv", key="hourly")

daily_src = resolve(up_daily, "daily_master_clean.csv")
if daily_src is None:
    st.error(
        "`data/daily_master_clean.csv` was not found. Run the export cell in the notebook, "
        "put the CSV files in the `data/` folder of this repo, or upload them from the sidebar."
    )
    st.stop()

daily = load_daily(daily_src)
quality = load_optional_csv(resolve(up_quality, "data_quality_report.csv"))
hourly = load_optional_csv(resolve(up_hourly, "hourly_profile.csv"), dtype={"Id": str})

st.sidebar.header("Filters")

min_d, max_d = daily["Date"].min().date(), daily["Date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
else:  # user is mid-selection
    start, end = min_d, max_d

all_ids = sorted(daily["Id"].unique())
sel_ids = st.sidebar.multiselect("Users", all_ids, default=all_ids, help="Anonymous user IDs")

sel_weekdays = st.sidebar.multiselect("Weekdays", WEEKDAYS, default=WEEKDAYS)

if "ActivityLevel" in daily.columns:
    sel_levels = st.sidebar.multiselect("Activity level", LEVELS, default=LEVELS)
else:
    sel_levels = None

min_steps = st.sidebar.number_input(
    "Exclude days below N steps (device not worn)", min_value=0, max_value=50_000, value=0, step=500,
    help="The notebook suggests filtering incomplete days, e.g. under 1,000 steps.",
)

mask = (
    (daily["Date"].dt.date >= start)
    & (daily["Date"].dt.date <= end)
    & daily["Id"].isin(sel_ids)
    & daily["Weekday"].isin(sel_weekdays)
    & (daily["Steps"] >= min_steps)
)
if sel_levels is not None:
    mask &= daily["ActivityLevel"].isin(sel_levels)
df = daily[mask].copy()

if df.empty:
    st.warning("No data matches the current filters. Widen the filters in the sidebar.")
    st.stop()


# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
st.title("Fitness Tracker Dashboard")
st.caption(
    "Descriptive analysis of cleaned smart-device data (one row per user per day). "
    "Small sample; correlations are not causation."
)

goal_pct = (df["Steps"] >= STEP_GOAL).mean() * 100
c = st.columns(6)
c[0].metric("Users", f"{df['Id'].nunique():,}")
c[1].metric("User-days", f"{len(df):,}")
c[2].metric("Avg daily steps", f"{df['Steps'].mean():,.0f}", help=f"Median {df['Steps'].median():,.0f}")
c[3].metric("Days ≥ 10k steps", f"{goal_pct:.1f}%")
c[4].metric("Avg calories / day", f"{df['Calories'].mean():,.0f}" if df["Calories"].notna().any() else "n/a")
if df["SleepHours"].notna().any():
    c[5].metric("Avg sleep (tracked)", f"{df['SleepHours'].mean():.1f} h",
                help=f"{(df['SleepHours'] < SLEEP_GOAL).mean() * 100:.0f}% of tracked days under {SLEEP_GOAL} h")
else:
    c[5].metric("Avg sleep (tracked)", "n/a")

tab_over, tab_hour, tab_rel, tab_sleep, tab_quality, tab_insights, tab_data = st.tabs(
    ["Overview", "Hourly patterns", "Relationships", "Sleep", "Data quality", "Insights", "Data"]
)

# --------------------------------------------------------------------------- #
# Overview: Visual 1 (steps distribution) + Visual 2 (steps by weekday)
# --------------------------------------------------------------------------- #
with tab_over:
    left, right = st.columns(2)

    with left:
        st.subheader("Distribution of daily steps")
        fig = px.histogram(df, x="Steps", nbins=40, labels={"Steps": "Steps per day"},
                           color_discrete_sequence=["#4C78A8"])
        fig.add_vline(x=df["Steps"].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Mean {df['Steps'].mean():,.0f}", annotation_position="top right")
        fig.add_vline(x=df["Steps"].median(), line_dash="dash", line_color="orange",
                      annotation_text=f"Median {df['Steps'].median():,.0f}", annotation_position="top left")
        fig.add_vline(x=STEP_GOAL, line_dash="dot", line_color="green",
                      annotation_text="10k goal", annotation_position="bottom right")
        fig.update_layout(yaxis_title="User-days", bargap=0.05, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bars far below 10,000 mean most usage is under the common health target. "
                   "Use it to size the under-active audience for coaching features.")

    with right:
        st.subheader("Average steps by weekday")
        g = df.groupby("Weekday", observed=True)["Steps"].agg(["mean", "sem", "count"]).reindex(WEEKDAYS)
        g["ci"] = 1.96 * g["sem"].fillna(0)
        g = g.dropna(subset=["mean"]).reset_index()
        fig = px.bar(g, x="Weekday", y="mean", error_y="ci", color_discrete_sequence=["#4C78A8"],
                     labels={"mean": "Average steps"})
        fig.update_layout(xaxis_title="", margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Error bars = 95% confidence interval. Overlapping bars mean the difference may not be real.")

    st.subheader("Share of user-days by activity level")
    if "ActivityLevel" in df.columns:
        share = (df["ActivityLevel"].value_counts(normalize=True).reindex(LEVELS).fillna(0) * 100).round(1)
        fig = px.bar(x=share.values, y=share.index, orientation="h", text=share.values,
                     labels={"x": "% of user-days", "y": ""}, color_discrete_sequence=["#72B7B2"])
        fig.update_traces(texttemplate="%{text}%")
        fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=LEVELS[::-1]), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Hourly patterns: Visual 3 (steps + calories) and Visual 5 (heart rate)
# --------------------------------------------------------------------------- #
with tab_hour:
    if hourly is None:
        st.info("`hourly_profile.csv` not found. Run the export cell in the notebook to enable this tab.")
    else:
        h = hourly[hourly["Id"].isin(sel_ids)]
        if h.empty:
            st.warning("No hourly data for the selected users.")
        else:
            st.caption("Hourly charts respond to the *Users* filter only (they are built from pre-aggregated hourly totals).")
            g = h.groupby("Hour")[["steps_sum", "steps_n", "cal_sum", "cal_n", "hr_sum", "hr_sumsq", "hr_n"]].sum()
            g = g.reindex(range(24), fill_value=0)
            prof = pd.DataFrame({
                "AvgSteps": g["steps_sum"] / g["steps_n"].replace(0, np.nan),
                "AvgCalories": g["cal_sum"] / g["cal_n"].replace(0, np.nan),
                "AvgHR": g["hr_sum"] / g["hr_n"].replace(0, np.nan),
            })
            var = g["hr_sumsq"] / g["hr_n"].replace(0, np.nan) - prof["AvgHR"] ** 2
            prof["StdHR"] = np.sqrt(var.clip(lower=0))
            prof.index.name = "Hour"

            left, right = st.columns(2)
            with left:
                st.subheader("Hourly activity profile")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=prof.index, y=prof["AvgSteps"], name="Avg steps",
                                         mode="lines+markers", line=dict(color="#4C78A8")))
                fig.add_trace(go.Scatter(x=prof.index, y=prof["AvgCalories"], name="Avg calories",
                                         mode="lines+markers", line=dict(color="#F58518"), yaxis="y2"))
                fig.update_layout(
                    xaxis=dict(title="Hour of day", dtick=1),
                    yaxis=dict(title="Average steps per hour", color="#4C78A8"),
                    yaxis2=dict(title="Average calories per hour", color="#F58518", overlaying="y", side="right"),
                    legend=dict(orientation="h", y=1.12), margin=dict(t=30),
                )
                st.plotly_chart(fig, use_container_width=True)
                if prof["AvgSteps"].notna().any():
                    st.caption(f"Peak step hour: **{int(prof['AvgSteps'].idxmax())}:00**. "
                               "Send reminders about an hour before the low-activity windows.")

            with right:
                st.subheader("Average heart rate by hour")
                if prof["AvgHR"].notna().any():
                    upper, lower = prof["AvgHR"] + prof["StdHR"], prof["AvgHR"] - prof["StdHR"]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=prof.index, y=upper, mode="lines", line=dict(width=0),
                                             showlegend=False, hoverinfo="skip"))
                    fig.add_trace(go.Scatter(x=prof.index, y=lower, mode="lines", line=dict(width=0),
                                             fill="tonexty", fillcolor="rgba(220,20,60,0.15)",
                                             name="±1 std dev", hoverinfo="skip"))
                    fig.add_trace(go.Scatter(x=prof.index, y=prof["AvgHR"], mode="lines+markers",
                                             name="Avg HR", line=dict(color="crimson")))
                    fig.update_layout(xaxis=dict(title="Hour of day", dtick=1),
                                      yaxis_title="Heart rate (bpm)", margin=dict(t=30),
                                      legend=dict(orientation="h", y=1.12))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Lowest values are usually overnight (resting); highest during exercise windows. "
                               "Only users with heart-rate data are included.")
                else:
                    st.info("None of the selected users has heart-rate data.")

# --------------------------------------------------------------------------- #
# Relationships: Visual 4 (steps vs calories) + Visual 7 (correlation heat-map)
# --------------------------------------------------------------------------- #
with tab_rel:
    left, right = st.columns(2)

    with left:
        st.subheader("Steps vs calories (daily)")
        sc = df.dropna(subset=["Steps", "Calories"])
        if len(sc) > 2:
            r = sc["Steps"].corr(sc["Calories"])
            fig = px.scatter(sc, x="Steps", y="Calories", opacity=0.4, color_discrete_sequence=["#4C78A8"],
                             hover_data=["Id", "Date"])
            slope, intercept = np.polyfit(sc["Steps"], sc["Calories"], 1)
            xs = np.array([sc["Steps"].min(), sc["Steps"].max()])
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                                     line=dict(color="red"), name="Trend"))
            fig.update_layout(margin=dict(t=30), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Correlation r = **{r:.2f}**. Vertical spread at the same steps reflects body size "
                       "and non-step exercise.")
        else:
            st.info("Not enough data for this chart.")

    with right:
        st.subheader("Correlation of daily metrics")
        num_cols = [c for c in ["Steps", "Calories", "TotalIntensity", "ActiveMinutes", "VeryActiveMinutes",
                                "AvgHR", "SleepHours", "WeightKg", "BMI"] if c in df.columns]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red = move together, blue = opposite, near 0 = no linear relation. "
                   "Cells are blank when there is too little overlapping data.")

# --------------------------------------------------------------------------- #
# Sleep: Visual 6
# --------------------------------------------------------------------------- #
with tab_sleep:
    sl = df.dropna(subset=["SleepHours"])
    if sl.empty:
        st.info("No sleep data for the current selection.")
    else:
        st.subheader("Hours asleep by weekday")
        fig = px.box(sl, x="Weekday", y="SleepHours", color="Weekday",
                     category_orders={"Weekday": WEEKDAYS}, labels={"SleepHours": "Hours asleep"},
                     color_discrete_sequence=px.colors.sequential.Viridis)
        fig.add_hline(y=SLEEP_GOAL, line_dash="dot", line_color="green", annotation_text="7 h recommended minimum")
        fig.update_layout(showlegend=False, xaxis_title="", margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{sl['Id'].nunique()} users with sleep data, {len(sl):,} tracked days. "
                   "Very short values may be naps or nights split across midnight.")

# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #
with tab_quality:
    if quality is None:
        st.info("`data_quality_report.csv` not found.")
    else:
        raw, clean = int(quality["rows_raw"].sum()), int(quality["rows_clean"].sum())
        q = st.columns(3)
        q[0].metric("Rows before cleaning", f"{raw:,}")
        q[1].metric("Rows after cleaning", f"{clean:,}")
        q[2].metric("Removed", f"{raw - clean:,}", delta=f"-{(raw - clean) / max(raw, 1) * 100:.2f}%",
                    delta_color="off")

        reasons = ["exact_duplicates", "invalid_id_or_timestamp", "invalid_or_inconsistent_values", "duplicate_keys"]
        reasons = [r for r in reasons if r in quality.columns]
        long = quality.melt(id_vars="table", value_vars=reasons, var_name="Reason", value_name="Rows removed")
        fig = px.bar(long, y="table", x="Rows removed", color="Reason", orientation="h",
                     labels={"table": ""})
        fig.update_layout(margin=dict(t=30), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(quality, use_container_width=True, hide_index=True)

    st.subheader("Data completeness (current selection)")
    cols = [c for c in ["Calories", "TotalIntensity", "AvgHR", "SleepHours", "ActiveMinutes",
                        "WeightKg", "BMI"] if c in df.columns]
    comp = (df[cols].notna().mean() * 100).round(1).rename("% non-missing").reset_index()
    comp.columns = ["Column", "% non-missing"]
    fig = px.bar(comp, x="% non-missing", y="Column", orientation="h", range_x=[0, 100],
                 color_discrete_sequence=["#54A24B"])
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
    if "steps_mismatch_flag" in df.columns:
        st.caption(f"{int(df['steps_mismatch_flag'].sum()):,} user-days differ by more than 5% between the "
                   "daily and hourly step files (flagged, not deleted).")

# --------------------------------------------------------------------------- #
# Insights (auto-generated, mirrors the notebook's summary block)
# --------------------------------------------------------------------------- #
with tab_insights:
    by_day = df.groupby("Weekday", observed=True)["Steps"].mean().dropna()
    lines = [
        f"**{df['Id'].nunique()}** users and **{len(df):,}** user-days in the current selection.",
        f"Average / median daily steps: **{df['Steps'].mean():,.0f}** / **{df['Steps'].median():,.0f}**.",
        f"**{goal_pct:.1f}%** of user-days reach the 10,000-step goal.",
        f"Most active weekday: **{by_day.idxmax()}** ({by_day.max():,.0f} steps); "
        f"least active: **{by_day.idxmin()}** ({by_day.min():,.0f} steps).",
    ]
    sc = df.dropna(subset=["Steps", "Calories"])
    if len(sc) > 2:
        lines.append(f"Steps-calories correlation: **{sc['Steps'].corr(sc['Calories']):.2f}**.")
    if df["SleepHours"].notna().any():
        lines.append(f"Average sleep (tracked users): **{df['SleepHours'].mean():.1f} h**, with "
                     f"**{(df['SleepHours'] < SLEEP_GOAL).mean() * 100:.0f}%** of tracked days under {SLEEP_GOAL} h.")
    for line in lines:
        st.markdown(f"- {line}")

    st.subheader("Decisions these numbers support")
    st.markdown(
        "- **Many sedentary / low-active days** → beginner challenges, streaks, gentle reminders\n"
        "- **Weekend vs weekday gap** → weekend-specific campaigns or content\n"
        "- **Hourly peaks** → schedule push notifications ahead of low-activity windows\n"
        "- **Strong steps↔calories link** → steps as headline KPI, add heart rate for accuracy\n"
        "- **Sleep under 7 h on many nights** → bedtime reminders, sleep-score feature\n"
        "- **Low heart-rate / weight / sleep coverage** → encourage night wear and scale syncing"
    )
    st.caption("Limitations: descriptive only, roughly 30 users, and some start/end days may be partial.")

# --------------------------------------------------------------------------- #
# Data explorer
# --------------------------------------------------------------------------- #
with tab_data:
    st.subheader("Filtered data")
    st.dataframe(df.drop(columns=[c for c in ["steps_mismatch_flag"] if c in df.columns]),
                 use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="daily_filtered.csv", mime="text/csv")
