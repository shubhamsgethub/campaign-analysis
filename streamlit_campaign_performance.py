# streamlit_campaign_performance.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import timedelta

# -----------------------------
# Utility Functions
# -----------------------------

def clean_data(df):
    """Basic cleaning: parse dates and ensure numeric columns."""
    df = df.copy()
    if "Date" not in df.columns:
        st.error("CSV must contain a 'Date' column.")
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("Date").sort_index()

def auto_select_control(df, target, exclude_cols):
    """Pick control with highest absolute correlation to target (excluding targets)."""
    correlations = {}
    for col in df.columns:
        if col not in exclude_cols:
            corr = df[target].corr(df[col])
            if not np.isnan(corr):
                correlations[col] = corr
    if not correlations:
        return None, {}
    best_control = max(correlations, key=lambda x: abs(correlations[x]))
    return best_control, correlations

def run_incrementality_analysis(df, target, control, pre_start, pre_end, camp_start, camp_end):
    """Difference-in-Differences style computation."""
    try:
        target_series = df[target].loc[pre_start:camp_end]
        control_series = df[control].loc[pre_start:camp_end]

        target_pre = target_series.loc[pre_start:pre_end]
        control_pre = control_series.loc[pre_start:pre_end]
        target_post = target_series.loc[camp_start:camp_end]
        control_post = control_series.loc[camp_start:camp_end]

        if target_pre.empty or target_post.empty or control_pre.empty or control_post.empty:
            return {"error": "Data unavailable for the selected range."}

        # Scale control to match target in pre period
        scale = target_pre.mean() / control_pre.mean() if control_pre.mean() != 0 else 1
        expected_post = control_post * scale

        # Incrementality = actual - expected
        incrementality = target_post.sum() - expected_post.sum()
        roi = (incrementality / expected_post.sum()) if expected_post.sum() != 0 else np.nan

        return {
            "actual": target_series,
            "expected": control_series * scale,
            "incrementality": incrementality,
            "roi": roi,
        }
    except Exception as e:
        return {"error": str(e)}

def make_plot(target, res, camp_start, camp_end):
    """Plot individual brand incrementality graph."""
    if "error" in res:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res["expected"].index, y=res["expected"], mode="lines",
        name=f"{target} Expected", line=dict(dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=res["actual"].index, y=res["actual"], mode="lines",
        name=f"{target} Actual"
    ))
    fig.add_vrect(x0=camp_start, x1=camp_end, fillcolor="LightSalmon", opacity=0.3,
                  layer="below", line_width=0)
    fig.update_layout(title=f"Incrementality Report: {target}",
                      xaxis_title="Date", yaxis_title="Sales")
    return fig

def make_total_plot(results, camp_start, camp_end):
    """Plot total combined incrementality graph."""
    dfs = []
    for res in results.values():
        if "error" in res:
            continue
        df = res["actual"].to_frame("actual")
        df["expected"] = res["expected"]
        dfs.append(df)

    if not dfs:
        return None, None, None

    combined = sum(dfs)
    total_incrementality = combined["actual"].sum() - combined["expected"].sum()
    total_roi = (total_incrementality / combined["expected"].sum()) if combined["expected"].sum() != 0 else np.nan

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=combined.index, y=combined["expected"], mode="lines",
        name="Total Expected", line=dict(dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=combined.index, y=combined["actual"], mode="lines",
        name="Total Actual"
    ))
    fig.add_vrect(x0=camp_start, x1=camp_end, fillcolor="LightGreen", opacity=0.3,
                  layer="below", line_width=0)
    fig.update_layout(title="Total Incrementality Report (All Targets Combined)",
                      xaxis_title="Date", yaxis_title="Sales")
    return fig, total_incrementality, total_roi

# -----------------------------
# Streamlit App
# -----------------------------

st.title("📊 Campaign Incrementality Analysis")

uploaded_file = st.file_uploader("Upload sales data CSV", type=["csv"])
if uploaded_file:
    df = clean_data(pd.read_csv(uploaded_file))
    if df is not None:
        st.success("Data uploaded successfully!")

        # Campaign info input
        st.subheader("Campaign Information")
        with st.expander("Upload Campaign CSV (optional)"):
            example_csv = pd.DataFrame({
                "Target": ["Brand1"],
                "CampaignStart": ["2023-07-01"],
                "CampaignEnd": ["2023-07-31"]
            })
            st.download_button("Download Empty Campaign Template", example_csv.to_csv(index=False), "campaign_template.csv")
            camp_file = st.file_uploader("Upload Campaign CSV", type=["csv"], key="camp_csv")
            if camp_file:
                campaigns = pd.read_csv(camp_file)
            else:
                valid_targets = [c for c in df.columns if c not in ["Date"]]
                if not valid_targets:
                    st.error("No valid target columns found in your CSV.")
                else:
                    target = st.selectbox("Select Target Brand", valid_targets)

                camp_start = st.date_input("Campaign Start")
                camp_end = st.date_input("Campaign End")
                campaigns = pd.DataFrame([{"Target": target, "CampaignStart": camp_start, "CampaignEnd": camp_end}])

        # PrePeriod input
        st.subheader("PrePeriod Selection")
        auto_pre = st.checkbox("Auto-select PrePeriod (6 months before campaign start)", value=True)
        manual_pre_start, manual_pre_end = None, None
        if not auto_pre:
            manual_pre_start = st.date_input("PrePeriod Start")
            manual_pre_end = st.date_input("PrePeriod End")

        # Run analysis
        if st.button("Generate Reports"):
            for _, camp in campaigns.iterrows():
                target = camp["Target"]
                camp_start = pd.to_datetime(camp["CampaignStart"])
                camp_end = pd.to_datetime(camp["CampaignEnd"])
                if auto_pre:
                    pre_start = camp_start - timedelta(days=180)
                    pre_end = camp_start - timedelta(days=1)
                else:
                    pre_start = pd.to_datetime(manual_pre_start)
                    pre_end = pd.to_datetime(manual_pre_end)

                # Auto select control
                control, corrs = auto_select_control(df, target, exclude_cols=[target])
                if not control:
                    st.warning(f"No valid control found for {target}. Skipping...")
                    continue

                result = {}
                res = run_incrementality_analysis(df, target, control, pre_start, pre_end, camp_start, camp_end)
                result[target] = res

                # Show correlations
                st.write(f"### {target} - Control Variables Correlations")
                corr_df = pd.DataFrame.from_dict(corrs, orient="index", columns=["Correlation"])
                st.dataframe(corr_df)

                # Individual plots
                fig = make_plot(target, res, camp_start, camp_end)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    if "error" not in res:
                        st.write(f"**Incrementality ({target})**: {res['incrementality']:.2f}")
                        st.write(f"**ROI ({target})**: {res['roi']:.2%}")

            # Combined total
            st.subheader("📊 Combined Impact Across All Targets")
            total_fig, total_inc, total_roi = make_total_plot(result, camp_start, camp_end)
            if total_fig:
                st.plotly_chart(total_fig, use_container_width=True)
                st.write(f"**Total Incrementality**: {total_inc:.2f}")
                st.write(f"**Total ROI**: {total_roi:.2%}")
            else:
                st.warning("No valid results for combined total.")
