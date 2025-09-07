import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
from datetime import timedelta

# ------------------------
# 1. Brand list (for dropdowns)
# ------------------------
brand_options = [
    'Anil Mehndi', 'Atmos', 'Aurelia', 'Avantra', 'BVB',
    'Beijing Bites', 'Bounce & Battery Car', 'Burger King', 'Busters',
    'Busters Soft Play', 'Café Cream', 'Celebrity',
    'Chennai Coffee Shop', 'Chinese Wok', 'Chocolate hut',
    'Coca-Cola Counter', 'Conical Gaufres', 'Cream Stone',
    'Crispy Catch', 'Crocs', 'Essentica', 'Estelle', 'Frankie',
    'Fry Land', 'Gadget Hub', 'Go colors', 'Golkonda Train Restaurant',
    "Haldiram's", 'Health & Glow', 'Hee Fashions',
    'Hishika Collections', 'Hishika Jewels', 'House of Mukwas',
    'Jockey', 'John Players', 'KFC', 'Kaira', 'Krispy kreme',
    'Lee cooper', 'Lifestyle', 'Makers of Milk Shakes', 'Mama Earth',
    'Market 99', 'Miniso', 'Mochi Brand', 'Monte Carlo', 'Movie Max',
    'Oneplus', 'Paradise', 'Pizza Hut',
    'Robotouch Massage Service and Equipment', 'Sizzling Shwarma',
    'Slushes', 'Softy Icecream', 'Squeeze Juice Bars', 'Style union',
    'Sugar', 'The House of Candy', 'Trends Man', 'Trends Women',
    'VIP Bags', 'Vision Express', 'Waffle World',
    'Wrappit Frankies Sharma', 'Zivame', 'Zomoz',
    'helium balloon wala'
]

# ------------------------
# 2. Blank Campaign Template Generator
# ------------------------
def generate_blank_campaign_template():
    cols = ["Campaign Name", "Campaign Start", "Campaign End",
            "Measurement Start", "Measurement End", "Spend",
            "Targets", "Controls", "PrePeriod Start", "PrePeriod End"]
    df = pd.DataFrame(columns=cols)
    return df

# ------------------------
# 3. Sales Computation Helpers
# ------------------------
def compute_expected_sales(df, target_brands, control_brands, pre_start, pre_end, post_start, post_end):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Filter to relevant period
    mask = (df["Date"] >= pre_start) & (df["Date"] <= post_end)
    df = df.loc[mask]

    # Aggregate sales
    sales = df.groupby(["Date", "Brand Name"])["sales_imputed"].sum().unstack().fillna(0)

    # Ensure target & controls exist
    missing_targets = [t for t in target_brands if t not in sales.columns]
    missing_controls = [c for c in control_brands if c not in sales.columns]
    if missing_targets or missing_controls:
        return None, None, None, None, None, f"Data unavailable for: {missing_targets + missing_controls}"

    target_series = sales[target_brands].sum(axis=1)
    control_series = sales[control_brands].sum(axis=1)

    # Correlation in PrePeriod
    corr = target_series.loc[pre_start:pre_end].corr(control_series.loc[pre_start:pre_end])

    # Scale control to match pre-period target
    scale = target_series.loc[pre_start:pre_end].sum() / control_series.loc[pre_start:pre_end].sum()
    expected = control_series * scale

    # Compute metrics in post period
    actual_post = target_series.loc[post_start:post_end].sum()
    expected_post = expected.loc[post_start:post_end].sum()
    incr_sales = actual_post - expected_post
    lift_pct = (incr_sales / expected_post * 100) if expected_post > 0 else np.nan

    return target_series, expected, control_series, lift_pct, incr_sales, corr

def plot_results(target_series, expected, control_series, pre_start, post_start, post_end, campaign_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=target_series.index, y=target_series, mode="lines", name="Actual Target Sales"))
    fig.add_trace(go.Scatter(x=expected.index, y=expected, mode="lines", name="Expected Sales"))
    fig.add_trace(go.Scatter(x=control_series.index, y=control_series, mode="lines", name="Control Sales", line=dict(dash="dot")))

    # Add campaign shading
    fig.add_vrect(x0=post_start, x1=post_end, fillcolor="LightSalmon", opacity=0.3, line_width=0)

    fig.update_layout(title=f"Campaign Report: {campaign_name}",
                      xaxis_title="Date", yaxis_title="Sales")
    return fig

# ------------------------
# 4. Streamlit App
# ------------------------
def main():
    st.title("Mall Marketing Effectiveness (Synthetic DID)")

    # --- Upload Sales Data ---
    st.header("Step 1: Upload Sales Data")
    uploaded_file = st.file_uploader("Upload sales CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Sales data uploaded")

        # --- Campaign Input Section ---
        st.header("Step 2: Provide Campaign Data")
        mode = st.radio("How do you want to input campaign data?",
                        ["Manual Entry", "Upload Campaign CSV"])

        campaigns = []

        if mode == "Upload Campaign CSV":
            campaign_file = st.file_uploader("Upload campaign CSV", type=["csv"])
            st.download_button(
                "📥 Download Blank Campaign Template",
                data=generate_blank_campaign_template().to_csv(index=False).encode("utf-8"),
                file_name="campaign_template.csv",
                mime="text/csv"
            )
            if campaign_file:
                campaigns = pd.read_csv(campaign_file).to_dict(orient="records")
                st.success("✅ Campaign data uploaded")
        else:
            with st.form("campaign_form"):
                name = st.text_input("Campaign Name")
                camp_start = st.date_input("Campaign Start")
                camp_end = st.date_input("Campaign End")
                meas_start = st.date_input("Measurement Start")
                meas_end = st.date_input("Measurement End")
                spend = st.number_input("Spend (₹)", min_value=0.0, step=1000.0)

                targets = st.multiselect("Select Target Brands", brand_options)
                controls = st.multiselect("Select Control Brands (optional)", brand_options)

                # PrePeriod choice
                pre_mode = st.radio("PrePeriod Mode", ["Auto (6 months before Measurement Start)", "Manual"])
                if pre_mode == "Manual":
                    pre_start = st.date_input("PrePeriod Start")
                    pre_end = st.date_input("PrePeriod End")
                else:
                    pre_start = meas_start - timedelta(days=180)
                    pre_end = meas_start - timedelta(days=1)

                submitted = st.form_submit_button("➕ Add Campaign")
                if submitted:
                    if not targets:
                        st.error("❌ Please select at least one target brand")
                    else:
                        campaigns.append({
                            "Campaign Name": name,
                            "Campaign Start": camp_start,
                            "Campaign End": camp_end,
                            "Measurement Start": meas_start,
                            "Measurement End": meas_end,
                            "Spend": spend,
                            "Targets": targets,
                            "Controls": controls if controls else "Auto",
                            "PrePeriod Start": pre_start,
                            "PrePeriod End": pre_end
                        })
                        st.success(f"✅ Added campaign: {name}")

        # --- Generate Reports ---
        if len(campaigns) > 0:
            st.write("📋 Current Campaigns")
            st.dataframe(pd.DataFrame(campaigns))

            if st.button("🚀 Generate Reports"):
                results = []
                for camp in campaigns:
                    st.subheader(f"Report: {camp['Campaign Name']}")

                    # Auto control selection (if none given)
                    if camp["Controls"] == "Auto":
                        all_brands = df["Brand Name"].unique().tolist()
                        potential_controls = [b for b in all_brands if b not in camp["Targets"]]
                        corr_scores = {}
                        for b in potential_controls:
                            t_series, _, c_series, _, _, corr = compute_expected_sales(
                                df, camp["Targets"], [b],
                                camp["PrePeriod Start"], camp["PrePeriod End"],
                                camp["Measurement Start"], camp["Measurement End"]
                            )
                            if corr is not None:
                                corr_scores[b] = abs(corr)
                        best_control = max(corr_scores, key=corr_scores.get)
                        control_brands = [best_control]
                        st.write(f"Auto-selected Control: {best_control} (corr={corr_scores[best_control]:.2f})")
                    else:
                        control_brands = camp["Controls"]

                    target_series, expected, control_series, lift_pct, incr_sales, corr = compute_expected_sales(
                        df, camp["Targets"], control_brands,
                        camp["PrePeriod Start"], camp["PrePeriod End"],
                        camp["Measurement Start"], camp["Measurement End"]
                    )

                    if target_series is None:
                        st.error("❌ Data unavailable for given brands/date range")
                        continue

                    roi = (incr_sales / camp["Spend"]) if camp["Spend"] > 0 else np.nan

                    fig = plot_results(target_series, expected, control_series,
                                       camp["PrePeriod Start"], camp["Measurement Start"], camp["Measurement End"],
                                       camp["Campaign Name"])
                    st.plotly_chart(fig)

                    st.markdown(f"""
                        - **Incremental Sales:** ₹{incr_sales:,.0f}  
                        - **Incrementality (%):** {lift_pct:.2f}%  
                        - **ROI:** {roi:.2f}  
                        - **Correlation (PrePeriod):** {corr:.2f}  
                        - **Controls Used:** {", ".join(control_brands)}
                    """)

                    results.append({
                        "Campaign Name": camp["Campaign Name"],
                        "Incremental Sales": incr_sales,
                        "Incrementality (%)": lift_pct,
                        "ROI": roi,
                        "Correlation": corr,
                        "Controls": ", ".join(control_brands)
                    })

                # Downloadable summary CSV
                results_df = pd.DataFrame(results)
                st.download_button(
                    "📥 Download Campaign Summary CSV",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="campaign_summary.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
