import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io

# =========================
#  Data Cleaning Function
# =========================
def clean_and_impute(df,
                     date_col='Date',
                     month_col='Month',
                     value_col='Value',
                     carpet_col='Carpet Area',
                     store_open_col='Store Open Date',
                     brand_col='Brand Name'):
    df = df.copy()

    # Rename sales column
    df = df.rename(columns={value_col: 'sales'})

    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['month_parsed'] = pd.to_datetime(
        df[month_col].astype(str).str.replace("'", "").str.strip(),
        format='%b%y',
        errors='coerce'
    )
    if store_open_col in df.columns:
        df[store_open_col] = pd.to_datetime(df[store_open_col], errors='coerce')

    # Clean numeric
    def clean_numeric(s):
        return pd.to_numeric(
            s.astype(str).str.replace(r'[^0-9\.\-]', '', regex=True),
            errors='coerce'
        )
    df['sales'] = clean_numeric(df['sales'])
    if carpet_col in df.columns:
        df[carpet_col] = clean_numeric(df[carpet_col])

    # Drop brands with >30% missing
    brand_missing = df.groupby(brand_col)['sales'].apply(lambda x: x.isna().mean())
    keep_brands = brand_missing[brand_missing <= 0.3].index
    df = df[df[brand_col].isin(keep_brands)]

    # Imputation
    def impute_brand(g):
        g = g.sort_values(date_col).set_index(date_col)
        s = g['sales']
        s_interp = s.interpolate(method='time', limit_direction='both')
        s_filled = s_interp.fillna(s.median()).fillna(0)
        g['sales_imputed'] = s_filled
        g['imputed_flag'] = s.isna() & g['sales_imputed'].notna()
        return g.reset_index()

    df = df.groupby(brand_col, group_keys=False).apply(impute_brand).reset_index(drop=True)

    return df


# =========================
#  Incrementality Function
# =========================
def run_incrementality_analysis(df, target, control, pre_start, pre_end,
                                post_start, post_end, camp_start, camp_end):
    # Slice periods
    mask_pre = (df['Date'] >= pre_start) & (df['Date'] <= pre_end) & (df['Brand Name'] == target)
    mask_post = (df['Date'] >= post_start) & (df['Date'] <= post_end) & (df['Brand Name'] == target)
    mask_ctrl = (df['Date'] >= pre_start) & (df['Date'] <= post_end) & (df['Brand Name'] == control)

    if mask_pre.sum() == 0 or mask_post.sum() == 0 or mask_ctrl.sum() == 0:
        return None

    # Series
    target_pre = df.loc[mask_pre, 'sales_imputed']
    target_post = df.loc[mask_post, 'sales_imputed']
    control_series = df.loc[mask_ctrl, ['Date', 'sales_imputed']].set_index('Date')

    # Expected = parallel shift
    scale = target_pre.mean() / control_series.loc[target_pre.index].mean()
    expected = control_series['sales_imputed'] * scale
    expected = expected.loc[target_post.index]

    actual = target_post
    incremental_sales = actual.sum() - expected.sum()
    roi = (incremental_sales / actual.sum()) if actual.sum() > 0 else 0
    corr = target_pre.corr(control_series.loc[target_pre.index, 'sales_imputed'])

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values,
                             mode='lines+markers', name="Actual Sales"))
    fig.add_trace(go.Scatter(x=expected.index, y=expected.values,
                             mode='lines+markers', name="Expected Sales"))
    fig.add_trace(go.Scatter(x=control_series.index, y=control_series['sales_imputed'],
                             mode='lines', name="Control Sales", line=dict(dash='dot')))
    fig.add_vrect(x0=camp_start, x1=camp_end, fillcolor="LightSalmon",
                  opacity=0.3, line_width=0, annotation_text="Campaign Period",
                  annotation_position="top left")

    return {
        "fig": fig,
        "incremental_sales": incremental_sales,
        "roi": roi,
        "correlation": corr,
        "control": control
    }


# =========================
#  Streamlit App
# =========================
st.title("📊 Marketing Incrementality Analysis")

# Upload Data
uploaded = st.file_uploader("Upload sales CSV", type=["csv"])
if uploaded:
    df_raw = pd.read_csv(uploaded)
    df = clean_and_impute(df_raw)
    st.success("✅ Data uploaded and cleaned.")
else:
    st.stop()

# Campaign CSV option
st.subheader("Campaign Info")
st.download_button(
    "⬇️ Download empty campaign template",
    data="Campaign Name,Target,Control,PrePeriodStart,PrePeriodEnd,MeasurementStart,MeasurementEnd,CampaignStart,CampaignEnd,Spend\n",
    file_name="campaign_template.csv",
    mime="text/csv"
)

campaign_file = st.file_uploader("Upload campaign CSV (optional)", type=["csv"])
if campaign_file:
    campaign_data = pd.read_csv(campaign_file)
else:
    st.info("Or enter campaigns manually:")
    campaign_data = pd.DataFrame([{
        "Campaign Name": st.text_input("Campaign Name"),
        "Target": st.selectbox("Target Brand", sorted(df['Brand Name'].unique())),
        "Control": st.selectbox("Control Brand", sorted(df['Brand Name'].unique())),
        "PrePeriodStart": st.date_input("PrePeriod Start"),
        "PrePeriodEnd": st.date_input("PrePeriod End"),
        "MeasurementStart": st.date_input("Measurement Start"),
        "MeasurementEnd": st.date_input("Measurement End"),
        "CampaignStart": st.date_input("Campaign Start"),
        "CampaignEnd": st.date_input("Campaign End"),
        "Spend": st.number_input("Spend", value=0.0, step=100.0)
    }])

# =========================
#  Generate Reports
# =========================
if st.button("Generate Reports"):
    if campaign_data.empty:
        st.error("⚠️ Please upload or enter campaign information.")
    else:
        all_results = []
        for _, camp in campaign_data.iterrows():
            st.subheader(f"📑 {camp['Campaign Name']}")

            result = run_incrementality_analysis(
                df, camp['Target'], camp['Control'],
                pd.to_datetime(camp['PrePeriodStart']),
                pd.to_datetime(camp['PrePeriodEnd']),
                pd.to_datetime(camp['MeasurementStart']),
                pd.to_datetime(camp['MeasurementEnd']),
                pd.to_datetime(camp['CampaignStart']),
                pd.to_datetime(camp['CampaignEnd'])
            )

            if result is None:
                st.warning(f"⚠️ Data unavailable for {camp['Campaign Name']}")
                continue

            # Show chart + KPIs
            st.plotly_chart(result['fig'], use_container_width=True)
            st.markdown(f"""
            - **Incremental Sales:** {result['incremental_sales']:.2f}  
            - **ROI:** {result['roi']:.2f}  
            - **Correlation (Pre-Period):** {result['correlation']:.2f}  
            - **Chosen Control:** {result['control']}  
            """)

            # Save HTML report
            html_buf = io.StringIO()
            result['fig'].write_html(html_buf, include_plotlyjs='cdn')
            st.download_button(
                label="⬇️ Download Report (HTML)",
                data=html_buf.getvalue(),
                file_name=f"{camp['Campaign Name']}_report.html",
                mime="text/html"
            )

            # Collect for CSV
            all_results.append({
                "Campaign Name": camp['Campaign Name'],
                "Incremental Sales": result['incremental_sales'],
                "ROI": result['roi'],
                "Correlation": result['correlation'],
                "Control": result['control'],
                "Spend": camp['Spend']
            })

        if all_results:
            summary_df = pd.DataFrame(all_results)
            st.dataframe(summary_df)
            csv_buf = summary_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Summary CSV",
                data=csv_buf,
                file_name="campaign_summary.csv",
                mime="text/csv"
            )
