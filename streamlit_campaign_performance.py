import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from datetime import timedelta
import plotly.graph_objects as go
import base64
import os

# -----------------------------
# 1. Data cleaning and imputation
# -----------------------------
def clean_and_impute(df,
                     date_col='Date',
                     month_col='Month',
                     value_col='Value',
                     carpet_col='Carpet Area',
                     store_open_col='Store Open Date',
                     brand_col='Brand Name'):
    df = df.copy()
    df = df.rename(columns={value_col: 'sales'})
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
    df[carpet_col] = clean_numeric(df[carpet_col])

    # Drop brands with >30% missing
    brand_missing = df.groupby(brand_col)['sales'].apply(lambda x: x.isna().mean())
    keep_brands = brand_missing[brand_missing <= 0.3].index
    dropped_brands = brand_missing[brand_missing > 0.3]
    df = df[df[brand_col].isin(keep_brands)]

    # Sales=0 before store open
    if store_open_col in df.columns:
        mask = df[store_open_col].notna() & df[date_col].notna() & (df[date_col] < df[store_open_col])
        df.loc[mask, 'sales'] = 0.0

    # Impute missing
    def impute_brand(g):
        g = g.sort_values(date_col).set_index(date_col)
        s = g['sales']
        s_interp = s.interpolate(method='time', limit_direction='both')
        s_filled = s_interp.fillna(s.median()).fillna(0)
        g['sales_imputed'] = s_filled
        g['imputed_flag'] = s.isna() & g['sales_imputed'].notna()
        return g.reset_index()
    df = df.groupby(brand_col, group_keys=False).apply(impute_brand).reset_index(drop=True)

    diag = {
        'rows': len(df),
        'brands_dropped': int(len(dropped_brands)),
        'remaining_brands': int(len(keep_brands)),
        'missing_after_imputation': int(df['sales_imputed'].isna().sum()),
        'total_imputed': int(df['imputed_flag'].sum())
    }
    st.write("Diagnostics:", diag)
    return df

# -----------------------------
# 2. Incrementality + DiD
# -----------------------------
def run_incrementality_analysis(df, targets, controls, pre_start, post_start, post_end, campaign_start, campaign_end, spend=None):
    df = df.copy()
    # Check data availability
    relevant_dates = pd.to_datetime([pre_start, post_start, post_end, campaign_start, campaign_end])
    if df['Date'].min() > relevant_dates.min() or df['Date'].max() < relevant_dates.max():
        st.warning("Data unavailable for some dates in pre/post/campaign period")
        return None

    results = []
    figs = []

    for target in targets:
        # Subset
        sub_df = df[df['Brand Name'].isin([target] + controls)]
        # DiD dummy
        sub_df['D'] = ((sub_df['Date'] >= post_start) & (sub_df['Brand Name']==target)).astype(int)
        # Regression formula
        formula = 'sales_imputed ~ D + C(`Brand Name`) + C(`Date`)'
        try:
            model = smf.ols(formula=formula, data=sub_df).fit(cov_type='cluster', cov_kwds={'groups': sub_df['Brand Name']})
            coef = model.params['D']
        except Exception as e:
            st.warning(f"Regression failed for {target}: {e}")
            continue

        # Expected sales = actual - lift
        actual = sub_df[sub_df['Brand Name']==target].set_index('Date')['sales_imputed'].sort_index()
        expected = actual - coef

        # Incremental lift %
        lift_pct = ((actual.sum() - expected.sum()) / expected.sum()) * 100
        roi = lift_pct / spend if spend else np.nan

        # Correlation with controls in pre period
        pre_mask = (sub_df['Date'] >= pre_start) & (sub_df['Date'] < post_start)
        control_series = sub_df[sub_df['Brand Name'].isin(controls)].groupby('Date')['sales_imputed'].sum()
        target_series = sub_df[(sub_df['Brand Name']==target) & pre_mask].set_index('Date')['sales_imputed']
        corr = target_series.corr(control_series.loc[target_series.index])

        results.append({
            'Target': target,
            'Lift_pct': lift_pct,
            'Incremental': actual.sum()-expected.sum(),
            'ROI': roi,
            'Controls': ", ".join(controls),
            'PreCorr': corr
        })

        # Plot individual
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual.index, y=actual.values, name='Actual Sales'))
        fig.add_trace(go.Scatter(x=expected.index, y=expected.values, name='Expected Sales'))
        fig.add_trace(go.Scatter(x=control_series.index, y=control_series.values, name='Control Total'))
        fig.update_layout(title=f"{target} Incrementality", xaxis_title='Date', yaxis_title='Sales')
        figs.append((target, fig))

    return results, figs

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.title("Mall Campaign Incrementality Analysis")

uploaded_file = st.file_uploader("Upload Sales CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = clean_and_impute(df)

    brand_options = sorted(df['Brand Name'].unique())

    st.header("Campaign Input")
    targets = st.multiselect("Select Target Brands", options=brand_options)
    controls = st.multiselect("Select Control Brands (optional, leave blank for auto)", options=brand_options)

    pre_auto = st.checkbox("Auto PrePeriod (6 months back from measurement start)", value=True)
    pre_start_manual = st.date_input("Manual PrePeriod Start", value=pd.to_datetime("2023-01-01")) if not pre_auto else None

    post_start = st.date_input("Measurement Start")
    post_end = st.date_input("Measurement End")
    campaign_start = st.date_input("Campaign Start")
    campaign_end = st.date_input("Campaign End")
    spend = st.number_input("Campaign Spend (optional)", value=0.0)

    if st.button("Generate Reports"):
        pre_start = post_start - pd.DateOffset(months=6) if pre_auto else pd.to_datetime(pre_start_manual)
        if not targets:
            st.warning("Please select at least one target brand")
        else:
            results, figs = run_incrementality_analysis(df, targets, controls or [brand_options[0]], pre_start, post_start, post_end, campaign_start, campaign_end, spend=spend)
            if results:
                st.subheader("Campaign Summary")
                st.table(pd.DataFrame(results))
                st.subheader("Plots")
                for t, fig in figs:
                    st.plotly_chart(fig, use_container_width=True)
