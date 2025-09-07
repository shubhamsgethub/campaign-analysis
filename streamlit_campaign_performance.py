# streamlit_campaign_performance.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.formula.api as smf

# --- 1. Data cleaning and imputation ---
def clean_and_impute(df,
                     date_col='Date',
                     month_col='Month',
                     value_col='Value',
                     carpet_col='Carpet Area',
                     store_open_col='Store Open Date',
                     brand_col='Brand Name'
                    ):
    df = df.copy()
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
    df[carpet_col] = clean_numeric(df[carpet_col])

    # Drop brands >30% missing
    brand_missing = df.groupby(brand_col)['sales'].apply(lambda x: x.isna().mean())
    keep_brands = brand_missing[brand_missing <= 0.3].index
    df = df[df[brand_col].isin(keep_brands)]

    # Sales = 0 before store opens
    if store_open_col in df.columns:
        mask = df[store_open_col].notna() & df[date_col].notna() & (df[date_col] < df[store_open_col])
        df.loc[mask, 'sales'] = 0.0

    # Impute per brand
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

# --- 2. Incrementality computation ---
def run_incrementality_analysis(df, targets, controls, pre_start, post_start, post_end, campaign_start, campaign_end, spend=None):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    # Ensure campaign period is within measurement window
    if not (post_start <= campaign_start <= post_end and post_start <= campaign_end <= post_end):
        st.warning("Campaign period not completely within measurement window. Proceeding but check inputs.")

    results = []
    figs = []

    # Total graph series
    total_actual = pd.Series(dtype=float)
    total_expected = pd.Series(dtype=float)

    for target in targets:
        # Subset
        sub_df = df[df['Brand Name'].isin([target] + controls)]
        if sub_df.empty:
            st.warning(f"Data unavailable for {target} and controls in the date range.")
            continue

        # DiD dummy
        sub_df['D'] = ((sub_df['Date'] >= post_start) & (sub_df['Brand Name']==target)).astype(int)

        # Regression
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

        # Correlation in pre-period
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

        # Sum for total graph
        total_actual = total_actual.add(actual, fill_value=0)
        total_expected = total_expected.add(expected, fill_value=0)

    # Total graph
    fig_total = go.Figure()
    fig_total.add_trace(go.Scatter(x=total_actual.index, y=total_actual.values, name='Actual Total'))
    fig_total.add_trace(go.Scatter(x=total_expected.index, y=total_expected.values, name='Expected Total'))
    fig_total.add_trace(go.Scatter(x=control_series.index, y=control_series.values, name='Control Total'))
    fig_total.update_layout(title="Total Incrementality", xaxis_title='Date', yaxis_title='Sales')

    return results, figs, fig_total

# --- 3. Streamlit app ---
st.title("Mall Campaign Incrementality Analysis")

uploaded_file = st.file_uploader("Upload sales CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = clean_and_impute(df)

    brand_options = df['Brand Name'].unique().tolist()

    # Target & controls
    targets = st.multiselect("Select Target Brands", brand_options)
    controls = st.multiselect("Select Control Brands (optional)", brand_options)

    # Periods
    pre_auto = st.checkbox("Auto PrePeriod (6 months back)")
    if pre_auto:
        pre_start = df['Date'].min()  # Example placeholder: could compute 6 months back from post_start
    else:
        pre_start = st.date_input("PrePeriod Start")

    post_start = st.date_input("Measurement Period Start")
    post_end = st.date_input("Measurement Period End")

    campaign_start = st.date_input("Campaign Start")
    campaign_end = st.date_input("Campaign End")

    spend = st.number_input("Campaign Spend (optional)", min_value=0.0)

    if st.button("Generate Reports"):
        results, figs, fig_total = run_incrementality_analysis(
            df, targets, controls or [brand_options[0]],
            pre_start, post_start, post_end,
            campaign_start, campaign_end,
            spend=spend
        )

        # Show total
        st.plotly_chart(fig_total, use_container_width=True)
        # Show individual
        for _, fig in figs:
            st.plotly_chart(fig, use_container_width=True)

        # Export CSV
        if results:
            summary_df = pd.DataFrame(results)
            st.download_button("Download Summary CSV", summary_df.to_csv(index=False), "summary.csv", "text/csv")
