# app_did.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.formula.api as smf
from datetime import timedelta

st.set_page_config(layout="wide", page_title="Campaign Analysis — DiD Regression")

# -------------------------
# 1) Cleaning & imputation (adapted from your function)
# -------------------------
def clean_and_impute(df,
                     date_col='Date',
                     month_col='Month',
                     value_col='Value',
                     carpet_col='Carpet Area',
                     store_open_col='Store Open Date',
                     brand_col='Brand Name'):
    df = df.copy()
    # rename Value -> sales if present
    if value_col in df.columns:
        df = df.rename(columns={value_col: 'sales'})
    # require date col
    if date_col not in df.columns:
        raise ValueError("Upload must contain a Date column named 'Date'.")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    # optional month parsing
    if month_col in df.columns:
        df['month_parsed'] = pd.to_datetime(df[month_col].astype(str).str.replace("'", "").str.strip(),
                                           format='%b%y', errors='coerce')
    if store_open_col in df.columns:
        df[store_open_col] = pd.to_datetime(df[store_open_col], errors='coerce')
    # clean numeric
    def clean_numeric(s):
        return pd.to_numeric(s.astype(str).str.replace(r'[^0-9\.\-]', '', regex=True), errors='coerce')
    if 'sales' in df.columns:
        df['sales'] = clean_numeric(df['sales'])
    if carpet_col in df.columns:
        df[carpet_col] = clean_numeric(df[carpet_col])
    if brand_col not in df.columns:
        raise ValueError(f"Upload must contain a brand column named '{brand_col}'")
    # drop brands >30% missing (on 'sales' if present)
    if 'sales' in df.columns:
        brand_missing = df.groupby(brand_col)['sales'].apply(lambda x: x.isna().mean())
        keep_brands = brand_missing[brand_missing <= 0.3].index.tolist()
        dropped_brands = brand_missing[brand_missing > 0.3]
        df = df[df[brand_col].isin(keep_brands)].copy()
    else:
        dropped_brands = pd.Series([])
    # sales=0 before store opens
    if store_open_col in df.columns and 'sales' in df.columns:
        mask = df[store_open_col].notna() & df[date_col].notna() & (df[date_col] < df[store_open_col])
        df.loc[mask, 'sales'] = 0.0
    # impute if sales_imputed not present
    if 'sales_imputed' not in df.columns:
        def impute_brand(g):
            g = g.sort_values(date_col).set_index(date_col)
            s = g['sales']
            s_interp = s.interpolate(method='time', limit_direction='both')
            s_filled = s_interp.fillna(s.median()).fillna(0)
            g['sales_imputed'] = s_filled
            g['imputed_flag'] = s.isna() & g['sales_imputed'].notna()
            return g.reset_index()
        df = df.groupby(brand_col, group_keys=False).apply(impute_brand).reset_index(drop=True)
    else:
        df['sales_imputed'] = pd.to_numeric(df['sales_imputed'], errors='coerce').fillna(0)
    diag = {
        'rows_after_cleaning': len(df),
        'brands_dropped': int(len(dropped_brands)),
        'remaining_brands': int(df[brand_col].nunique()),
        'missing_after_imputation': int(df['sales_imputed'].isna().sum()) if 'sales_imputed' in df.columns else 0,
        'total_imputed': int(df['imputed_flag'].sum()) if 'imputed_flag' in df.columns else 0
    }
    return df, diag, dropped_brands

# -------------------------
# 2) Pivot helper
# -------------------------
def build_pivot(df):
    df2 = df.copy()
    df2['Date'] = pd.to_datetime(df2['Date'])
    pivot = df2.pivot_table(index='Date', columns='Brand Name', values='sales_imputed', aggfunc='sum')
    pivot = pivot.sort_index().fillna(0)
    return pivot

# -------------------------
# 3) pick top control by pre-period correlation (for display / suggestions)
# -------------------------
def pick_top_control(pivot, target, pre_start, pre_end, exclude=None):
    exclude = exclude or []
    if target not in pivot.columns:
        return None, {}
    candidates = [c for c in pivot.columns if c != target and c not in exclude]
    corrs = {}
    target_pre = pivot[target].loc[pre_start:pre_end]
    if target_pre.empty:
        return None, {}
    for c in candidates:
        c_pre = pivot[c].loc[pre_start:pre_end]
        if c_pre.empty:
            continue
        corr = target_pre.corr(c_pre)
        if not np.isnan(corr):
            corrs[c] = corr
    if not corrs:
        return None, {}
    best = max(corrs, key=lambda k: abs(corrs[k]))
    return best, corrs

# -------------------------
# 4) DiD regression (log model) and counterfactual construction
# -------------------------
def run_did_and_counterfactual(panel_df, targets, controls_pool, pre_start, pre_end, meas_start, meas_end, use_log=True):
    """
    panel_df: long DataFrame with columns ['Date','Brand Name','sales_imputed']
    targets: list of treated brands
    controls_pool: list of brands to be considered controls (if empty, will use all non-treated)
    pre_start/pre_end/meas_start/meas_end: timestamps (inclusive)
    Returns:
      - model summary info (beta, se, p)
      - per-target dictionaries with actual_meas (series), expected_meas (series), incremental, lift%
      - total combined actual & expected series & totals
    """
    df = panel_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    # subset to pre + measurement window (we don't need outside)
    window_mask = (df['Date'] >= pre_start) & (df['Date'] <= meas_end)
    df = df.loc[window_mask].copy()
    if df.empty:
        return {"error": "No data in required windows."}

    # treated indicator (brand-level)
    df['treated'] = df['Brand Name'].isin(targets).astype(int)
    # post indicator for measurement period (1 during measurement, 0 during pre)
    df['post'] = df['Date'].between(meas_start, meas_end).astype(int)
    # interaction
    df['D'] = df['treated'] * df['post']

    # If controls_pool empty, use all non-treated brands
    if not controls_pool:
        controls_pool = [b for b in df['Brand Name'].unique() if b not in targets]

    # Build a control aggregate series (sum of controls) to optionally include as covariate or for plotting
    control_mask = df['Brand Name'].isin(controls_pool)
    # control_sum by date (for plotting)
    control_sum = df.loc[control_mask].groupby('Date')['sales_imputed'].sum().rename('control_sum')

    # add log outcome if requested
    if use_log:
        df['y'] = np.log1p(df['sales_imputed'])
    else:
        df['y'] = df['sales_imputed']

    # fit regression: y ~ D + C(Brand Name) + C(Date)
    # note: we don't add 'treated' or 'post' separately because fixed effects (brand/date) soak them
    formula = 'y ~ D + C(Brand Name) + C(Date)'
    try:
        mod = smf.ols(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['Brand Name']})
    except Exception as e:
        return {"error": f"Regression failed: {e}"}

    # Extract DiD coef
    if 'D' in mod.params:
        beta = float(mod.params['D'])
        se = float(mod.bse['D'])
        pval = float(mod.pvalues['D'])
    else:
        beta, se, pval = np.nan, np.nan, np.nan

    # Build counterfactual predictions: set D=0 for treated rows in measurement period
    df_cf = df.copy()
    df_cf.loc[(df_cf['treated'] == 1) & (df_cf['post'] == 1), 'D'] = 0

    # Predict using model (note: model was trained with categorical Brand & Date; using same df ensures mapping)
    # predicted values in y space (log or level)
    pred_log = mod.predict(df)         # fitted values (with actual D)
    pred_cf_log = mod.predict(df_cf)   # predicted counterfactual (D=0 for treated in post)

    # convert back to sales scale if use_log
    if use_log:
        # expm1 to invert log1p
        df['pred_sales'] = np.expm1(pred_log)
        df_cf['pred_cf_sales'] = np.expm1(pred_cf_log)
    else:
        df['pred_sales'] = pred_log
        df_cf['pred_cf_sales'] = pred_cf_log

    # Now compute per-target actual vs expected during measurement period
    results_per_target = {}
    for t in targets:
        mask_t_meas = (df['Brand Name'] == t) & (df['Date'].between(meas_start, meas_end))
        actual_series = df.loc[mask_t_meas].set_index('Date')['sales_imputed'].sort_index()
        expected_series = df_cf.loc[mask_t_meas].set_index('Date')['pred_cf_sales'].sort_index()
        # If index mismatch (e.g., missing dates), reindex to union and fill zeros for missing actuals
        idx = actual_series.index.union(expected_series.index).sort_values()
        actual_series = actual_series.reindex(idx).fillna(0)
        expected_series = expected_series.reindex(idx).fillna(0)
        actual_total = actual_series.sum()
        expected_total = expected_series.sum()
        incr = actual_total - expected_total
        lift_pct = (incr / expected_total * 100) if expected_total != 0 else np.nan
        # pre-period corr between this target and control_sum
        target_pre = df.loc[(df['Brand Name'] == t) & (df['Date'].between(pre_start, pre_end))].set_index('Date')['sales_imputed']
        control_pre = control_sum.loc[control_sum.index.intersection(target_pre.index)]
        corr_pre = None
        if not target_pre.empty and not control_pre.empty:
            corr_pre = float(target_pre.corr(control_pre))
        results_per_target[t] = {
            "actual_series": actual_series,
            "expected_series": expected_series,
            "actual_total": float(actual_total),
            "expected_total": float(expected_total),
            "incremental_sales": float(incr),
            "lift_pct": float(lift_pct) if not np.isnan(lift_pct) else None,
            "corr_pre": corr_pre
        }

    # Combined (all targets together)
    mask_targets_meas = (df['Brand Name'].isin(targets)) & (df['Date'].between(meas_start, meas_end))
    actual_comb = df.loc[mask_targets_meas].groupby('Date')['sales_imputed'].sum().sort_index()
    expected_comb = df_cf.loc[mask_targets_meas].groupby('Date')['pred_cf_sales'].sum().sort_index()
    idx = actual_comb.index.union(expected_comb.index).sort_values()
    actual_comb = actual_comb.reindex(idx).fillna(0)
    expected_comb = expected_comb.reindex(idx).fillna(0)
    total_actual = actual_comb.sum()
    total_expected = expected_comb.sum()
    total_incr = total_actual - total_expected
    total_lift_pct = (total_incr / total_expected * 100) if total_expected != 0 else np.nan

    # DiD interpretation from log model: beta is in log(sales+1) units -> approx percent = exp(beta)-1
    approx_pct = np.expm1(beta) if use_log else beta

    return {
        "model": mod,
        "beta": beta,
        "beta_se": se,
        "beta_pval": pval,
        "approx_pct_lift_from_beta": approx_pct,
        "per_target": results_per_target,
        "combined_actual_series": actual_comb,
        "combined_expected_series": expected_comb,
        "total_actual": float(total_actual),
        "total_expected": float(total_expected),
        "total_incremental": float(total_incr),
        "total_lift_pct": float(total_lift_pct) if not np.isnan(total_lift_pct) else None,
        "control_sum_series": control_sum
    }

# -------------------------
# 5) plotting helpers
# -------------------------
def plot_combined_series(actual_series, expected_series, control_series, pre_start, meas_start, meas_end, campaign_period, title="Combined"):
    # actual_series and expected_series indexed by Date
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_series.index, y=actual_series.values, mode='lines', name='Actual (targets)'))
    fig.add_trace(go.Scatter(x=expected_series.index, y=expected_series.values, mode='lines', name='Expected (counterfactual)', line=dict(dash='dash')))
    if control_series is not None:
        cs = control_series.reindex(actual_series.index.union(control_series.index)).fillna(0)
        fig.add_trace(go.Scatter(x=cs.index, y=cs.values, mode='lines', name='Control sum', line=dict(dash='dot')))
    # shade measurement/campaign
    camp_start, camp_end = campaign_period
    fig.add_vrect(x0=camp_start, x1=camp_end, fillcolor="LightSalmon", opacity=0.2, line_width=0)
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Sales')
    # restrict x-axis to pre_start..meas_end
    fig.update_xaxes(range=[pd.to_datetime(pre_start), pd.to_datetime(meas_end)])
    return fig

def plot_target_series(actual_series_full, expected_series_meas, control_series_full, pre_start, meas_end, campaign_period, target_name):
    fig = go.Figure()
    # plot full actual history for the target (pre+meas)
    fig.add_trace(go.Scatter(x=actual_series_full.index, y=actual_series_full.values, mode='lines', name=f'{target_name} Actual'))
    # expected only during measurement
    if expected_series_meas is not None and not expected_series_meas.empty:
        fig.add_trace(go.Scatter(x=expected_series_meas.index, y=expected_series_meas.values, mode='lines', name=f'{target_name} Expected', line=dict(dash='dash')))
    if control_series_full is not None:
        fig.add_trace(go.Scatter(x=control_series_full.index, y=control_series_full.values, mode='lines', name='Control sum', line=dict(dash='dot')))
    # campaign shading
    camp_start, camp_end = campaign_period
    fig.add_vrect(x0=camp_start, x1=camp_end, fillcolor="LightSalmon", opacity=0.2, line_width=0)
    fig.update_layout(title=f"{target_name} — Pre + Measurement", xaxis_title='Date', yaxis_title='Sales')
    fig.update_xaxes(range=[pd.to_datetime(pre_start), pd.to_datetime(meas_end)])
    return fig

# -------------------------
# 6) Streamlit UI
# -------------------------
st.title("DiD Campaign Measurement")

st.markdown("""
Upload your cleaned (or raw) sales CSV. Required columns: **Date**, **Brand Name**, and either **Value** or **sales** or **sales_imputed**.
The app will impute missing sales if necessary.
""")

uploaded = st.file_uploader("Upload sales CSV", type=['csv'])
if uploaded is None:
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Clean & impute
try:
    df_clean, diag, dropped = clean_and_impute(raw)
except Exception as e:
    st.error(f"Cleaning failed: {e}")
    st.stop()

st.success("Data cleaned & imputed")
st.write(f"- Rows after cleaning: {diag['rows_after_cleaning']:,}")
st.write(f"- Brands dropped (>30% missing): {diag['brands_dropped']}")
st.write(f"- Remaining brands: {diag['remaining_brands']}")
st.write(f"- Total imputed values: {diag['total_imputed']}")

# Build pivot & panel
pivot = build_pivot(df_clean)
panel = df_clean[['Date','Brand Name','sales_imputed']].copy()

present_brands = sorted(pivot.columns.tolist())

st.markdown("---")
st.header("Campaign input")

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Add campaign manually")
    with st.form("campform"):
        camp_name = st.text_input("Campaign name")
        meas_start = st.date_input("Measurement start (inclusive)", key="mstart")
        meas_end = st.date_input("Measurement end (inclusive)", key="mend")
        camp_start = st.date_input("Campaign start (should be inside measurement window)", key="cstart")
        camp_end = st.date_input("Campaign end", key="cend")
        spend = st.number_input("Campaign spend (optional)", min_value=0.0, value=0.0, step=100.0)
        targets = st.multiselect("Targets (treated brands)", options=present_brands)
        controls = st.multiselect("Controls (optional)", options=present_brands)
        pre_mode = st.radio("Pre-period:", ["Auto (6 months before measurement start)", "Manual"])
        if pre_mode == "Manual":
            pre_start_manual = st.date_input("Pre-period start", key="prestart")
            pre_end_manual = st.date_input("Pre-period end", key="preend")
        submitted = st.form_submit_button("Add campaign")
        if submitted:
            if not targets:
                st.error("Please select at least one target brand.")
            else:
                if pre_mode == "Manual":
                    pre_start = pd.to_datetime(pre_start_manual)
                    pre_end = pd.to_datetime(pre_end_manual)
                else:
                    pre_start = pd.to_datetime(meas_start) - pd.DateOffset(months=6)
                    pre_end = pd.to_datetime(meas_start) - pd.Timedelta(days=1)
                # store in session
                if 'campaigns' not in st.session_state:
                    st.session_state['campaigns'] = []
                st.session_state['campaigns'].append({
                    "Campaign Name": camp_name or "Campaign",
                    "Measurement Start": pd.to_datetime(meas_start),
                    "Measurement End": pd.to_datetime(meas_end),
                    "Campaign Start": pd.to_datetime(camp_start),
                    "Campaign End": pd.to_datetime(camp_end),
                    "PrePeriod Start": pre_start,
                    "PrePeriod End": pre_end,
                    "Spend": float(spend),
                    "Targets": targets,
                    "Controls": controls
                })
                st.success("Campaign added.")

with col2:
    st.subheader("Or upload campaigns CSV")
    template = pd.DataFrame(columns=["Campaign Name","Campaign Start","Campaign End","Measurement Start","Measurement End","PrePeriod Start","PrePeriod End","Spend","Targets","Controls"])
    st.download_button("Download template", data=template.to_csv(index=False).encode('utf-8'), file_name='campaign_template.csv')
    camp_file = st.file_uploader("Upload campaigns CSV (optional)", type=['csv'], key='campupload')
    if camp_file is not None:
        try:
            camp_df = pd.read_csv(camp_file)
            # parse targets/controls as lists
            def to_list(x):
                if pd.isna(x) or x == '':
                    return []
                return [s.strip() for s in str(x).replace("|",";").replace(",",";").split(";") if s.strip()]
            parsed = []
            for _, r in camp_df.iterrows():
                parsed.append({
                    "Campaign Name": r.get("Campaign Name", "Campaign"),
                    "Campaign Start": pd.to_datetime(r.get("Campaign Start")),
                    "Campaign End": pd.to_datetime(r.get("Campaign End")),
                    "Measurement Start": pd.to_datetime(r.get("Measurement Start")),
                    "Measurement End": pd.to_datetime(r.get("Measurement End")),
                    "PrePeriod Start": pd.to_datetime(r.get("PrePeriod Start")) if not pd.isna(r.get("PrePeriod Start")) else None,
                    "PrePeriod End": pd.to_datetime(r.get("PrePeriod End")) if not pd.isna(r.get("PrePeriod End")) else None,
                    "Spend": float(r.get("Spend",0)),
                    "Targets": to_list(r.get("Targets")),
                    "Controls": to_list(r.get("Controls"))
                })
            st.session_state['campaigns_from_file'] = parsed
            st.success(f"Loaded {len(parsed)} campaigns from CSV")
        except Exception as e:
            st.error(f"Could not parse campaigns CSV: {e}")

# show queued campaigns
queued = st.session_state.get('campaigns', []) + st.session_state.get('campaigns_from_file', [])
if queued:
    st.markdown("### Campaigns queued")
    st.dataframe(pd.DataFrame(queued))

# Run DiD
if st.button("Run DiD for queued campaigns"):
    campaigns_to_run = st.session_state.get('campaigns_from_file', []) + st.session_state.get('campaigns', [])
    if not campaigns_to_run:
        st.error("No campaigns queued.")
    else:
        summary_rows = []
        for camp in campaigns_to_run:
            st.header(f"Campaign: {camp['Campaign Name']}")
            # unpack
            meas_start = pd.to_datetime(camp['Measurement Start'])
            meas_end = pd.to_datetime(camp['Measurement End'])
            camp_start = pd.to_datetime(camp['Campaign Start'])
            camp_end = pd.to_datetime(camp['Campaign End'])
            pre_start = camp.get('PrePeriod Start')
            pre_end = camp.get('PrePeriod End')
            spend = float(camp.get('Spend', 0.0))
            targets = camp.get('Targets') or []
            controls_user = camp.get('Controls') or []
            if pre_start is None or pd.isna(pre_start):
                pre_start = meas_start - pd.DateOffset(months=6)
                pre_end = meas_start - pd.Timedelta(days=1)

            # validation
            if not (meas_start <= camp_start <= camp_end <= meas_end):
                st.warning("Campaign period not completely within measurement window. Proceeding but check inputs.")

            # pick control pool
            if controls_user:
                control_pool = [c for c in controls_user if c in present_brands and c not in targets]
                if not control_pool:
                    st.warning("Controls provided are not found in data; will use all non-treated brands as control pool.")
                    control_pool = [b for b in present_brands if b not in targets]
            else:
                control_pool = [b for b in present_brands if b not in targets]

            # run DiD
            res = run_did_and_counterfactual(panel, targets, control_pool, pre_start, pre_end, meas_start, meas_end, use_log=True)
            if 'error' in res:
                st.error(f"Analysis failed: {res['error']}")
                continue

            # show regression results summary
            st.subheader("DiD regression result (log outcome)")
            mod = res['model']
            beta = res['beta']
            st.write(f"- DiD coefficient (beta on interaction D): {beta:.6f}")
            st.write(f"- Approx percent lift (exp(beta)-1): {res['approx_pct_lift_from_beta']:.3%}")
            st.write(f"- SE (clustered by brand): {res['beta_se']:.6f}, p-value: {res['beta_pval']:.4f}")

            # correlation diagnostics: for each target show top control suggestion & corr
            st.subheader("Pre-period diagnostics")
            corr_table_rows = []
            for t in targets:
                top_ctl, corrs = pick_top_control(pivot, t, pre_start, pre_end, exclude=targets)
                if top_ctl:
                    corr_table_rows.append({'Target': t, 'Top control (suggested)': top_ctl, 'Corr (pre)': corrs[top_ctl]})
                else:
                    corr_table_rows.append({'Target': t, 'Top control (suggested)': None, 'Corr (pre)': None})
            st.table(pd.DataFrame(corr_table_rows))

            # Combined plot
            st.subheader("Combined (all targets)")
            fig_comb = plot_combined_series(res['combined_actual_series'], res['combined_expected_series'], res['control_sum_series'], pre_start, meas_start, meas_end, (camp_start, camp_end), title=f"Combined targets — {camp['Campaign Name']}")
            st.plotly_chart(fig_comb, use_container_width=True)

            # combined numbers
            st.markdown(f"- Combined actual (measurement total): ₹{res['total_actual']:.2f}")
            st.markdown(f"- Combined expected (measurement total): ₹{res['total_expected']:.2f}")
            st.markdown(f"- Combined incremental sales: ₹{res['total_incremental']:.2f}")
            st.markdown(f"- Combined lift %: {res['total_lift_pct']:.2f}%")
            st.markdown(f"- Combined ROI (increment/spend): { (res['total_incremental']/spend) if (spend and spend!=0) else 'n/a' }")

            # per-target plots & numbers and per-target reports
            st.subheader("Per-target detail")
            for t, out in res['per_target'].items():
                st.markdown(f"**{t}**")
                if 'error' in out:
                    st.warning(out['error'])
                    continue
                # show timeseries plots (actual full pre+meas and expected during measurement)
                # Need full actual history for target: extract from pivot
                target_full = pivot[t].loc[pre_start:meas_end]
                # expected for measurement (we have out['expected_series'])
                fig_t = plot_target_series(target_full, out['expected_series'], res['control_sum_series'].loc[pre_start:meas_end], pre_start, meas_end, (camp_start, camp_end), t)
                st.plotly_chart(fig_t, use_container_width=True)
                st.markdown(f"- Actual (measurement total): ₹{out['actual_total']:.2f}")
                st.markdown(f"- Expected (measurement total): ₹{out['expected_total']:.2f}")
                st.markdown(f"- Incremental sales: ₹{out['incremental_sales']:.2f}")
                st.markdown(f"- Lift %: {out['lift_pct']:.2f}%")
                st.markdown(f"- Correlation with chosen controls (pre): {out['corr_pre'] if out['corr_pre'] is not None else 'n/a'}")
                # per-target HTML fragment
                frag = pio.to_html(fig_t, include_plotlyjs='cdn', full_html=False)
                html_piece = f"<h1>{camp['Campaign Name']} — {t}</h1><p>Incremental: {out['incremental_sales']:.2f}</p>" + frag
                st.download_button(f"Download HTML for {camp['Campaign Name']} - {t}", data=html_piece, file_name=f"{camp['Campaign Name']}_{t}_report.html", mime="text/html")

            # whole campaign HTML (combined + per-target)
            combined_html = pio.to_html(fig_comb, include_plotlyjs='cdn', full_html=False)
            html_out = f"<h1>Campaign: {camp['Campaign Name']}</h1>"
            html_out += "<h2>Combined</h2>"
            html_out += f"<p>Combined incremental: {res['total_incremental']:.2f}<br>Combined lift%: {res['total_lift_pct']}</p>"
            html_out += combined_html
            for t, out in res['per_target'].items():
                html_out += f"<hr><h2>Target: {t}</h2>"
                if 'error' in out:
                    html_out += f"<p>Error: {out['error']}</p>"
                    continue
                html_out += f"<p>Incremental: {out['incremental_sales']:.2f}<br>Lift%: {out['lift_pct']:.2f}%<br>Correlation pre: {out['corr_pre']}</p>"
            st.download_button(f"Download full campaign HTML: {camp['Campaign Name']}", data=html_out, file_name=f"{camp['Campaign Name'].replace(' ','_')}_report.html", mime="text/html")

            # append summary row
            summary_rows.append({
                "Campaign Name": camp['Campaign Name'],
                "Campaign Period": f"{camp_start.date()} to {camp_end.date()}",
                "Measurement Period": f"{meas_start.date()} to {meas_end.date()}",
                "Targets": ";".join(targets),
                "Controls used": ";".join(control_pool),
                "Incremental Sales": res['total_incremental'],
                "Lift %": res['total_lift_pct'],
                "ROI (increment/spend)": (res['total_incremental']/spend) if (spend and spend!=0) else None,
                "DiD_beta_log": res['beta'],
                "DiD_approx_pct": res['approx_pct_lift_from_beta']
            })

        # end for campaigns
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            st.download_button("Download combined campaigns summary CSV", data=summary_df.to_csv(index=False).encode('utf-8'), file_name='did_campaigns_summary.csv', mime='text/csv')
