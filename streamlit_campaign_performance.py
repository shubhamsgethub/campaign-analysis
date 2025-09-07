# app_did_reports.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.formula.api as smf
from datetime import timedelta

st.set_page_config(page_title="Campaign DiD Reports", layout="wide")

# -----------------------
# 0. Utilities
# -----------------------
def parse_list_field(x):
    """Parse a CSV cell listing items separated by ; , or | into a list of trimmed strings."""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x)
    parts = [p.strip() for p in s.replace("|", ";").replace(",", ";").split(";") if p.strip()]
    return parts

# -----------------------
# 1. Cleaning & imputation (your function, robustified)
# -----------------------
def clean_and_impute(df,
                     date_col='Date',
                     month_col='Month',
                     value_col='Value',
                     carpet_col='Carpet Area',
                     store_open_col='Store Open Date',
                     brand_col='Brand Name'):
    df = df.copy()

    # rename provided sales/value column if present
    if value_col in df.columns and 'sales' not in df.columns:
        df = df.rename(columns={value_col: 'sales'})

    # must have Date & Brand Name
    if date_col not in df.columns or brand_col not in df.columns:
        raise ValueError("Uploaded CSV must include 'Date' and 'Brand Name' columns.")

    # parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
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

    # drop brands with >30% missing sales (only if sales exists)
    if 'sales' in df.columns:
        brand_missing = df.groupby(brand_col)['sales'].apply(lambda x: x.isna().mean())
        keep_brands = brand_missing[brand_missing <= 0.3].index.tolist()
        dropped_brands = brand_missing[brand_missing > 0.3]
        df = df[df[brand_col].isin(keep_brands)].copy()
    else:
        dropped_brands = pd.Series([])

    # sales = 0 before store open if store_open_col provided
    if store_open_col in df.columns and 'sales' in df.columns:
        mask = df[store_open_col].notna() & df[date_col].notna() & (df[date_col] < df[store_open_col])
        df.loc[mask, 'sales'] = 0.0

    # If 'sales_imputed' already present, use it (ensure numeric). Otherwise impute per brand.
    if 'sales_imputed' in df.columns:
        df['sales_imputed'] = pd.to_numeric(df['sales_imputed'], errors='coerce').fillna(0.0)
        df['imputed_flag'] = False
    else:
        def impute_brand(g):
            g = g.sort_values(date_col).set_index(date_col)
            s = g['sales'] if 'sales' in g else pd.Series(dtype=float)
            s = s.astype(float)
            s_interp = s.interpolate(method='time', limit_direction='both')
            s_filled = s_interp.fillna(s.median()).fillna(0.0)
            g['sales_imputed'] = s_filled
            g['imputed_flag'] = s.isna() & g['sales_imputed'].notna()
            return g.reset_index()
        df = df.groupby(brand_col, group_keys=False).apply(impute_brand).reset_index(drop=True)

    # Diagnostics
    diag = {
        'rows_after_cleaning': int(len(df)),
        'brands_dropped': int(len(dropped_brands)),
        'remaining_brands': int(df[brand_col].nunique()),
        'missing_after_imputation': int(df['sales_imputed'].isna().sum()),
        'total_imputed': int(df['imputed_flag'].sum()) if 'imputed_flag' in df.columns else 0
    }
    return df, diag, dropped_brands

# -----------------------
# 2. Helpers: pivot & control picker
# -----------------------
def build_pivot(df, date_col='Date', brand_col='Brand Name', value_col='sales_imputed'):
    df2 = df.copy()
    df2[date_col] = pd.to_datetime(df2[date_col])
    pivot = df2.pivot_table(index=date_col, columns=brand_col, values=value_col, aggfunc='sum')
    pivot = pivot.sort_index().fillna(0)
    return pivot

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

# -----------------------
# 3. DiD regression + counterfactual builder
# -----------------------
def run_did_and_build_counterfactual(panel_df, targets, control_pool, pre_start, pre_end, meas_start, meas_end, use_log=True):
    """
    panel_df: DataFrame long format with columns ['Date','Brand Name','sales_imputed']
    targets: list of treated brands
    control_pool: list of candidate controls (brands)
    returns: dict with model, per_target results, combined series and totals
    """
    df = panel_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    # restrict to pre + measurement window (we may need brand/date fixed effects)
    window_mask = (df['Date'] >= pre_start) & (df['Date'] <= meas_end)
    df = df.loc[window_mask].copy()
    if df.empty:
        return {"error": "No data in the selected pre+measurement window."}

    # define treated & post dummies
    df['treated'] = df['Brand Name'].isin(targets).astype(int)
    df['post'] = df['Date'].between(meas_start, meas_end).astype(int)
    df['D'] = df['treated'] * df['post']

    # control pool fallback
    if not control_pool:
        control_pool = [b for b in df['Brand Name'].unique() if b not in targets]

    # aggregate control sum (for diagnostics/plot)
    control_sum = df.loc[df['Brand Name'].isin(control_pool)].groupby('Date')['sales_imputed'].sum().rename('control_sum')

    # outcome for regression (log or level)
    if use_log:
        df['y'] = np.log1p(df['sales_imputed'])
    else:
        df['y'] = df['sales_imputed']

    # regression formula: D + brand FE + date FE
    # Use backticks for column names with spaces
    formula = 'y ~ D + C(`Brand Name`) + C(`Date`)'
    try:
        mod = smf.ols(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['Brand Name']})
    except Exception as e:
        return {"error": f"Regression failed: {e}"}

    # Build counterfactual: set D=0 for treated rows in measurement period
    df_cf = df.copy()
    df_cf.loc[(df_cf['treated'] == 1) & (df_cf['post'] == 1), 'D'] = 0

    # predictions (on y scale)
    pred_y = mod.predict(df)
    pred_cf_y = mod.predict(df_cf)

    # convert back to sales scale
    if use_log:
        df['pred_sales'] = np.expm1(pred_y)
        df_cf['pred_cf_sales'] = np.expm1(pred_cf_y)
    else:
        df['pred_sales'] = pred_y
        df_cf['pred_cf_sales'] = pred_cf_y

    # Per-target results
    per_target = {}
    for t in targets:
        mask_t = (df['Brand Name'] == t)
        # measurement slice
        meas_mask = mask_t & (df['Date'].between(meas_start, meas_end))
        actual_series = df.loc[meas_mask, ['Date', 'sales_imputed']].set_index('Date').sort_index()['sales_imputed']
        expected_series = df_cf.loc[meas_mask, ['Date', 'pred_cf_sales']].set_index('Date').sort_index()['pred_cf_sales']
        # align indices
        idx = actual_series.index.union(expected_series.index).sort_values()
        actual_series = actual_series.reindex(idx).fillna(0.0)
        expected_series = expected_series.reindex(idx).fillna(0.0)
        actual_total = actual_series.sum()
        expected_total = expected_series.sum()
        incremental = actual_total - expected_total
        lift_pct = (incremental / expected_total * 100) if expected_total != 0 else None

        # pre-period correlation with control_sum (alignment)
        t_pre = df.loc[(df['Brand Name'] == t) & (df['Date'].between(pre_start, pre_end))].set_index('Date')['sales_imputed']
        control_pre = control_sum.loc[control_sum.index.intersection(t_pre.index)]
        corr_pre = float(t_pre.corr(control_pre)) if (not t_pre.empty and not control_pre.empty) else None

        per_target[t] = {
            'actual_series': actual_series,
            'expected_series': expected_series,
            'actual_total': float(actual_total),
            'expected_total': float(expected_total),
            'incremental': float(incremental),
            'lift_pct': float(lift_pct) if lift_pct is not None else None,
            'corr_pre': corr_pre
        }

    # Combined across targets
    mask_targets_meas = (df['Brand Name'].isin(targets)) & (df['Date'].between(meas_start, meas_end))
    actual_comb = df.loc[mask_targets_meas].groupby('Date')['sales_imputed'].sum().sort_index()
    expected_comb = df_cf.loc[mask_targets_meas].groupby('Date')['pred_cf_sales'].sum().sort_index()
    # reindex
    idx_all = actual_comb.index.union(expected_comb.index).sort_values()
    actual_comb = actual_comb.reindex(idx_all).fillna(0.0)
    expected_comb = expected_comb.reindex(idx_all).fillna(0.0)
    total_actual = float(actual_comb.sum())
    total_expected = float(expected_comb.sum())
    total_incremental = float(total_actual - total_expected)
    total_lift_pct = float(total_incremental / total_expected * 100) if total_expected != 0 else None

    # DiD beta interpretation on log-scale
    beta = float(mod.params['D']) if 'D' in mod.params else None
    beta_se = float(mod.bse['D']) if 'D' in mod.bse else None
    beta_pval = float(mod.pvalues['D']) if 'D' in mod.pvalues else None
    approx_pct = np.expm1(beta) if use_log and beta is not None else beta

    return {
        'model': mod,
        'beta': beta,
        'beta_se': beta_se,
        'beta_pval': beta_pval,
        'approx_pct': approx_pct,
        'per_target': per_target,
        'actual_combined': actual_comb,
        'expected_combined': expected_comb,
        'total_actual': total_actual,
        'total_expected': total_expected,
        'total_incremental': total_incremental,
        'total_lift_pct': total_lift_pct,
        'control_sum_series': control_sum
    }

# -----------------------
# 4. Plot helpers & HTML report builder
# -----------------------
def plot_target(out, pre_start, meas_end, campaign_start, campaign_end, target_name):
    if out is None or 'actual_series' not in out:
        return None
    # full actual history for plot: actual pre+meas from pivot would be ideal; here we plot measurement actual + expected and control
    fig = go.Figure()
    # actual measurement
    fig.add_trace(go.Scatter(x=out['actual_series'].index, y=out['actual_series'].values, mode='lines+markers', name='Actual (measurement)'))
    # expected measurement
    fig.add_trace(go.Scatter(x=out['expected_series'].index, y=out['expected_series'].values, mode='lines+markers', name='Expected (cf)', line=dict(dash='dash')))
    # campaign shading
    fig.add_vrect(x0=campaign_start, x1=campaign_end, fillcolor="LightSalmon", opacity=0.2, line_width=0)
    fig.update_layout(title=f"{target_name} — Measurement", xaxis_title='Date', yaxis_title='Sales')
    fig.update_xaxes(range=[pd.to_datetime(pre_start), pd.to_datetime(meas_end)])
    return fig

def plot_combined(actual_comb, expected_comb, control_sum, pre_start, meas_end, campaign_start, campaign_end):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_comb.index, y=actual_comb.values, mode='lines', name='Actual (targets sum)'))
    fig.add_trace(go.Scatter(x=expected_comb.index, y=expected_comb.values, mode='lines', name='Expected (cf sum)', line=dict(dash='dash')))
    if control_sum is not None:
        cs = control_sum.reindex(actual_comb.index.union(control_sum.index)).fillna(0.0)
        fig.add_trace(go.Scatter(x=cs.index, y=cs.values, mode='lines', name='Control sum', line=dict(dash='dot')))
    fig.add_vrect(x0=campaign_start, x1=campaign_end, fillcolor="LightGreen", opacity=0.2, line_width=0)
    fig.update_layout(title="Combined targets — Pre + Measurement", xaxis_title='Date', yaxis_title='Sales')
    fig.update_xaxes(range=[pd.to_datetime(pre_start), pd.to_datetime(meas_end)])
    return fig

def make_campaign_html(campaign, res, per_target_figs):
    """Return HTML string for campaign containing combined + per-target plots and metrics."""
    title = campaign['Campaign Name']
    html_parts = [f"<h1>Campaign: {title}</h1>"]
    html_parts.append("<h2>Summary metrics</h2><ul>")
    html_parts.append(f"<li>Campaign Period: {campaign['Campaign Start'].date()} to {campaign['Campaign End'].date()}</li>")
    html_parts.append(f"<li>Measurement Period: {campaign['Measurement Start'].date()} to {campaign['Measurement End'].date()}</li>")
    html_parts.append(f"<li>Spend: {campaign.get('Spend', 0)}</li>")
    html_parts.append(f"<li>Targets: {', '.join(campaign['Targets'])}</li>")
    html_parts.append(f"<li>Controls used: {', '.join(campaign.get('ControlsUsed', []))}</li>")
    html_parts.append("</ul>")

    # Combined metrics
    html_parts.append("<h3>Combined totals (measurement)</h3><ul>")
    html_parts.append(f"<li>Actual: {res['total_actual']:.2f}</li>")
    html_parts.append(f"<li>Expected (cf): {res['total_expected']:.2f}</li>")
    html_parts.append(f"<li>Incremental: {res['total_incremental']:.2f}</li>")
    html_parts.append(f"<li>Lift %: {res['total_lift_pct']:.2f}%</li>")
    html_parts.append(f"<li>DiD beta (log model): {res['beta']:.6f} &nbsp; approx pct (exp(beta)-1): {res['approx_pct']:.3%}</li>")
    html_parts.append("</ul>")

    # Combined plot
    combined_html = pio.to_html(plot_combined(res['actual_combined'], res['expected_combined'], res['control_sum_series'],
                                              campaign['PrePeriod Start'], campaign['Measurement End'],
                                              campaign['Campaign Start'], campaign['Campaign End']),
                                 include_plotlyjs='cdn', full_html=False)
    html_parts.append("<h2>Combined plot</h2>")
    html_parts.append(combined_html)

    # Per-target sections
    for t in campaign['Targets']:
        out = res['per_target'].get(t, {})
        html_parts.append(f"<hr><h2>Target: {t}</h2>")
        if 'incremental' not in out:
            html_parts.append("<p>Error / no data for this target</p>")
            continue
        html_parts.append("<ul>")
        html_parts.append(f"<li>Actual (measurement total): {out['actual_total']:.2f}</li>")
        html_parts.append(f"<li>Expected (measurement total): {out['expected_total']:.2f}</li>")
        html_parts.append(f"<li>Incremental: {out['incremental']:.2f}</li>")
        html_parts.append(f"<li>Lift %: {out['lift_pct']:.2f}%</li>")
        html_parts.append(f"<li>Correlation (pre) with control-sum: {out['corr_pre'] if out['corr_pre'] is not None else 'n/a'}</li>")
        html_parts.append("</ul>")
        if t in per_target_figs and per_target_figs[t] is not None:
            html_parts.append(pio.to_html(per_target_figs[t], include_plotlyjs=False, full_html=False))

    html = "<html><head><meta charset='utf-8'></head><body>" + "\n".join(html_parts) + "</body></html>"
    return html

# -----------------------
# 5. Streamlit UI
# -----------------------
st.title("Campaign Measurement — DiD (regression) with reports")

st.markdown("""
**Upload sales CSV** (must include `Date`, `Brand Name`, and either `Value`/`sales` or `sales_imputed`).  
App will clean + impute if needed and then allow campaign analysis.
""")

uploaded = st.file_uploader("1) Upload sales CSV", type=["csv"])
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
    st.error(f"Cleaning/imputation failed: {e}")
    st.stop()

st.success("Data cleaned and imputed.")
st.markdown(f"- Rows after cleaning: {diag['rows_after_cleaning'] if 'rows_after_cleaning' in diag else diag.get('rows_after_cleaning', diag.get('rows_after_cleaning', len(df_clean)))}")
st.markdown(f"- Remaining brands: {diag['remaining_brands']}")
st.markdown(f"- Total imputed values: {diag['total_imputed']}")

# Build pivot and present brands
pivot = build_pivot(df_clean)
present_brands = sorted(pivot.columns.tolist())

st.markdown("---")
st.header("2) Campaign input")

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Add campaign manually")
    with st.form("camp_form"):
        cname = st.text_input("Campaign name")
        meas_start = st.date_input("Measurement start (inclusive)")
        meas_end = st.date_input("Measurement end (inclusive)")
        camp_start = st.date_input("Campaign start (should be inside measurement)")
        camp_end = st.date_input("Campaign end")
        spend = st.number_input("Spend (optional)", min_value=0.0, value=0.0, step=100.0)
        targets_sel = st.multiselect("Targets (one or more)", options=present_brands)
        controls_sel = st.multiselect("Controls (optional; leave blank to auto-select)", options=present_brands)
        pre_mode = st.radio("Pre-period:", ["Auto (6 months before measurement start)", "Manual"])
        if pre_mode == "Manual":
            pre_start_manual = st.date_input("Pre-period start")
            pre_end_manual = st.date_input("Pre-period end")
        submitted = st.form_submit_button("Add campaign")
        if submitted:
            if not targets_sel:
                st.error("Please pick at least one target.")
            else:
                if pre_mode == "Manual":
                    pre_start_val = pd.to_datetime(pre_start_manual)
                    pre_end_val = pd.to_datetime(pre_end_manual)
                else:
                    pre_start_val = pd.to_datetime(meas_start) - pd.DateOffset(months=6)
                    pre_end_val = pd.to_datetime(meas_start) - pd.Timedelta(days=1)
                # ensure date types -> pd.Timestamp
                campaign_obj = {
                    "Campaign Name": cname or "Campaign",
                    "Campaign Start": pd.to_datetime(camp_start),
                    "Campaign End": pd.to_datetime(camp_end),
                    "Measurement Start": pd.to_datetime(meas_start),
                    "Measurement End": pd.to_datetime(meas_end),
                    "PrePeriod Start": pre_start_val,
                    "PrePeriod End": pre_end_val,
                    "Spend": float(spend),
                    "Targets": targets_sel,
                    "Controls": controls_sel or []
                }
                if "campaigns" not in st.session_state:
                    st.session_state['campaigns'] = []
                st.session_state['campaigns'].append(campaign_obj)
                st.success("Campaign added to queue.")

with col2:
    st.subheader("Or upload campaigns CSV")
    # provide template
    template_df = pd.DataFrame([{
        "Campaign Name": "My Campaign",
        "Campaign Start": "2023-10-01",
        "Campaign End": "2023-10-14",
        "Measurement Start": "2023-09-01",
        "Measurement End": "2023-10-31",
        "PrePeriod Start": "",
        "PrePeriod End": "",
        "Spend": "",
        "Targets": "BVB;Frankie",
        "Controls": "Burger King;KFC"
    }])
    st.download_button("Download campaign template (CSV)", data=template_df.to_csv(index=False).encode('utf-8'),
                       file_name="campaign_template.csv", mime="text/csv")
    camp_file = st.file_uploader("Upload campaigns CSV (optional)", type=["csv"], key="camp_upload")
    if camp_file is not None:
        try:
            camp_df = pd.read_csv(camp_file)
            parsed = []
            for _, r in camp_df.iterrows():
                parsed.append({
                    "Campaign Name": r.get("Campaign Name", "Campaign"),
                    "Campaign Start": pd.to_datetime(r.get("Campaign Start")),
                    "Campaign End": pd.to_datetime(r.get("Campaign End")),
                    "Measurement Start": pd.to_datetime(r.get("Measurement Start")),
                    "Measurement End": pd.to_datetime(r.get("Measurement End")),
                    "PrePeriod Start": pd.to_datetime(r.get("PrePeriod Start")) if not pd.isna(r.get("PrePeriod Start")) and r.get("PrePeriod Start") != "" else None,
                    "PrePeriod End": pd.to_datetime(r.get("PrePeriod End")) if not pd.isna(r.get("PrePeriod End")) and r.get("PrePeriod End") != "" else None,
                    "Spend": float(r.get("Spend", 0)) if not pd.isna(r.get("Spend", 0)) else 0.0,
                    "Targets": parse_list_field(r.get("Targets")),
                    "Controls": parse_list_field(r.get("Controls"))
                })
            st.session_state['campaigns_from_file'] = parsed
            st.success(f"Loaded {len(parsed)} campaigns from uploaded CSV")
        except Exception as e:
            st.error(f"Could not parse campaign CSV: {e}")

# show queued campaigns
queued = st.session_state.get('campaigns', []) + st.session_state.get('campaigns_from_file', [])
if queued:
    st.markdown("### Campaigns queued")
    st.dataframe(pd.DataFrame(queued))

# -----------------------
# 6. Run analysis & generate reports
# -----------------------
if st.button("Generate reports for queued campaigns"):
    campaigns_to_run = st.session_state.get('campaigns_from_file', []) + st.session_state.get('campaigns', [])
    if not campaigns_to_run:
        st.error("No campaigns to run. Add manually or upload CSV.")
    else:
        summary_rows = []
        for camp in campaigns_to_run:
            st.header(f"Campaign: {camp['Campaign Name']}")
            # normalize dates (pd.Timestamp)
            try:
                meas_start = pd.to_datetime(camp['Measurement Start'])
                meas_end = pd.to_datetime(camp['Measurement End'])
                camp_start = pd.to_datetime(camp['Campaign Start'])
                camp_end = pd.to_datetime(camp['Campaign End'])
                pre_start = camp.get('PrePeriod Start')
                pre_end = camp.get('PrePeriod End')
                spend = float(camp.get('Spend', 0.0))
                targets = camp.get('Targets') or []
                controls_user = camp.get('Controls') or []
            except Exception as e:
                st.error(f"Invalid campaign dates/format: {e}")
                continue

            # default pre-period if not provided
            if pre_start is None or pd.isna(pre_start):
                pre_start = meas_start - pd.DateOffset(months=6)
                pre_end = meas_start - pd.Timedelta(days=1)

            # Basic date validation
            if not (meas_start <= camp_start <= camp_end <= meas_end):
                st.warning("Campaign period is not fully inside measurement period. Results will still compute but please verify input dates.")

            # auto-select controls if not provided
            chosen_controls = []
            corr_table = {}
            if controls_user:
                # use only controls that exist in pivot and are not targets
                chosen_controls = [c for c in controls_user if c in present_brands and c not in targets]
                if not chosen_controls:
                    chosen_controls = [b for b in present_brands if b not in targets]
            else:
                # pick top control per target, then deduplicate
                picked = set()
                for t in targets:
                    top, corrs = pick_top_control(pivot, t, pre_start, pre_end, exclude=targets)
                    if top:
                        picked.add(top)
                        corr_table[t] = corrs
                chosen_controls = sorted(list(picked)) if picked else [b for b in present_brands if b not in targets]
            # ensure chosen_controls non-empty
            if not chosen_controls:
                st.error("No valid control brands available in the data for this campaign. Skipping.")
                continue

            # Run DiD pooled regression + counterfactual
            res = run_did_and_build_counterfactual(df_clean[['Date','Brand Name','sales_imputed']], targets, chosen_controls, pre_start, pre_end, meas_start, meas_end, use_log=True)
            if 'error' in res:
                st.error(f"Analysis failed: {res['error']}")
                continue

            # show regression info
            st.subheader("DiD regression (log outcome) result")
            st.write(f"- DiD coefficient (beta on interaction D): {res['beta']:.6f}")
            st.write(f"- Approx percent lift (exp(beta)-1): {res['approx_pct']:.3%}" if res['approx_pct'] is not None else "-")
            st.write(f"- SE (clustered by brand): {res['beta_se']:.6f}, p-value: {res['beta_pval']:.4f}")

            # show correlation table for pre-period (per target)
            if corr_table:
                st.subheader("Pre-period correlations — suggested controls")
                rows = []
                for t, corrs in corr_table.items():
                    best = max(corrs, key=lambda k: abs(corrs[k])) if corrs else None
                    rows.append({'Target': t, 'Suggested control': best, 'Correlation': corrs.get(best) if best else None})
                st.dataframe(pd.DataFrame(rows))

            # Combined plot and numbers
            st.subheader("Combined (all targets)")
            fig_comb = plot_combined(res['actual_combined'], res['expected_combined'], res['control_sum_series'],
                                     pre_start, meas_end, camp_start, camp_end)
            st.plotly_chart(fig_comb, use_container_width=True)
            st.markdown(f"- Combined actual (measurement total): ₹{res['total_actual']:.2f}")
            st.markdown(f"- Combined expected (measurement total): ₹{res['total_expected']:.2f}")
            st.markdown(f"- Combined incremental sales: ₹{res['total_incremental']:.2f}")
            st.markdown(f"- Combined lift %: {res['total_lift_pct']:.2f}%" if res['total_lift_pct'] is not None else "-")
            st.markdown(f"- Combined ROI (increment / spend): { (res['total_incremental']/spend) if (spend and spend!=0) else 'n/a' }")

            # Per-target detail + per-target figure + per-target HTML download
            st.subheader("Per-target detail")
            per_target_figs = {}
            for t, out in res['per_target'].items():
                st.markdown(f"**{t}**")
                if 'incremental' not in out:
                    st.warning(f"No data for target {t}")
                    continue
                fig_t = plot_target(out, pre_start, meas_end, camp_start, camp_end, t)
                per_target_figs[t] = fig_t
                if fig_t:
                    st.plotly_chart(fig_t, use_container_width=True)
                st.markdown(f"- Actual (measurement total): ₹{out['actual_total']:.2f}")
                st.markdown(f"- Expected (measurement total): ₹{out['expected_total']:.2f}")
                st.markdown(f"- Incremental sales: ₹{out['incremental']:.2f}")
                st.markdown(f"- Lift %: {out['lift_pct']:.2f}%" if out['lift_pct'] is not None else "-")
                st.markdown(f"- Correlation (pre) with control-sum: {out['corr_pre'] if out['corr_pre'] is not None else 'n/a'}")

                # per-target HTML fragment
                frag = pio.to_html(fig_t, include_plotlyjs='cdn', full_html=False) if fig_t else ""
                html_piece = f"<h1>{camp['Campaign Name']} — {t}</h1><p>Incremental: {out['incremental']:.2f}</p>" + frag
                st.download_button(f"Download HTML for {camp['Campaign Name']} - {t}", data=html_piece, file_name=f"{camp['Campaign Name']}_{t}_report.html", mime="text/html")

            # campaign HTML (combined + per-target)
            campaign_meta = camp.copy()
            campaign_meta['ControlsUsed'] = chosen_controls
            campaign_meta['Measurement End'] = meas_end
            campaign_meta['Measurement Start'] = meas_start
            campaign_meta['PrePeriod Start'] = pre_start
            campaign_meta['PrePeriod End'] = pre_end
            campaign_meta['Campaign Start'] = camp_start
            campaign_meta['Campaign End'] = camp_end

            campaign_html = make_campaign_html(campaign_meta, res, per_target_figs)
            st.download_button(f"Download full campaign HTML: {camp['Campaign Name']}", data=campaign_html, file_name=f"{camp['Campaign Name'].replace(' ','_')}_report.html", mime="text/html")

            # add summary row
            avg_corr = np.mean([v['corr_pre'] for v in res['per_target'].values() if v.get('corr_pre') is not None]) if any([v.get('corr_pre') is not None for v in res['per_target'].values()]) else None
            summary_rows.append({
                "Campaign Name": camp['Campaign Name'],
                "Campaign Period": f"{camp_start.date()} to {camp_end.date()}",
                "Measurement Period": f"{meas_start.date()} to {meas_end.date()}",
                "Targets": ";".join(targets),
                "Controls used": ";".join(chosen_controls),
                "Spend": spend,
                "Incremental Sales": res['total_incremental'],
                "Lift %": res['total_lift_pct'],
                "ROI (increment/spend)": (res['total_incremental']/spend) if (spend and spend!=0) else None,
                "Avg pre-corr (target vs control_sum)": avg_corr,
                "DiD_beta_log": res['beta'],
                "DiD_approx_pct": res['approx_pct']
            })

        # end loop campaigns
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            st.subheader("All campaigns summary")
            st.dataframe(summary_df)
            st.download_button("Download combined campaigns CSV", data=summary_df.to_csv(index=False).encode('utf-8'),
                               file_name="campaigns_summary.csv", mime="text/csv")
