# app_parallel_baseline.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from datetime import timedelta

st.set_page_config(page_title="Campaign Incrementality — Parallel-scale (Baseline)", layout="wide")

# -------------------------
# Utilities
# -------------------------
def parse_list_field(x):
    if pd.isna(x) or x == "":
        return []
    if isinstance(x, list):
        return x
    s = str(x)
    parts = [p.strip() for p in s.replace("|", ";").replace(",", ";").split(";") if p.strip()]
    return parts

# -------------------------
# Cleaning & imputation (keeps your logic)
# -------------------------
def clean_and_impute(df,
                     date_col='Date',
                     month_col='Month',
                     value_col='Value',
                     carpet_col='Carpet Area',
                     store_open_col='Store Open Date',
                     brand_col='Brand Name'):
    df = df.copy()

    # rename if needed
    if value_col in df.columns and 'sales' not in df.columns:
        df = df.rename(columns={value_col: 'sales'})

    # required columns
    if date_col not in df.columns or brand_col not in df.columns:
        raise ValueError("CSV must include 'Date' and 'Brand Name' columns.")

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

    # drop brands with >30% missing sales (only when sales exists)
    if 'sales' in df.columns:
        brand_missing = df.groupby(brand_col)['sales'].apply(lambda x: x.isna().mean())
        keep_brands = brand_missing[brand_missing <= 0.3].index.tolist()
        dropped_brands = brand_missing[brand_missing > 0.3]
        df = df[df[brand_col].isin(keep_brands)].copy()
    else:
        dropped_brands = pd.Series([])

    # sales = 0 before store open
    if store_open_col in df.columns and 'sales' in df.columns:
        mask = df[store_open_col].notna() & df[date_col].notna() & (df[date_col] < df[store_open_col])
        df.loc[mask, 'sales'] = 0.0

    # imputation
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
            g['imputed_flag'] = s.isna() & g['sales_imputed'].notna() if 'sales' in g else False
            return g.reset_index()
        df = df.groupby(brand_col, group_keys=False).apply(impute_brand).reset_index(drop=True)

    diag = {
        'rows_after_cleaning': int(len(df)),
        'brands_dropped': int(len(dropped_brands)),
        'remaining_brands': int(df[brand_col].nunique()),
        'missing_after_imputation': int(df['sales_imputed'].isna().sum()),
        'total_imputed': int(df['imputed_flag'].sum()) if 'imputed_flag' in df.columns else 0
    }
    return df, diag, dropped_brands

# -------------------------
# Pivot & control selection
# -------------------------
def build_pivot(df, date_col='Date', brand_col='Brand Name', value_col='sales_imputed'):
    df2 = df.copy()
    df2[date_col] = pd.to_datetime(df2[date_col])
    pivot = df2.pivot_table(index=date_col, columns=brand_col, values=value_col, aggfunc='sum').sort_index().fillna(0.0)
    return pivot

def pick_top_control(pivot, target, pre_start, pre_end, exclude=None):
    exclude = set(exclude or [])
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
# Expected via scaling
# -------------------------
def expected_by_scale(pivot, target, controls, pre_start, pre_end, meas_start, meas_end, level_method='mean', eps=1e-9):
    pre_start = pd.to_datetime(pre_start)
    pre_end = pd.to_datetime(pre_end)
    meas_start = pd.to_datetime(meas_start)
    meas_end = pd.to_datetime(meas_end)

    if target not in pivot.columns:
        return {"error": f"target '{target}' not in data"}
    missing_controls = [c for c in controls if c not in pivot.columns]
    if missing_controls:
        return {"error": f"control(s) missing: {missing_controls}"}

    full_start = pre_start
    full_end = meas_end
    T_full = pivot[target].loc[full_start:full_end].astype(float)
    C_full = pivot[controls].sum(axis=1).astype(float).loc[full_start:full_end]

    T_pre = T_full.loc[pre_start:pre_end]
    C_pre = C_full.loc[pre_start:pre_end]
    T_meas = T_full.loc[meas_start:meas_end]
    C_meas = C_full.loc[meas_start:meas_end]

    if T_pre.empty or C_pre.empty or T_meas.empty or C_meas.empty:
        return {"error": "Data unavailable for pre or measurement periods for the target/control combination."}

    if level_method == 'sum':
        denom = C_pre.sum() if C_pre.sum() != 0 else eps
        scale = T_pre.sum() / denom
    else:
        denom = C_pre.mean() if C_pre.mean() != 0 else eps
        scale = T_pre.mean() / denom

    expected_full = C_full * scale

    expected_meas = expected_full.loc[meas_start:meas_end]
    actual_meas = T_meas

    incr_sales = actual_meas.sum() - expected_meas.sum()
    lift_pct = (incr_sales / expected_meas.sum() * 100) if expected_meas.sum() != 0 else None

    corr_pre = float(T_pre.corr(C_pre)) if (not T_pre.empty and not C_pre.empty) else None

    return {
        "target_full": T_full,
        "control_full": C_full,
        "expected_full": expected_full,
        "expected_meas": expected_meas,
        "actual_meas": actual_meas,
        "incremental_sales": float(incr_sales),
        "lift_pct": float(lift_pct) if lift_pct is not None else None,
        "scale": float(scale),
        "corr_pre": corr_pre,
        "controls_used": controls
    }

# -------------------------
# Plot helpers
# -------------------------
def plot_target_full(target_name, target_full, expected_full, control_full, baseline_start, meas_end, campaign_start, campaign_end):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=target_full.index, y=target_full.values, mode='lines', name=f'{target_name} Actual'))
    fig.add_trace(go.Scatter(x=expected_full.index, y=expected_full.values, mode='lines', name=f'{target_name} Expected', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=control_full.index, y=control_full.values, mode='lines', name='Control sum', line=dict(dash='dot')))
    fig.add_vrect(x0=campaign_start, x1=campaign_end, fillcolor="LightSalmon", opacity=0.2, line_width=0)
    fig.update_layout(title=f"{target_name} — Baseline + Measurement", xaxis_title='Date', yaxis_title='Sales')
    fig.update_xaxes(range=[pd.to_datetime(baseline_start), pd.to_datetime(meas_end)])
    return fig

def plot_combined_full(actual_comb, expected_comb, control_sum, baseline_start, meas_end, campaign_start, campaign_end):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_comb.index, y=actual_comb.values, mode='lines', name='Total Actual'))
    fig.add_trace(go.Scatter(x=expected_comb.index, y=expected_comb.values, mode='lines', name='Total Expected', line=dict(dash='dash')))
    if control_sum is not None:
        cs = control_sum.reindex(actual_comb.index.union(control_sum.index)).fillna(0.0)
        fig.add_trace(go.Scatter(x=cs.index, y=cs.values, mode='lines', name='Control sum', line=dict(dash='dot')))
    fig.add_vrect(x0=campaign_start, x1=campaign_end, fillcolor="LightGreen", opacity=0.2, line_width=0)
    fig.update_layout(title="Combined targets — Baseline + Measurement", xaxis_title='Date', yaxis_title='Sales')
    fig.update_xaxes(range=[pd.to_datetime(baseline_start), pd.to_datetime(meas_end)])
    return fig

# -------------------------
# HTML Report builder
# -------------------------
def make_campaign_html(campaign_meta, combined_fig, per_target_figs, per_target_metrics):
    parts = []
    parts.append(f"<h1>Campaign: {campaign_meta['Campaign Name']}</h1>")
    parts.append("<h3>Campaign meta</h3><ul>")
    parts.append(f"<li>Campaign period: {campaign_meta['Campaign Start'].date()} to {campaign_meta['Campaign End'].date()}</li>")
    parts.append(f"<li>Measurement period: {campaign_meta['Measurement Start'].date()} to {campaign_meta['Measurement End'].date()}</li>")
    parts.append(f"<li>Baseline period: {campaign_meta['PrePeriod Start'].date()} to {campaign_meta['PrePeriod End'].date()}</li>")
    parts.append(f"<li>Spend: {campaign_meta.get('Spend', 0)}</li>")
    parts.append(f"<li>Targets: {', '.join(campaign_meta['Targets'])}</li>")
    parts.append(f"<li>Controls used: {', '.join(campaign_meta.get('ControlsUsed', []))}</li>")
    parts.append("</ul>")

    parts.append("<h2>Combined totals</h2>")
    parts.append(f"<p>Actual (measurement total): {campaign_meta.get('CombinedActual', 0):.2f}<br>")
    parts.append(f"Expected (measurement total): {campaign_meta.get('CombinedExpected', 0):.2f}<br>")
    parts.append(f"Incremental: {campaign_meta.get('CombinedIncremental', 0):.2f}<br>")
    parts.append(f"Lift %: {campaign_meta.get('CombinedLiftPct', 'n/a')}</p>")
    parts.append(pio.to_html(combined_fig, include_plotlyjs='cdn', full_html=False))

    parts.append("<h2>Per-target details</h2>")
    for t in campaign_meta['Targets']:
        parts.append(f"<h3>{t}</h3>")
        met = per_target_metrics.get(t, {})
        if met:
            parts.append("<ul>")
            parts.append(f"<li>Actual (measurement total): {met.get('actual_total', 0):.2f}</li>")
            parts.append(f"<li>Expected (measurement total): {met.get('expected_total', 0):.2f}</li>")
            parts.append(f"<li>Incremental: {met.get('incremental', 0):.2f}</li>")
            parts.append(f"<li>Lift %: {met.get('lift_pct', 'n/a')}</li>")
            parts.append(f"<li>Pre-corr (target vs control): {met.get('corr_pre', 'n/a')}</li>")
            parts.append("</ul>")
        fig = per_target_figs.get(t)
        if fig:
            parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))

    html = "<html><head><meta charset='utf-8'></head><body>" + "\n".join(parts) + "</body></html>"
    return html

# -------------------------
# Streamlit UI
# -------------------------
st.title("Campaign Incrementality — Parallel-scale (Baseline)")

st.markdown("Upload sales CSV (must contain `Date`, `Brand Name`, and either `Value`/`sales` or `sales_imputed`). App will clean & impute if needed.")

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

st.success("Data cleaned & imputed")
st.write(f"- Rows after cleaning: {diag['rows_after_cleaning']}")
st.write(f"- Remaining brands: {diag['remaining_brands']}")
st.write(f"- Total imputed values: {diag['total_imputed']}")

# Build pivot
pivot = build_pivot(df_clean)
present_brands = sorted(pivot.columns.tolist())

st.markdown("---")
st.header("2) Campaign input")

colA, colB = st.columns([2,1])
with colA:
    st.markdown("**Add campaign manually**")
    with st.form("camp_form"):
        cname = st.text_input("Campaign name")
        meas_start = st.date_input("Measurement start (inclusive)", key="ms")
        meas_end = st.date_input("Measurement end (inclusive)", key="me")
        camp_start = st.date_input("Campaign start (should be inside measurement)", key="cs")
        camp_end = st.date_input("Campaign end", key="ce")
        spend = st.number_input("Spend (optional)", min_value=0.0, value=0.0, step=100.0)
        targets_sel = st.multiselect("Targets (one or more)", options=present_brands)
        controls_sel = st.multiselect("Controls (optional; leave blank to auto-select)", options=present_brands)

        # Baseline Period widget (renamed)
        baseline_option = st.radio("Baseline period",
                                  ["Automatic (6 months before measurement start)", "Manual"],
                                  index=0)
        if baseline_option == "Manual":
            baseline_start_manual = st.date_input("Baseline start date (must be before the measurement start)", help="This date must be before the measurement start date.")
            pre_start_val = pd.to_datetime(baseline_start_manual)
            # baseline end for computation is measurement start - 1 day (we'll set after form submission)
            pre_end_val = None
        else:
            pre_start_val = None
            pre_end_val = None

        submitted = st.form_submit_button("Add campaign")
        if submitted:
            if not targets_sel:
                st.error("Pick at least one target.")
            else:
                # determine pre_start & pre_end
                meas_start_ts = pd.to_datetime(meas_start)
                if baseline_option == "Manual":
                    baseline_start_ts = pd.to_datetime(baseline_start_manual)
                    if baseline_start_ts >= meas_start_ts:
                        st.error("Baseline start must be before measurement start.")
                        st.stop()
                    pre_start = baseline_start_ts
                    pre_end = meas_start_ts - pd.Timedelta(days=1)
                else:
                    pre_start = meas_start_ts - pd.DateOffset(months=6)
                    pre_end = meas_start_ts - pd.Timedelta(days=1)

                campaign_obj = {
                    "Campaign Name": cname or "Campaign",
                    "Campaign Start": pd.to_datetime(camp_start),
                    "Campaign End": pd.to_datetime(camp_end),
                    "Measurement Start": meas_start_ts,
                    "Measurement End": pd.to_datetime(meas_end),
                    "PrePeriod Start": pre_start,
                    "PrePeriod End": pre_end,
                    "Spend": float(spend),
                    "Targets": targets_sel,
                    "Controls": controls_sel or []
                }
                if "campaigns" not in st.session_state:
                    st.session_state["campaigns"] = []
                st.session_state["campaigns"].append(campaign_obj)
                st.success("Campaign added to queue")

with colB:
    st.markdown("**Or upload campaigns CSV**")
    template = pd.DataFrame([{
        "Campaign Name":"Campaign",
        "Campaign Start":"2023-10-15",
        "Campaign End":"2023-10-21",
        "Measurement Start":"2023-09-15",
        "Measurement End":"2023-10-31",
        "PrePeriod Start":"",  # optional; leave blank to auto
        "PrePeriod End":"",
        "Spend":"0",
        "Targets":"BVB;Frankie",
        "Controls":"Burger King;KFC"
    }])
    st.download_button("Download campaign template (CSV)", data=template.to_csv(index=False).encode('utf-8'), file_name="campaign_template.csv")
    camp_file = st.file_uploader("Upload campaigns CSV (optional)", type=["csv"], key="camp_file")
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
            st.session_state["campaigns_from_file"] = parsed
            st.success(f"Loaded {len(parsed)} campaigns from CSV")
        except Exception as e:
            st.error(f"Could not parse campaign CSV: {e}")

# show queued campaigns
queued = st.session_state.get("campaigns", []) + st.session_state.get("campaigns_from_file", [])
if queued:
    st.markdown("### Campaigns queued")
    st.dataframe(pd.DataFrame(queued))

# -------------------------
# Run analysis & reports
# -------------------------
if st.button("Generate reports for queued campaigns"):
    campaigns_to_run = st.session_state.get("campaigns_from_file", []) + st.session_state.get("campaigns", [])
    if not campaigns_to_run:
        st.error("No campaigns to run. Add manually or upload CSV.")
    else:
        summary_rows = []

        for camp in campaigns_to_run:
            st.header(f"Campaign: {camp['Campaign Name']}")
            # normalize dates
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

            # default pre if not provided (6 months back)
            if pre_start is None or pd.isna(pre_start):
                pre_start = pd.to_datetime(meas_start) - pd.DateOffset(months=6)
                pre_end = pd.to_datetime(meas_start) - pd.Timedelta(days=1)

            # validate baseline start < measurement start
            if pd.to_datetime(pre_start) >= pd.to_datetime(meas_start):
                st.error("Baseline start must be before measurement start. Skipping campaign.")
                continue

            # basic date validation for campaign inside measurement window (warn but allow)
            if not (meas_start <= camp_start <= camp_end <= meas_end):
                st.warning("Campaign period is not fully inside measurement period. Proceeding but please verify input dates.")

            # select controls
            if controls_user:
                chosen_controls = [c for c in controls_user if c in present_brands and c not in targets]
                if not chosen_controls:
                    chosen_controls = [b for b in present_brands if b not in targets]
            else:
                picked = set()
                corr_table = {}
                for t in targets:
                    top, corrs = pick_top_control(pivot, t, pre_start, pre_end, exclude=targets)
                    if top:
                        picked.add(top)
                        corr_table[t] = corrs
                chosen_controls = sorted(list(picked)) if picked else [b for b in present_brands if b not in targets]

            if not chosen_controls:
                st.error("No valid control brands available. Skipping campaign.")
                continue

            # per-target processing
            per_target_metrics = {}
            per_target_figs = {}

            combined_actual = None
            combined_expected = None
            control_sum_for_combo = None

            for t in targets:
                out = expected_by_scale(pivot, t, chosen_controls, pre_start, pre_end, meas_start, meas_end, level_method='mean')
                if 'error' in out:
                    st.warning(f"Target {t}: {out['error']}")
                    per_target_metrics[t] = {}
                    per_target_figs[t] = None
                    continue

                per_target_metrics[t] = {
                    'actual_total': float(out['actual_meas'].sum()),
                    'expected_total': float(out['expected_meas'].sum()),
                    'incremental': float(out['incremental_sales']),
                    'lift_pct': float(out['lift_pct']) if out['lift_pct'] is not None else None,
                    'corr_pre': out['corr_pre']
                }

                # full series pre..meas_end
                target_full = out['target_full'].loc[pre_start:meas_end]
                expected_full = out['expected_full'].loc[pre_start:meas_end]
                control_full = out['control_full'].loc[pre_start:meas_end]

                fig_t = plot_target_full(t, target_full, expected_full, control_full, pre_start, meas_end, camp_start, camp_end)
                per_target_figs[t] = fig_t

                # accumulate
                if combined_actual is None:
                    combined_actual = target_full.reindex(expected_full.index.union(target_full.index)).fillna(0.0)
                else:
                    combined_actual = combined_actual.add(target_full.reindex(combined_actual.index.union(target_full.index)).fillna(0.0), fill_value=0.0)
                if combined_expected is None:
                    combined_expected = expected_full.reindex(target_full.index.union(expected_full.index)).fillna(0.0)
                else:
                    combined_expected = combined_expected.add(expected_full.reindex(combined_expected.index.union(expected_full.index)).fillna(0.0), fill_value=0.0)
                control_sum_for_combo = control_full if control_sum_for_combo is None else control_sum_for_combo.add(control_full.reindex(control_sum_for_combo.index.union(control_full.index)).fillna(0.0), fill_value=0.0)

            if combined_actual is None or combined_expected is None:
                st.error("No valid target data for this campaign. Skipping.")
                continue

            full_index = combined_actual.index.union(combined_expected.index).sort_values()
            combined_actual = combined_actual.reindex(full_index).fillna(0.0).loc[pre_start:meas_end]
            combined_expected = combined_expected.reindex(full_index).fillna(0.0).loc[pre_start:meas_end]
            control_sum_for_combo = control_sum_for_combo.reindex(full_index).fillna(0.0).loc[pre_start:meas_end]

            combined_actual_meas = combined_actual.loc[meas_start:meas_end]
            combined_expected_meas = combined_expected.loc[meas_start:meas_end]
            total_incr = float(combined_actual_meas.sum() - combined_expected_meas.sum())
            total_expected_meas = float(combined_expected_meas.sum())
            total_lift_pct = (total_incr / total_expected_meas * 100) if total_expected_meas != 0 else None

            # combined plot
            fig_comb = plot_combined_full(combined_actual, combined_expected, control_sum_for_combo, pre_start, meas_end, camp_start, camp_end)
            st.plotly_chart(fig_comb, use_container_width=True)

            st.markdown(f"- Combined actual (measurement total): ₹{combined_actual_meas.sum():.2f}")
            st.markdown(f"- Combined expected (measurement total): ₹{combined_expected_meas.sum():.2f}")
            st.markdown(f"- Combined incremental sales: ₹{total_incr:.2f}")
            st.markdown(f"- Combined lift %: {total_lift_pct:.2f}%" if total_lift_pct is not None else "-")
            st.markdown(f"- Combined ROI (increment / spend): {(total_incr/spend) if (spend and spend!=0) else 'n/a'}")

            st.subheader("Per-target detail")
            for t in targets:
                st.markdown(f"**{t}**")
                met = per_target_metrics.get(t, {})
                if not met:
                    st.warning(f"No data for {t}")
                    continue
                st.markdown(f"- Actual (measurement total): ₹{met['actual_total']:.2f}")
                st.markdown(f("- Expected (measurement total): ₹{met['expected_total']:.2f}"))
                st.markdown(f"- Incremental sales: ₹{met['incremental']:.2f}")
                st.markdown(f"- Lift %: {met['lift_pct']:.2f}%" if met['lift_pct'] is not None else "-")
                st.markdown(f"- Correlation (pre) with control-sum: {met['corr_pre'] if met['corr_pre'] is not None else 'n/a'}")
                if per_target_figs.get(t) is not None:
                    st.plotly_chart(per_target_figs[t], use_container_width=True)

            # single HTML per campaign
            campaign_meta = {
                'Campaign Name': camp['Campaign Name'],
                'Campaign Start': camp_start,
                'Campaign End': camp_end,
                'Measurement Start': meas_start,
                'Measurement End': meas_end,
                'PrePeriod Start': pre_start,
                'PrePeriod End': pre_end,
                'Targets': targets,
                'ControlsUsed': chosen_controls,
                'CombinedActual': float(combined_actual_meas.sum()),
                'CombinedExpected': float(combined_expected_meas.sum()),
                'CombinedIncremental': float(total_incr),
                'CombinedLiftPct': float(total_lift_pct) if total_lift_pct is not None else None,
                'Spend': spend
            }
            campaign_html = make_campaign_html(campaign_meta, fig_comb, per_target_figs, per_target_metrics)
            st.download_button(f"Download campaign HTML: {camp['Campaign Name']}", data=campaign_html, file_name=f"{camp['Campaign Name'].replace(' ','_')}_report.html", mime="text/html")

            avg_corrs = [v['corr_pre'] for v in per_target_metrics.values() if v.get('corr_pre') is not None]
            avg_corr = float(np.mean(avg_corrs)) if avg_corrs else None
            summary_rows.append({
                "Campaign Name": camp['Campaign Name'],
                "Campaign Period": f"{camp_start.date()} to {camp_end.date()}",
                "Measurement Period": f"{meas_start.date()} to {meas_end.date()}",
                "Targets": ";".join(targets),
                "Controls used": ";".join(chosen_controls),
                "Spend": spend,
                "Incremental Sales": total_incr,
                "Lift %": total_lift_pct,
                "ROI (increment/spend)": (total_incr/spend) if (spend and spend!=0) else None,
                "Avg pre-corr (target vs control_sum)": avg_corr
            })

        # after campaigns loop
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            st.subheader("All campaigns summary")
            st.dataframe(summary_df)
            st.download_button("Download combined campaigns CSV", data=summary_df.to_csv(index=False).encode('utf-8'), file_name="campaigns_summary.csv", mime="text/csv")
