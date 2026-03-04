"""
pages/results_page.py
Renders the full results UI in Streamlit tab 3.
"""

import streamlit as st
import json, io
import numpy as np


def render_results(r: dict):
    exp = r['exp_profile']
    opt = r['opt_profile']
    m   = r['metrics']
    figs = r['figures']

    # ── Status badges ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        if r['model_loaded']:
            st.success("🧠 Real PyTorch model used")
        else:
            st.warning("⚙️ Physics-based prediction (demo)")
    with c2:
        if r['used_fea_maps']:
            st.success("📊 Ground-truth FEA maps (experimental)")
        else:
            st.info("📊 Model-predicted experimental maps")
    with c3:
        if r.get('pareto_info'):
            st.success(f"🎯 {r['pareto_info']['n_solutions']} Pareto solutions found")
        else:
            st.info("🎯 DE fallback optimizer used")

    st.divider()

    # ── Top KPI cards ──────────────────────────────────────────────────────────
    st.markdown("#### 📊 Key Performance Indicators")
    kpis = [
        ("Exp Peak Temp",   f"{exp['peak_temp_C']:.1f} °C",   None,       ""),
        ("Opt Peak Temp",   f"{opt['peak_temp_C']:.1f} °C",
         f"{opt['peak_temp_C']-exp['peak_temp_C']:+.1f}°C",   "neg"),
        ("PCB Mean % ΔT",   f"{m['pcb_mean_pct_dT']:.3f} %",  None,       ""),
        ("DIE Mean % ΔT",   f"{m['die_mean_pct_dT']:.4f} %",  None,       ""),
        ("PCB CV Improv.",  f"{m['pcb_cv_improvement_pct']:+.1f} %", None, ""),
        ("DIE CV Improv.",  f"{m['die_cv_improvement_pct']:+.1f} %", None, ""),
    ]
    cols = st.columns(6)
    for col, (lbl, val, delta, dclass) in zip(cols, kpis):
        delta_html = ""
        if delta:
            color = "#16a34a" if dclass == "neg" else "#dc2626"
            delta_html = f'<div style="font-size:0.85rem;color:{color};font-weight:600;">{delta}</div>'
        col.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                    padding:0.9rem 1rem;text-align:center;">
            <div style="font-size:1.4rem;font-weight:700;color:#1565c0;">{val}</div>
            <div style="font-size:0.75rem;color:#64748b;margin-top:0.2rem;">{lbl}</div>
            {delta_html}
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Fig 6 side-by-side ────────────────────────────────────────────────────
    st.markdown("#### 🌡️ Fig 6 — Temperature Contour Maps")

    st.markdown("**PCB Temperature Contour**")
    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"Experimental  (peak = {exp['peak_temp_C']:.0f}°C)")
        st.image(figs['fig6a_exp_pcb'], use_container_width=True)
    with c2:
        st.caption(f"Optimized  (peak = {opt['peak_temp_C']:.1f}°C)")
        st.image(figs['fig6c_opt_pcb'], use_container_width=True)

    st.markdown("**DIE Temperature Contour**")
    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"Experimental  (peak = {exp['peak_temp_C']:.0f}°C)")
        st.image(figs['fig6b_exp_die'], use_container_width=True)
    with c2:
        st.caption(f"Optimized  (peak = {opt['peak_temp_C']:.1f}°C)")
        st.image(figs['fig6d_opt_die'], use_container_width=True)

    st.divider()

    # ── Full comparison figure ────────────────────────────────────────────────
    st.markdown("#### 📈 Full Comparison — % ΔT Analysis")
    st.image(figs['comparison'], use_container_width=True)

    # ── Pareto front ──────────────────────────────────────────────────────────
    if figs.get('pareto'):
        st.divider()
        st.markdown("#### 🎯 NSGA-II Pareto Front")
        st.image(figs['pareto'], use_container_width=True)

    st.divider()

    # ── Profile comparison table ───────────────────────────────────────────────
    st.markdown("#### 📋 Profile Parameters — Experimental vs Optimized")
    import pandas as pd
    rows = []
    units = {
        'peak_temp_C':'°C','tal_s':'s','soak_temp_C':'°C','soak_time_s':'s',
        'ramp_rate_Cps':'°C/s','cooling_rate_Cps':'°C/s','t_total_s':'s',
        'T_amb_C':'°C','copper_area_fraction':'—','paste_coverage_fraction':'—',
        'k_die_WmK':'W/m·K','k_pcb_WmK':'W/m·K',
    }
    for k in exp:
        ev = exp[k]; ov = opt[k]; dv = ov - ev
        rows.append({
            "Parameter": k,
            "Unit": units.get(k,''),
            "Experimental": f"{ev:.4f}",
            "Optimized":    f"{ov:.4f}",
            "Δ":            f"{dv:+.4f}",
            "Changed": "⬇" if dv < -0.001 else ("⬆" if dv > 0.001 else "–"),
        })
    df = pd.DataFrame(rows)
    df_styled = df.style.apply(
        lambda row: ['background-color: #e8f5e9' if row['Parameter']=='peak_temp_C' else '']*len(row),
        axis=1
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Metrics detail ────────────────────────────────────────────────────────
    st.markdown("#### 🔢 Thermal Field Metrics Detail")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**PCB**")
        pcb_data = {
            "Metric": ["Temp Range (ΔT)", "CV (Uniformity)", "Mean % ΔT", "Min % ΔT", "Max % ΔT"],
            "Experimental": [
                f"{m['pcb_range_exp']:.3f} °C",
                f"{m['pcb_cv_exp']:.5f}",
                "—", "—", "—",
            ],
            "Optimized": [
                f"{m['pcb_range_opt']:.3f} °C",
                f"{m['pcb_cv_opt']:.5f}",
                f"{m['pcb_mean_pct_dT']:.3f}%",
                f"{m['pcb_min_pct_dT']:.3f}%",
                f"{m['pcb_max_pct_dT']:.3f}%",
            ],
        }
        st.dataframe(pd.DataFrame(pcb_data), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**DIE**")
        die_data = {
            "Metric": ["Temp Range (ΔT)", "CV (Uniformity)", "Mean % ΔT", "Min % ΔT", "Max % ΔT"],
            "Experimental": [
                f"{m['die_range_exp']:.5f} °C",
                f"{m['die_cv_exp']:.6f}",
                "—", "—", "—",
            ],
            "Optimized": [
                f"{m['die_range_opt']:.5f} °C",
                f"{m['die_cv_opt']:.6f}",
                f"{m['die_mean_pct_dT']:.4f}%",
                f"{m['die_min_pct_dT']:.4f}%",
                f"{m['die_max_pct_dT']:.4f}%",
            ],
        }
        st.dataframe(pd.DataFrame(die_data), use_container_width=True, hide_index=True)

    st.divider()

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown("#### ⬇️ Download All Outputs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("📥 Exp PCB Fig 6",  figs['fig6a_exp_pcb'],
                           "fig6a_exp_PCB.png", "image/png", use_container_width=True)
        st.download_button("📥 Exp DIE Fig 6",  figs['fig6b_exp_die'],
                           "fig6b_exp_DIE.png", "image/png", use_container_width=True)
    with c2:
        st.download_button("📥 Opt PCB Fig 6",  figs['fig6c_opt_pcb'],
                           "fig6c_opt_PCB.png", "image/png", use_container_width=True)
        st.download_button("📥 Opt DIE Fig 6",  figs['fig6d_opt_die'],
                           "fig6d_opt_DIE.png", "image/png", use_container_width=True)
    with c3:
        st.download_button("📥 Full Comparison", figs['comparison'],
                           "fig7_full_comparison.png", "image/png", use_container_width=True)
        if figs.get('pareto'):
            st.download_button("📥 Pareto Front", figs['pareto'],
                               "pareto_front.png", "image/png", use_container_width=True)
    with c4:
        # JSON results
        results_json = json.dumps({
            'experimental_profile': exp,
            'optimized_profile':    opt,
            'metrics':              m,
        }, indent=2)
        st.download_button("📥 Results JSON", results_json,
                           "results.json", "application/json", use_container_width=True)

        # Download all as zip
        import zipfile
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("fig6a_exp_PCB.png",       figs['fig6a_exp_pcb'])
            zf.writestr("fig6b_exp_DIE.png",       figs['fig6b_exp_die'])
            zf.writestr("fig6c_opt_PCB.png",       figs['fig6c_opt_pcb'])
            zf.writestr("fig6d_opt_DIE.png",       figs['fig6d_opt_die'])
            zf.writestr("fig7_full_comparison.png",figs['comparison'])
            if figs.get('pareto'):
                zf.writestr("pareto_front.png",    figs['pareto'])
            zf.writestr("results.json",            results_json)
        zip_buf.seek(0)
        st.download_button("📦 Download All (ZIP)", zip_buf.read(),
                           "reflow_results.zip", "application/zip", use_container_width=True)
