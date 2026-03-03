"""
app_pages/results_tab.py
Tab 3: all figures, metrics, and downloads
"""

import streamlit as st
import pandas as pd
import json, io, zipfile


def render_results(r: dict):
    exp     = r["exp_profile"]
    opt     = r["opt_profile"]
    m       = r["metrics"]
    figs    = r["figures"]

    # ── Status row ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tag = "🧠 Real model" if m.get("exp_risk") else "⚙️ Demo mode"
        st.info(tag)
    with c2:
        st.info(f"🎯 {m['pareto_solutions']} Pareto solutions")
    with c3:
        st.info(f"Exp risk: **{m['exp_risk']}**")
    with c4:
        st.info(f"Opt risk: **{m['opt_risk']}**")

    st.divider()

    # ── KPI cards ──────────────────────────────────────────────────────────────
    st.markdown("#### Key Metrics")
    kpis = [
        ("Exp Peak Temp",   f"{exp.peak_temp_C:.1f}°C",    None,    ""),
        ("Opt Peak Temp",   f"{opt.peak_temp_C:.1f}°C",
         f"{opt.peak_temp_C - exp.peak_temp_C:+.1f}°C",    "dn"),
        ("PCB Mean % ΔT",  f"{m['pcb_mean_pct_dT']:.3f}%", None,    ""),
        ("DIE Mean % ΔT",  f"{m['die_mean_pct_dT']:.4f}%", None,    ""),
        ("PCB CV Δ",        f"{m['pcb_cv_improvement_pct']:+.1f}%",  None, ""),
        ("Score Δ",
         f"{m['exp_composite']:.4f}→{m['opt_composite']:.4f}",      None, ""),
    ]
    cols = st.columns(len(kpis))
    for col, (lbl, val, delta, cls) in zip(cols, kpis):
        dhtml = ""
        if delta:
            color = "#16a34a" if cls == "dn" else "#dc2626"
            dhtml = f'<div style="font-size:.85rem;color:{color};font-weight:600">{delta}</div>'
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-val">{val}</div>
            <div class="kpi-label">{lbl}</div>{dhtml}</div>""",
            unsafe_allow_html=True)

    st.divider()

    # ── Fig 6 contour plots ────────────────────────────────────────────────────
    st.markdown("#### 🌡️ Fig 6 — Temperature Contour Maps")

    st.markdown("**PCB**")
    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"Experimental — peak={exp.peak_temp_C:.0f}°C  TAL={exp.tal_s:.0f}s")
        st.image(figs["fig6a"], use_container_width=True)
    with c2:
        st.caption(f"Optimized — peak={opt.peak_temp_C:.1f}°C  TAL={opt.tal_s:.1f}s")
        st.image(figs["fig6c"], use_container_width=True)

    st.markdown("**DIE**")
    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"Experimental — peak={exp.peak_temp_C:.0f}°C")
        st.image(figs["fig6b"], use_container_width=True)
    with c2:
        st.caption(f"Optimized — peak={opt.peak_temp_C:.1f}°C")
        st.image(figs["fig6d"], use_container_width=True)

    st.divider()

    # ── Full comparison figure ─────────────────────────────────────────────────
    st.markdown("#### 📈 Full Comparison — % ΔT Analysis")
    st.image(figs["comparison"], use_container_width=True)

    # ── Pareto front ──────────────────────────────────────────────────────────
    if figs.get("pareto"):
        st.divider()
        st.markdown("#### 🎯 NSGA-II Pareto Front")
        st.image(figs["pareto"], use_container_width=True)

    st.divider()

    # ── Profile parameter comparison table ────────────────────────────────────
    st.markdown("#### 📋 Profile Parameter Comparison")
    UNITS = {
        "peak_temp_C":"°C","tal_s":"s","soak_temp_C":"°C","soak_time_s":"s",
        "ramp_rate_Cps":"°C/s","cooling_rate_Cps":"°C/s","t_total_s":"s",
        "T_amb_C":"°C","copper_area_fraction":"—","paste_coverage_fraction":"—",
        "k_die_WmK":"W/m·K","k_pcb_WmK":"W/m·K",
    }
    rows = []
    for k in exp.FEATURE_COLS:
        ev = getattr(exp, k); ov = getattr(opt, k); dv = ov - ev
        rows.append({
            "Parameter": k,
            "Unit": UNITS.get(k, ""),
            "Experimental": round(ev, 4),
            "Optimized":    round(ov, 4),
            "Δ":            f"{dv:+.4f}",
            "":             "⬇" if dv < -0.001 else ("⬆" if dv > 0.001 else "–"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Metrics detail ────────────────────────────────────────────────────────
    st.markdown("#### 🔢 Thermal Field Metrics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**PCB**")
        st.dataframe(pd.DataFrame({
            "Metric": ["Temp Range (ΔT)", "CV (Uniformity)", "CV Improvement",
                       "Mean % ΔT", "Min % ΔT", "Max % ΔT"],
            "Experimental": [f"{m['pcb_range_exp']:.3f}°C", f"{m['pcb_cv_exp']:.5f}", "—","—","—","—"],
            "Optimized":    [f"{m['pcb_range_opt']:.3f}°C", f"{m['pcb_cv_opt']:.5f}",
                             f"{m['pcb_cv_improvement_pct']:+.1f}%",
                             f"{m['pcb_mean_pct_dT']:.3f}%",
                             f"{m['pcb_min_pct_dT']:.3f}%",
                             f"{m['pcb_max_pct_dT']:.3f}%"],
        }), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**DIE**")
        st.dataframe(pd.DataFrame({
            "Metric": ["Temp Range (ΔT)", "CV (Uniformity)", "CV Improvement",
                       "Mean % ΔT", "Min % ΔT", "Max % ΔT"],
            "Experimental": [f"{m['die_range_exp']:.5f}°C", f"{m['die_cv_exp']:.6f}", "—","—","—","—"],
            "Optimized":    [f"{m['die_range_opt']:.5f}°C", f"{m['die_cv_opt']:.6f}",
                             f"{m['die_cv_improvement_pct']:+.1f}%",
                             f"{m['die_mean_pct_dT']:.4f}%",
                             f"{m['die_min_pct_dT']:.4f}%",
                             f"{m['die_max_pct_dT']:.4f}%"],
        }), use_container_width=True, hide_index=True)

    st.divider()

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown("#### ⬇️ Download Outputs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("📥 Exp PCB Fig 6",  figs["fig6a"], "fig6a_exp_PCB.png",  "image/png", use_container_width=True)
        st.download_button("📥 Exp DIE Fig 6",  figs["fig6b"], "fig6b_exp_DIE.png",  "image/png", use_container_width=True)
    with c2:
        st.download_button("📥 Opt PCB Fig 6",  figs["fig6c"], "fig6c_opt_PCB.png",  "image/png", use_container_width=True)
        st.download_button("📥 Opt DIE Fig 6",  figs["fig6d"], "fig6d_opt_DIE.png",  "image/png", use_container_width=True)
    with c3:
        st.download_button("📥 Full Comparison", figs["comparison"], "comparison.png", "image/png", use_container_width=True)
        if figs.get("pareto"):
            st.download_button("📥 Pareto Front", figs["pareto"], "pareto_front.png", "image/png", use_container_width=True)
    with c4:
        results_json = json.dumps({
            "experimental_profile": exp.to_dict(),
            "optimized_profile":    opt.to_dict(),
            "metrics": m,
        }, indent=2)
        st.download_button("📥 Results JSON", results_json, "results.json", "application/json", use_container_width=True)

        # ZIP all
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("fig6a_exp_PCB.png",    figs["fig6a"])
            zf.writestr("fig6b_exp_DIE.png",    figs["fig6b"])
            zf.writestr("fig6c_opt_PCB.png",    figs["fig6c"])
            zf.writestr("fig6d_opt_DIE.png",    figs["fig6d"])
            zf.writestr("comparison.png",        figs["comparison"])
            if figs.get("pareto"):
                zf.writestr("pareto_front.png", figs["pareto"])
            zf.writestr("results.json",          results_json)
        zip_buf.seek(0)
        st.download_button("📦 Download All (ZIP)", zip_buf.read(),
                           "reflow_results.zip", "application/zip", use_container_width=True)
