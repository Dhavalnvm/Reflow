"""
app_pages/setup_tab.py
Tab 1: file uploads + experimental profile parameters
"""

import streamlit as st
import tempfile, os
import numpy as np
import pandas as pd


def render_setup():
    col_left, col_right = st.columns([1, 1], gap="large")

    # ─── LEFT: File uploads ───────────────────────────────────────────────────
    with col_left:
        st.markdown("### 📁 File Uploads")

        # 1. Model
        st.markdown("**① Trained Model (.pth)**")
        model_file = st.file_uploader(
            "thermal_digital_twin (1).pth",
            type=["pth"],
            help="From: saved_models_stage1\\thermal_digital_twin (1).pth",
            key="upload_model",
        )
        if model_file:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
            tmp.write(model_file.read()); tmp.flush()
            st.session_state.model_path = tmp.name
            size_mb = os.path.getsize(tmp.name) / 1e6
            st.success(f"✅ Model loaded — {size_mb:.1f} MB")
        else:
            st.session_state.model_path = None
            st.warning("⚠️ No model → demo mode (physics-based prediction for optimized profile)")

        st.divider()

        # 2. FEA Maps
        st.markdown("**② Ground-Truth FEA Thermal Maps (CSV, 50×50)**")
        c1, c2 = st.columns(2)
        with c1:
            pcb_file = st.file_uploader("PCB Tmap CSV", type=["csv"], key="upload_pcb")
        with c2:
            die_file = st.file_uploader("DIE Tmap CSV", type=["csv"], key="upload_die")

        if pcb_file and die_file:
            pcb_arr = pd.read_csv(pcb_file, header=None).values.astype(np.float64)
            die_arr = pd.read_csv(die_file, header=None).values.astype(np.float64)
            if pcb_arr.shape == (50, 50) and die_arr.shape == (50, 50):
                st.session_state.pcb_exp = pcb_arr
                st.session_state.die_exp = die_arr
                st.success(f"✅ PCB {pcb_arr.shape}  {pcb_arr.min():.2f}→{pcb_arr.max():.2f}°C")
                st.success(f"✅ DIE {die_arr.shape}  {die_arr.min():.4f}→{die_arr.max():.4f}°C")
            else:
                st.error(f"CSVs must be 50×50. Got PCB={pcb_arr.shape}, DIE={die_arr.shape}")
        elif pcb_file or die_file:
            st.warning("Upload both PCB and DIE CSV files.")
        else:
            if "pcb_exp" in st.session_state: del st.session_state["pcb_exp"]
            if "die_exp" in st.session_state: del st.session_state["die_exp"]
            st.info("ℹ️ No FEA maps → model prediction used for experimental profile too")

        st.divider()

        # 3. Training data (optional, for normalization)
        st.markdown("**③ TRAINING_DATA.xlsx** *(optional — for normalization)*")
        train_file = st.file_uploader("TRAINING_DATA.xlsx", type=["xlsx"], key="upload_train")
        if train_file:
            st.session_state.train_xls_bytes = train_file.read()
            st.success("✅ Training data loaded")
        else:
            st.session_state.train_xls_bytes = None
            st.caption("Using built-in C1–C10 defaults for normalization.")

    # ─── RIGHT: Profile + settings ────────────────────────────────────────────
    with col_right:
        st.markdown("### 🌡️ Experimental Reflow Profile (C1 defaults pre-filled)")

        with st.expander("Reflow Process Parameters", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                peak   = st.number_input("Peak Temp (°C)",        220.0, 280.0, 245.0, 0.5)
                tal    = st.number_input("TAL (s)",                10.0, 120.0,  47.0, 1.0)
                soak_t = st.number_input("Soak Temp (°C)",        130.0, 200.0, 165.0, 1.0)
                soak_s = st.number_input("Soak Time (s)",          30.0, 200.0,  90.0, 5.0)
                ramp   = st.number_input("Ramp Rate (°C/s)",        0.5,   5.0,   1.5, 0.1)
                cool   = st.number_input("Cooling Rate (°C/s)",     0.5,   8.0,   3.0, 0.1)
            with c2:
                t_tot  = st.number_input("Total Time (s)",        100.0, 600.0, 291.0, 5.0)
                t_amb  = st.number_input("Ambient Temp (°C)",      15.0,  40.0,  25.0, 1.0)
                cu_fr  = st.number_input("Copper Area Fraction",    0.05,  0.60,  0.20, 0.01)
                pa_fr  = st.number_input("Paste Coverage",         0.005, 0.05,  0.01, 0.001, format="%.3f")
                k_die  = st.number_input("k_die (W/m·K)",          50.0, 200.0, 130.0, 1.0)
                k_pcb  = st.number_input("k_pcb (W/m·K)",           0.1,   1.0,   0.30, 0.01)

        with st.expander("PCB / DIE Geometry", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                l_pcb = st.number_input("PCB side length (mm)", value=40.0, step=1.0)
            with c2:
                l_die = st.number_input("DIE side length (mm)", value=15.0, step=0.5)

        with st.expander("⚡ Optimization Settings (NSGA-II)", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                nsga_pop  = st.slider("Population size",   20, 100, 60, 10)
                nsga_gens = st.slider("Generations",       20, 150, 80, 10)
            with c2:
                peak_ub = st.number_input(
                    "Optimizer peak temp upper bound (°C)\n(must be < experimental)",
                    min_value=220.0,
                    max_value=float(peak - 0.1),
                    value=float(peak - 0.1),
                    step=0.5,
                )
            st.caption("Hard constraint enforced: optimized peak_temp < experimental peak_temp")

        # Store everything in session state
        st.session_state.exp_profile_kwargs = dict(
            peak_temp_C=peak, tal_s=tal, soak_temp_C=soak_t,
            soak_time_s=soak_s, ramp_rate_Cps=ramp, cooling_rate_Cps=cool,
            t_total_s=t_tot, T_amb_C=t_amb,
            copper_area_fraction=cu_fr, paste_coverage_fraction=pa_fr,
            k_die_WmK=k_die, k_pcb_WmK=k_pcb,
        )
        st.session_state.geometry = dict(l_pcb=l_pcb, l_die=l_die)
        st.session_state.opt_settings = dict(
            nsga_pop=nsga_pop, nsga_gens=nsga_gens, peak_ub=peak_ub
        )

        # Live SAC305 compliance preview
        st.divider()
        st.markdown("**SAC305 Compliance Preview**")
        checks = [
            ("Peak temp (235–260°C)", peak, 235, 260),
            ("TAL (30–60s)",          tal,   30,  60),
            ("Soak temp (150–180°C)", soak_t,150,180),
            ("Ramp rate (≤3°C/s)",    ramp,  0,    3),
            ("Cooling (1.5–4°C/s)",   cool,  1.5,  4),
        ]
        cc = st.columns(len(checks))
        for col, (lbl, val, lo, hi) in zip(cc, checks):
            ok  = lo <= val <= hi
            ico = "✅" if ok else "⚠️"
            col.markdown(f"<div style='text-align:center;font-size:1.4rem;'>{ico}</div>"
                         f"<div style='text-align:center;font-size:0.7rem;color:#555'>{lbl}<br><b>{val}</b></div>",
                         unsafe_allow_html=True)
