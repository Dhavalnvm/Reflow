"""
app.py  —  Streamlit UI
PCB Reflow Thermal Digital Twin

Run:   streamlit run app.py

Imports the EXISTING project modules directly:
    inference_engine.py  →  InferenceEngine, ReflowProfile
    reflow_advisor.py    →  ReflowAdvisor
    optimizer.py         →  ReflowOptimizer, BOUNDS_ARRAY, PARAM_KEYS
    visualizer.py        →  Visualizer, build_tt_curve
    experimental_pipeline.py → make_fig6_contour, make_comparison_figure, load_fea_maps
"""

import sys, os
import streamlit as st

st.set_page_config(
    page_title="PCB Reflow Thermal Digital Twin",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 60%, #1976d2 100%);
        padding: 1.8rem 2.5rem; border-radius: 12px;
        margin-bottom: 1.5rem; color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.9rem; }
    .main-header p  { color: #bbdefb; margin: 0.25rem 0 0 0; font-size: 0.95rem; }

    .kpi-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 1rem 0.8rem; text-align: center;
    }
    .kpi-val   { font-size: 1.5rem; font-weight: 700; color: #1565c0; }
    .kpi-label { font-size: 0.75rem; color: #64748b; margin-top: 0.15rem; }
    .kpi-delta { font-size: 0.85rem; font-weight: 600; margin-top: 0.2rem; }
    .dn { color: #16a34a; } .up { color: #dc2626; }

    .badge-ok   { color: #16a34a; font-weight: 600; }
    .badge-warn { color: #d97706; font-weight: 600; }
    .stButton > button { border-radius: 8px; font-weight: 600; }
    .stDownloadButton > button { width: 100%; border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>🔥 PCB Reflow Thermal Digital Twin</h1>
  <p>Experimental Profile → Fig 6 Thermal Contours → NSGA-II Optimization → % ΔT Comparison
     &nbsp;·&nbsp; SAC305 Lead-Free Solder</p>
</div>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
for key, default in [("ran", False), ("results", None), ("model_path", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# =============================================================================
#  Tabs
# =============================================================================
tab1, tab2, tab3 = st.tabs([
    "⚙️  Step 1 — Inputs",
    "▶️  Step 2 — Run Pipeline",
    "📊  Step 3 — Results",
])

# =============================================================================
#  TAB 1 — Inputs
# =============================================================================
with tab1:
    from app_pages.setup_tab import render_setup
    render_setup()

# =============================================================================
#  TAB 2 — Run
# =============================================================================
with tab2:
    from app_pages.run_tab import render_run
    render_run()

# =============================================================================
#  TAB 3 — Results
# =============================================================================
with tab3:
    from app_pages.results_tab import render_results
    if st.session_state.ran and st.session_state.results:
        render_results(st.session_state.results)
    else:
        st.info("▶️  Run the pipeline in **Step 2** first.")
