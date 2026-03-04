"""
=============================================================================
experimental_pipeline.py  —  Stage 3
PCB Reflow Thermal Digital Twin
=============================================================================
Feeds your experimental reflow profile into the trained digital twin,
optimizes it, and produces Fig 6 style temperature contour comparisons.

DEPENDS ON (existing files — keep in same folder):
    inference_engine.py
    reflow_advisor.py
    optimizer.py
    visualizer.py

Usage (CLI):
    python experimental_pipeline.py
    python experimental_pipeline.py --model "path/to/thermal_digital_twin (1).pth"
    python experimental_pipeline.py --model model.pth --pcb C1_PCB.csv --die C1_DIE.csv
    python experimental_pipeline.py --quick      # 20-gen fast test

All arguments are optional — sensible defaults are used when omitted.
=============================================================================
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Import existing modules (must be in same directory) ──────────────────────
from inference_engine import InferenceEngine, ReflowProfile, PredictionResult
from reflow_advisor   import ReflowAdvisor, RiskLevel
from optimizer        import ReflowOptimizer, PROFILE_BOUNDS, PARAM_KEYS, BOUNDS_ARRAY
from visualizer       import Visualizer, build_tt_curve, THERMAL_CMAP

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("figures_stage3")
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Defaults  (matches C1 from TRAINING_DATA.xlsx)
# =============================================================================

DEFAULT_MODEL_PATH = r"saved_models_stage1\thermal_digital_twin (1).pth"

# C1 experimental reflow profile
EXPERIMENTAL_PROFILE = ReflowProfile(
    peak_temp_C             = 245.0,
    tal_s                   = 47.0,
    soak_temp_C             = 165.0,
    soak_time_s             = 90.0,
    ramp_rate_Cps           = 1.5,
    cooling_rate_Cps        = 3.0,
    t_total_s               = 291.0,
    T_amb_C                 = 25.0,
    copper_area_fraction    = 0.20,
    paste_coverage_fraction = 0.01,
    k_die_WmK               = 130.0,
    k_pcb_WmK               = 0.30,
)

# C1 PCB geometry (mm) — used for physical coordinate axes on Fig 6
PCB_SIDE_MM = 40.0
DIE_SIDE_MM = 15.0


# =============================================================================
# Helpers
# =============================================================================

def banner(title, width=65):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def load_fea_maps(pcb_csv: str, die_csv: str):
    """Load ground-truth FEA thermal maps from CSV files."""
    pcb = pd.read_csv(pcb_csv, header=None).values.astype(np.float64)
    die = pd.read_csv(die_csv, header=None).values.astype(np.float64)
    assert pcb.shape == (50, 50), f"PCB map must be 50×50, got {pcb.shape}"
    assert die.shape == (50, 50), f"DIE map must be 50×50, got {die.shape}"
    return pcb, die


def make_fig6_contour(tmap: np.ndarray, L_mm: float, title: str,
                      save_path: Path, n_contours: int = 22):
    """
    Fig 6 style temperature contour plot with physical mm axes.
    Matches the style of the reference figures in the research paper.
    """
    x_mm = np.linspace(-L_mm / 2, L_mm / 2, tmap.shape[1])
    z_mm = np.linspace(-L_mm / 2, L_mm / 2, tmap.shape[0])
    X, Z = np.meshgrid(x_mm, z_mm)

    lvls = np.linspace(tmap.min(), tmap.max(), n_contours + 1)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cf   = ax.contourf(X, Z, tmap, levels=lvls, cmap="viridis")
    ax.contour(X, Z, tmap, levels=lvls[::4], colors="k", linewidths=0.4, alpha=0.45)

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("Temperature (°C)", fontsize=11)

    ax.set_xlabel("X (mm)", fontsize=11)
    ax.set_ylabel("Z (mm)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_aspect("equal")

    # Stats annotation
    stats = (f"min={tmap.min():.2f}°C   max={tmap.max():.2f}°C\n"
             f"ΔT={tmap.max()-tmap.min():.3f}°C   "
             f"CV={tmap.std()/tmap.mean():.5f}")
    ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=7.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"))

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def make_comparison_figure(
    pcb_exp: np.ndarray, die_exp: np.ndarray,
    pcb_opt: np.ndarray, die_opt: np.ndarray,
    exp_profile: ReflowProfile, opt_profile: ReflowProfile,
    pct_pcb: np.ndarray, pct_die: np.ndarray,
    save_path: Path,
):
    """
    3-row × 4-column publication figure:
      Row 0: PCB  [Exp | Opt | %ΔT | T-t curve]
      Row 1: DIE  [Exp | Opt | %ΔT | Metrics table]
      Row 2: % ΔT histograms  [PCB | DIE]
    """
    plt.rcParams.update({"font.family": "serif", "figure.dpi": 120})

    fig = plt.figure(figsize=(22, 17))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.42)

    x_pcb = np.linspace(-PCB_SIDE_MM/2, PCB_SIDE_MM/2, 50)
    z_pcb = np.linspace(-PCB_SIDE_MM/2, PCB_SIDE_MM/2, 50)
    x_die = np.linspace(-DIE_SIDE_MM/2, DIE_SIDE_MM/2, 50)
    z_die = np.linspace(-DIE_SIDE_MM/2, DIE_SIDE_MM/2, 50)
    XP, ZP = np.meshgrid(x_pcb, z_pcb)
    XD, ZD = np.meshgrid(x_die, z_die)

    N = 22
    pcb_lvls = np.linspace(min(pcb_exp.min(), pcb_opt.min()),
                           max(pcb_exp.max(), pcb_opt.max()), N + 1)
    die_lvls = np.linspace(min(die_exp.min(), die_opt.min()),
                           max(die_exp.max(), die_opt.max()), N + 1)

    def _contour_ax(ax, X, Z, data, lvls, cbar_lbl, title):
        cf = ax.contourf(X, Z, data, levels=lvls, cmap="viridis")
        ax.contour(X, Z, data, levels=lvls[::4], colors="k", lw=0.35, alpha=0.45)
        fig.colorbar(cf, ax=ax, label=cbar_lbl, pad=0.02)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("X (mm)", fontsize=9)
        ax.set_ylabel("Z (mm)", fontsize=9)
        ax.set_aspect("equal")

    # ── Row 0: PCB ────────────────────────────────────────────────────────────
    _contour_ax(fig.add_subplot(gs[0, 0]), XP, ZP, pcb_exp, pcb_lvls, "T (°C)",
                f"PCB — Experimental\n"
                f"peak={exp_profile.peak_temp_C:.0f}°C  "
                f"cool={exp_profile.cooling_rate_Cps:.1f}°C/s  "
                f"TAL={exp_profile.tal_s:.0f}s")

    _contour_ax(fig.add_subplot(gs[0, 1]), XP, ZP, pcb_opt, pcb_lvls, "T (°C)",
                f"PCB — Optimized\n"
                f"peak={opt_profile.peak_temp_C:.1f}°C  "
                f"cool={opt_profile.cooling_rate_Cps:.2f}°C/s  "
                f"TAL={opt_profile.tal_s:.1f}s")

    # PCB % ΔT contour
    ax_pct = fig.add_subplot(gs[0, 2])
    lim     = max(abs(pct_pcb.min()), abs(pct_pcb.max()))
    cf_pct  = ax_pct.contourf(XP, ZP, pct_pcb,
                               levels=np.linspace(-lim, lim, N + 1), cmap="RdBu_r")
    ax_pct.contour(XP, ZP, pct_pcb,
                   levels=np.linspace(-lim, lim, N + 1)[::4],
                   colors="k", lw=0.3, alpha=0.4)
    fig.colorbar(cf_pct, ax=ax_pct, label="% ΔT", pad=0.02)
    ax_pct.set_title(f"PCB — % ΔT  (Opt − Exp)\nmean={pct_pcb.mean():.3f}%",
                     fontsize=9.5, fontweight="bold")
    ax_pct.set_xlabel("X (mm)", fontsize=9)
    ax_pct.set_ylabel("Z (mm)", fontsize=9)
    ax_pct.set_aspect("equal")

    # T-t curve
    ax_tt = fig.add_subplot(gs[0, 3])
    for prof, lbl, col in [
        (exp_profile, f"Experimental ({exp_profile.peak_temp_C:.0f}°C)", "#1565c0"),
        (opt_profile, f"Optimized ({opt_profile.peak_temp_C:.1f}°C)",    "#c62828"),
    ]:
        t_arr, T_arr = build_tt_curve(prof)   # uses existing visualizer helper
        ax_tt.plot(t_arr, T_arr, lw=2, color=col, label=lbl)
    ax_tt.axhline(217, ls="--", lw=1.2, color="orange", label="Liquidus 217°C")
    ax_tt.set_xlabel("Time (s)", fontsize=9)
    ax_tt.set_ylabel("Temp (°C)", fontsize=9)
    ax_tt.set_title("Reflow T-t Profile\nExperimental vs Optimized",
                    fontsize=9.5, fontweight="bold")
    ax_tt.legend(fontsize=8); ax_tt.set_ylim(0, 285); ax_tt.grid(alpha=0.3)

    # ── Row 1: DIE ────────────────────────────────────────────────────────────
    _contour_ax(fig.add_subplot(gs[1, 0]), XD, ZD, die_exp, die_lvls, "T (°C)",
                f"DIE — Experimental  peak={exp_profile.peak_temp_C:.0f}°C")

    _contour_ax(fig.add_subplot(gs[1, 1]), XD, ZD, die_opt, die_lvls, "T (°C)",
                f"DIE — Optimized  peak={opt_profile.peak_temp_C:.1f}°C")

    ax_dpct = fig.add_subplot(gs[1, 2])
    lim2    = max(abs(pct_die.min()), abs(pct_die.max()))
    cf_dpct = ax_dpct.contourf(XD, ZD, pct_die,
                                levels=np.linspace(-lim2, lim2, N + 1), cmap="RdBu_r")
    ax_dpct.contour(XD, ZD, pct_die,
                    levels=np.linspace(-lim2, lim2, N + 1)[::4],
                    colors="k", lw=0.3, alpha=0.4)
    fig.colorbar(cf_dpct, ax=ax_dpct, label="% ΔT", pad=0.02)
    ax_dpct.set_title(f"DIE — % ΔT  (Opt − Exp)\nmean={pct_die.mean():.4f}%",
                      fontsize=9.5, fontweight="bold")
    ax_dpct.set_xlabel("X (mm)", fontsize=9)
    ax_dpct.set_ylabel("Z (mm)", fontsize=9)
    ax_dpct.set_aspect("equal")

    # Metrics summary table
    ax_tbl = fig.add_subplot(gs[1, 3])
    ax_tbl.axis("off")

    pcb_cv_e = pcb_exp.std() / pcb_exp.mean()
    pcb_cv_o = pcb_opt.std() / pcb_opt.mean()
    die_cv_e = die_exp.std() / die_exp.mean()
    die_cv_o = die_opt.std() / die_opt.mean()

    rows = [
        ["Peak Temp (°C)",
         f"{exp_profile.peak_temp_C:.1f}",
         f"{opt_profile.peak_temp_C:.2f}",
         f"{opt_profile.peak_temp_C - exp_profile.peak_temp_C:+.1f}"],
        ["TAL (s)",
         f"{exp_profile.tal_s:.1f}",
         f"{opt_profile.tal_s:.1f}",
         f"{opt_profile.tal_s - exp_profile.tal_s:+.1f}"],
        ["Soak Time (s)",
         f"{exp_profile.soak_time_s:.0f}",
         f"{opt_profile.soak_time_s:.0f}",
         f"{opt_profile.soak_time_s - exp_profile.soak_time_s:+.0f}"],
        ["Cooling (°C/s)",
         f"{exp_profile.cooling_rate_Cps:.2f}",
         f"{opt_profile.cooling_rate_Cps:.2f}",
         f"{opt_profile.cooling_rate_Cps - exp_profile.cooling_rate_Cps:+.2f}"],
        ["Ramp (°C/s)",
         f"{exp_profile.ramp_rate_Cps:.2f}",
         f"{opt_profile.ramp_rate_Cps:.2f}",
         f"{opt_profile.ramp_rate_Cps - exp_profile.ramp_rate_Cps:+.2f}"],
        ["PCB ΔT (°C)",
         f"{pcb_exp.max()-pcb_exp.min():.3f}",
         f"{pcb_opt.max()-pcb_opt.min():.3f}",
         f"{(pcb_opt.max()-pcb_opt.min())-(pcb_exp.max()-pcb_exp.min()):+.3f}"],
        ["DIE ΔT (°C)",
         f"{die_exp.max()-die_exp.min():.5f}",
         f"{die_opt.max()-die_opt.min():.5f}",
         f"{(die_opt.max()-die_opt.min())-(die_exp.max()-die_exp.min()):+.5f}"],
        ["PCB CV",
         f"{pcb_cv_e:.5f}",
         f"{pcb_cv_o:.5f}",
         f"{(pcb_cv_e-pcb_cv_o)/pcb_cv_e*100:+.1f}%"],
        ["DIE CV",
         f"{die_cv_e:.6f}",
         f"{die_cv_o:.6f}",
         f"{(die_cv_e-die_cv_o)/die_cv_e*100:+.1f}%"],
        ["PCB mean % ΔT", "—", "—", f"{pct_pcb.mean():.3f}%"],
        ["DIE mean % ΔT", "—", "—", f"{pct_die.mean():.4f}%"],
    ]
    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=["Metric", "Experimental", "Optimized", "Δ"],
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.05, 1.62)
    for j in range(4):
        tbl[0, j].set_facecolor("#1565c0")
        tbl[0, j].get_text().set_color("white")
        tbl[0, j].get_text().set_fontweight("bold")
    # Highlight peak temp row (green = reduced)
    for j in range(4):
        tbl[1, j].set_facecolor("#e8f5e9")
    ax_tbl.set_title("Metrics Summary", fontweight="bold", fontsize=10, pad=8)

    # ── Row 2: % ΔT histograms ────────────────────────────────────────────────
    ax_ph = fig.add_subplot(gs[2, :2])
    ax_ph.hist(pct_pcb.ravel(), bins=60, color="#1565c0", edgecolor="k",
               linewidth=0.3, alpha=0.8)
    ax_ph.axvline(pct_pcb.mean(), ls="--", lw=1.8, color="#c62828",
                  label=f"Mean = {pct_pcb.mean():.3f}%")
    ax_ph.axvline(0, ls="-", lw=1, color="k", alpha=0.4)
    ax_ph.set_xlabel("% ΔT  =  (T_opt − T_exp) / T_exp × 100", fontsize=10)
    ax_ph.set_ylabel("Pixel count", fontsize=10)
    ax_ph.set_title("PCB — % Temperature Change Distribution (2D Surface)",
                    fontweight="bold", fontsize=10)
    ax_ph.legend(fontsize=9); ax_ph.grid(alpha=0.3)

    ax_dh = fig.add_subplot(gs[2, 2:])
    ax_dh.hist(pct_die.ravel(), bins=60, color="#00897b", edgecolor="k",
               linewidth=0.3, alpha=0.8)
    ax_dh.axvline(pct_die.mean(), ls="--", lw=1.8, color="#c62828",
                  label=f"Mean = {pct_die.mean():.4f}%")
    ax_dh.axvline(0, ls="-", lw=1, color="k", alpha=0.4)
    ax_dh.set_xlabel("% ΔT  =  (T_opt − T_exp) / T_exp × 100", fontsize=10)
    ax_dh.set_ylabel("Pixel count", fontsize=10)
    ax_dh.set_title("DIE — % Temperature Change Distribution (2D Surface)",
                    fontweight="bold", fontsize=10)
    ax_dh.legend(fontsize=9); ax_dh.grid(alpha=0.3)

    fig.suptitle(
        "PCB Reflow Thermal Digital Twin — Experimental vs Optimized Profile\n"
        f"Experimental peak: {exp_profile.peak_temp_C:.0f}°C  →  "
        f"Optimized peak: {opt_profile.peak_temp_C:.1f}°C  "
        f"(Δ = {opt_profile.peak_temp_C - exp_profile.peak_temp_C:+.1f}°C)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# =============================================================================
# Main pipeline
# =============================================================================

def run_stage3(args):

    # ── Step 1: Load engine ───────────────────────────────────────────────────
    banner("Step 1 — Load Inference Engine")
    engine = InferenceEngine(args.model, verbose=True)
    engine.summary()

    # ── Step 2: Experimental profile ──────────────────────────────────────────
    banner("Step 2 — Experimental Profile")
    exp_profile = EXPERIMENTAL_PROFILE

    print(f"  Peak temp   : {exp_profile.peak_temp_C}°C")
    print(f"  TAL         : {exp_profile.tal_s}s")
    print(f"  Soak temp   : {exp_profile.soak_temp_C}°C  /  {exp_profile.soak_time_s}s")
    print(f"  Ramp rate   : {exp_profile.ramp_rate_Cps}°C/s")
    print(f"  Cooling rate: {exp_profile.cooling_rate_Cps}°C/s")

    # ── Step 3: Get experimental thermal maps ─────────────────────────────────
    banner("Step 3 — Experimental Thermal Maps")

    if args.pcb and args.die and os.path.exists(args.pcb) and os.path.exists(args.die):
        print("  Loading ground-truth FEA maps from CSV...")
        pcb_exp, die_exp = load_fea_maps(args.pcb, args.die)
        print("  ✅ FEA maps loaded (ground truth)")
    else:
        print("  No CSV supplied — predicting experimental maps with model...")
        result_exp = engine.predict(exp_profile)
        pcb_exp    = result_exp.pcb_map
        die_exp    = result_exp.die_map
        print("  ✅ Predicted by model")

    print(f"  PCB: {pcb_exp.min():.2f} → {pcb_exp.max():.2f} °C  "
          f"(ΔT={pcb_exp.max()-pcb_exp.min():.3f}°C)")
    print(f"  DIE: {die_exp.min():.4f} → {die_exp.max():.4f} °C  "
          f"(ΔT={die_exp.max()-die_exp.min():.5f}°C)")

    # ── Step 4: Advisor evaluation of experimental profile ───────────────────
    banner("Step 4 — Advisor Evaluation (Experimental)")
    advisor     = ReflowAdvisor(engine)
    exp_report  = advisor.evaluate(exp_profile, verbose=True)

    # ── Step 5: Optimize  (hard constraint: peak_temp < experimental) ─────────
    banner(f"Step 5 — NSGA-II Optimization  "
           f"(peak_temp must be < {exp_profile.peak_temp_C:.0f}°C)")

    # Temporarily tighten the peak_temp upper bound in PROFILE_BOUNDS
    # so the optimizer never exceeds the experimental value
    import copy
    tight_bounds = copy.deepcopy(BOUNDS_ARRAY)
    peak_idx = PARAM_KEYS.index('peak_temp_C')
    tight_bounds[peak_idx, 1] = exp_profile.peak_temp_C - 0.1

    opt = ReflowOptimizer(advisor, bounds=tight_bounds, seed=42)

    pareto = opt.run_nsga2(
        n_pop         = args.nsga_pop,
        n_generations = args.nsga_gens,
        verbose       = True,
    )

    if not pareto:
        print("\n  ⚠️  NSGA-II unavailable (pymoo not installed). Running GA fallback...")
        opt_ga = ReflowOptimizer(advisor, bounds=tight_bounds, seed=42)
        opt_profile, ga_history = opt_ga.run_ga(
            n_pop=args.nsga_pop, n_generations=args.nsga_gens,
            initial_profile=exp_profile, verbose=True,
        )
        pareto = None
    else:
        # Best = minimum combined objective
        best_sol    = pareto[0]
        opt_profile = best_sol.profile
        print(f"\n  Pareto solutions : {len(pareto)}")

    # Verify constraint
    assert opt_profile.peak_temp_C < exp_profile.peak_temp_C, \
        "Constraint violated: optimized peak_temp must be < experimental!"
    print(f"\n  ✅ Constraint satisfied:  "
          f"{opt_profile.peak_temp_C:.2f}°C < {exp_profile.peak_temp_C:.0f}°C")

    # Print parameter comparison
    print(f"\n  {'Parameter':<30} {'Experimental':>14} {'Optimized':>12} {'Δ':>10}")
    print("  " + "-" * 70)
    for k in ReflowProfile.FEATURE_COLS:
        ev = getattr(exp_profile, k)
        ov = getattr(opt_profile, k)
        marker = "  ◄ CONSTRAINED" if k == "peak_temp_C" else ""
        print(f"  {k:<30} {ev:>14.4f} {ov:>12.4f} {ov-ev:>+10.4f}{marker}")

    # ── Step 6: Advisor evaluation of optimized profile ───────────────────────
    banner("Step 6 — Advisor Evaluation (Optimized)")
    opt_result = engine.predict(opt_profile)
    opt_report = advisor.evaluate(opt_profile, verbose=True)

    # ── Step 7: Get optimized thermal maps ────────────────────────────────────
    banner("Step 7 — Optimized Thermal Maps")
    pcb_opt = opt_result.pcb_map
    die_opt = opt_result.die_map
    print(f"  PCB: {pcb_opt.min():.2f} → {pcb_opt.max():.2f} °C  "
          f"(ΔT={pcb_opt.max()-pcb_opt.min():.3f}°C)")
    print(f"  DIE: {die_opt.min():.4f} → {die_opt.max():.4f} °C  "
          f"(ΔT={die_opt.max()-die_opt.min():.5f}°C)")

    # ── Step 8: Compute % ΔT ─────────────────────────────────────────────────
    banner("Step 8 — % ΔT Analysis")
    pct_pcb = (pcb_opt - pcb_exp) / np.abs(pcb_exp) * 100.0
    pct_die = (die_opt - die_exp) / np.abs(die_exp) * 100.0

    pcb_cv_e = pcb_exp.std() / pcb_exp.mean()
    pcb_cv_o = pcb_opt.std() / pcb_opt.mean()
    die_cv_e = die_exp.std() / die_exp.mean()
    die_cv_o = die_opt.std() / die_opt.mean()

    print(f"  PCB % ΔT  : mean={pct_pcb.mean():.3f}%  "
          f"min={pct_pcb.min():.3f}%  max={pct_pcb.max():.3f}%")
    print(f"  DIE % ΔT  : mean={pct_die.mean():.4f}%  "
          f"min={pct_die.min():.4f}%  max={pct_die.max():.4f}%")
    print(f"  PCB CV    : {pcb_cv_e:.5f} → {pcb_cv_o:.5f}  "
          f"({(pcb_cv_e-pcb_cv_o)/pcb_cv_e*100:+.1f}%)")
    print(f"  DIE CV    : {die_cv_e:.6f} → {die_cv_o:.6f}  "
          f"({(die_cv_e-die_cv_o)/die_cv_e*100:+.1f}%)")

    # ── Step 9: Generate figures ───────────────────────────────────────────────
    banner("Step 9 — Generating Figures")

    viz = Visualizer(str(OUTPUT_DIR) + "/")

    # Fig 6a — Experimental PCB
    make_fig6_contour(
        pcb_exp, PCB_SIDE_MM,
        f"Experimental PCB Temperature Contour\n"
        f"peak={exp_profile.peak_temp_C:.0f}°C  "
        f"TAL={exp_profile.tal_s:.0f}s  "
        f"cool={exp_profile.cooling_rate_Cps:.1f}°C/s",
        OUTPUT_DIR / "fig6a_exp_PCB.png"
    )

    # Fig 6b — Experimental DIE
    make_fig6_contour(
        die_exp, DIE_SIDE_MM,
        f"Experimental DIE Temperature Contour\n"
        f"peak={exp_profile.peak_temp_C:.0f}°C  "
        f"TAL={exp_profile.tal_s:.0f}s",
        OUTPUT_DIR / "fig6b_exp_DIE.png"
    )

    # Fig 6c — Optimized PCB
    make_fig6_contour(
        pcb_opt, PCB_SIDE_MM,
        f"Optimized PCB Temperature Contour\n"
        f"peak={opt_profile.peak_temp_C:.1f}°C  "
        f"TAL={opt_profile.tal_s:.1f}s  "
        f"cool={opt_profile.cooling_rate_Cps:.2f}°C/s",
        OUTPUT_DIR / "fig6c_opt_PCB.png"
    )

    # Fig 6d — Optimized DIE
    make_fig6_contour(
        die_opt, DIE_SIDE_MM,
        f"Optimized DIE Temperature Contour\n"
        f"peak={opt_profile.peak_temp_C:.1f}°C  "
        f"TAL={opt_profile.tal_s:.1f}s",
        OUTPUT_DIR / "fig6d_opt_DIE.png"
    )

    # Fig 7 — Full comparison
    make_comparison_figure(
        pcb_exp, die_exp, pcb_opt, die_opt,
        exp_profile, opt_profile,
        pct_pcb, pct_die,
        OUTPUT_DIR / "fig7_comparison.png"
    )

    # Fig 8 — existing Visualizer comparison (PCB+DIE side by side using Visualizer)
    exp_pred_result = PredictionResult(
        pcb_map=pcb_exp, die_map=die_exp,
        pcb_uniformity=pcb_cv_e, die_uniformity=die_cv_e,
        pcb_gradient_max=float(np.max(np.sqrt(sum(g**2 for g in np.gradient(pcb_exp))))),
        die_gradient_max=float(np.max(np.sqrt(sum(g**2 for g in np.gradient(die_exp))))),
        pcb_range=float(pcb_exp.max()-pcb_exp.min()),
        die_range=float(die_exp.max()-die_exp.min()),
        inference_time_ms=0.0,
    )
    viz.plot_thermal_comparison(
        exp_pred_result, opt_result,
        save_name="fig8_thermal_comparison"
    )

    # Fig 9 — T-t profile comparison using Visualizer
    viz.plot_reflow_profile(
        exp_profile, compare_profile=opt_profile,
        save_name="fig9_profile_comparison"
    )

    # Fig 10 — Advisor dashboard for optimized profile
    viz.plot_dashboard(opt_report, save_name="fig10_opt_dashboard")

    # ── Step 10: Save JSON ────────────────────────────────────────────────────
    banner("Step 10 — Saving Results")
    results = {
        "experimental_profile": exp_profile.to_dict(),
        "optimized_profile":    opt_profile.to_dict(),
        "peak_temp_reduction_C": float(exp_profile.peak_temp_C - opt_profile.peak_temp_C),
        "advisor": {
            "exp_composite_score": float(exp_report.composite_score),
            "opt_composite_score": float(opt_report.composite_score),
            "exp_risk":            exp_report.overall_risk.value,
            "opt_risk":            opt_report.overall_risk.value,
        },
        "thermal_metrics": {
            "pcb_range_exp":    float(pcb_exp.max()-pcb_exp.min()),
            "pcb_range_opt":    float(pcb_opt.max()-pcb_opt.min()),
            "die_range_exp":    float(die_exp.max()-die_exp.min()),
            "die_range_opt":    float(die_opt.max()-die_opt.min()),
            "pcb_cv_exp":       float(pcb_cv_e),
            "pcb_cv_opt":       float(pcb_cv_o),
            "die_cv_exp":       float(die_cv_e),
            "die_cv_opt":       float(die_cv_o),
            "pcb_cv_improvement_pct": float((pcb_cv_e-pcb_cv_o)/pcb_cv_e*100),
            "die_cv_improvement_pct": float((die_cv_e-die_cv_o)/die_cv_e*100),
            "pcb_mean_pct_dT":  float(pct_pcb.mean()),
            "pcb_min_pct_dT":   float(pct_pcb.min()),
            "pcb_max_pct_dT":   float(pct_pcb.max()),
            "die_mean_pct_dT":  float(pct_die.mean()),
            "die_min_pct_dT":   float(pct_die.min()),
            "die_max_pct_dT":   float(pct_die.max()),
        },
        "pareto_solutions": len(pareto) if pareto else 0,
    }

    json_path = OUTPUT_DIR / "stage3_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {json_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    banner("STAGE 3 COMPLETE — SUMMARY")
    print(f"""
  Peak temp reduction : {exp_profile.peak_temp_C:.0f}°C → {opt_profile.peak_temp_C:.1f}°C
  PCB mean % ΔT       : {pct_pcb.mean():.3f}%
  DIE mean % ΔT       : {pct_die.mean():.4f}%
  PCB CV improvement  : {(pcb_cv_e-pcb_cv_o)/pcb_cv_e*100:+.1f}%
  DIE CV improvement  : {(die_cv_e-die_cv_o)/die_cv_e*100:+.1f}%
  Exp composite score : {exp_report.composite_score:.4f}
  Opt composite score : {opt_report.composite_score:.4f}
  Figures → {OUTPUT_DIR}/
  Results → {json_path}
""")
    return results


# =============================================================================
# CLI entry point
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 3 — Experimental Profile Optimization & Fig 6 Generation"
    )
    p.add_argument("--model",     default=DEFAULT_MODEL_PATH,
                   help="Path to thermal_digital_twin.pth")
    p.add_argument("--pcb",       default=None,
                   help="Path to C1_PCB_Tmap_50x50.csv  (optional — uses model prediction if omitted)")
    p.add_argument("--die",       default=None,
                   help="Path to C1_DIE_Tmap_50x50.csv  (optional)")
    p.add_argument("--train",     default=None,
                   help="Path to TRAINING_DATA.xlsx  (not needed for Stage 3)")
    p.add_argument("--nsga-pop",  dest="nsga_pop",  type=int, default=60)
    p.add_argument("--nsga-gens", dest="nsga_gens", type=int, default=80)
    p.add_argument("--quick",     action="store_true",
                   help="Quick test: 20 pop × 20 gens")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.nsga_pop  = 20
        args.nsga_gens = 20
    run_stage3(args)
