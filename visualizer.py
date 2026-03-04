"""
=============================================================================
Stage 2 — Visualizer  (UPDATED for Phase 1 model)
PCB Reflow Thermal Digital Twin
=============================================================================
Publication-quality figures for:
  • Thermal field heatmaps (PCB & DIE, real °C)
  • Baseline vs optimized comparisons
  • Pareto front scatter
  • GA convergence curves
  • Reflow T-t profile curves
  • Composite advisor dashboard

All figures saved as 300 DPI PNG + PDF, IEEE/Elsevier ready.

Usage:
    viz = Visualizer("figures/")
    viz.plot_thermal_comparison(baseline_result, opt_result)
    viz.plot_pareto_front(pareto_solutions)
    viz.plot_convergence(history)
    viz.plot_reflow_profile(profile)
    viz.plot_dashboard(report)
=============================================================================
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Optional, Tuple

from inference_engine import ReflowProfile, PredictionResult
from reflow_advisor   import AdvisorReport, RiskLevel


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# Custom thermal colormap
THERMAL_CMAP = LinearSegmentedColormap.from_list(
    "thermal_reflow",
    [(0.00, "#0d1b6e"), (0.20, "#1565c0"), (0.40, "#00bcd4"),
     (0.55, "#4caf50"), (0.70, "#ffeb3b"), (0.85, "#ff5722"),
     (1.00, "#b71c1c")],
    N=512
)


# ---------------------------------------------------------------------------
# Reflow T-t curve builder (uses 12-feature ReflowProfile)
# ---------------------------------------------------------------------------

def build_tt_curve(p: ReflowProfile):
    """
    Build a realistic temperature-time curve from the 12-feature ReflowProfile.
    Returns (time_array, temp_array).
    """
    # Zones: ambient → soak → ramp to peak → TAL plateau → cooling
    t, T = [0.0], [p.T_amb_C]

    # Preheat ramp to soak temp
    preheat_time = (p.soak_temp_C - p.T_amb_C) / max(p.ramp_rate_Cps, 0.01)
    t.append(t[-1] + preheat_time)
    T.append(p.soak_temp_C)

    # Soak zone
    t.append(t[-1] + p.soak_time_s)
    T.append(p.soak_temp_C)

    # Ramp to peak
    ramp_time = (p.peak_temp_C - p.soak_temp_C) / max(p.ramp_rate_Cps, 0.01)
    t.append(t[-1] + ramp_time)
    T.append(p.peak_temp_C)

    # TAL plateau (approximate as flat at peak)
    t.append(t[-1] + p.tal_s)
    T.append(p.peak_temp_C)

    # Cooling
    cool_time = (p.peak_temp_C - p.T_amb_C) / max(p.cooling_rate_Cps, 0.01)
    t.append(t[-1] + cool_time)
    T.append(p.T_amb_C)

    return np.array(t), np.array(T)


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class Visualizer:
    """Publication-quality visualization for the thermal digital twin."""

    def __init__(self, output_dir: str = "figures/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save(self, fig, name: str):
        for fmt, dpi in [("png", 200), ("pdf", 300)]:
            path = os.path.join(self.output_dir, f"{name}.{fmt}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  [Saved] {os.path.join(self.output_dir, name)}.png/.pdf")

    # ------------------------------------------------------------------
    # 1. Single Thermal Field
    # ------------------------------------------------------------------

    def plot_thermal_field(
        self,
        thermal_map: np.ndarray,
        title:       str = "Thermal Field",
        ax:          Optional[plt.Axes] = None,
        show_cbar:   bool = True,
        save_name:   str  = None,
    ) -> plt.Figure:
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(5, 4.5))
        else:
            fig = ax.get_figure()

        im = ax.imshow(thermal_map, cmap=THERMAL_CMAP, origin="lower",
                       interpolation="bilinear", aspect="equal")
        ax.set_title(title, pad=8, fontweight="bold")
        ax.set_xlabel("X position (pixels)")
        ax.set_ylabel("Y position (pixels)")

        if show_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.08)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label("Temperature (°C)")

        stats = (f"min={thermal_map.min():.2f}  max={thermal_map.max():.2f}\n"
                 f"range={thermal_map.max()-thermal_map.min():.3f}°C  "
                 f"CV={np.std(thermal_map)/np.mean(thermal_map):.5f}")
        ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=7, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="gray"))

        if standalone:
            plt.tight_layout()
            if save_name:
                self._save(fig, save_name)
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # 2. Thermal Comparison (baseline vs optimized)
    # ------------------------------------------------------------------

    def plot_thermal_comparison(
        self,
        baseline:  PredictionResult,
        optimized: PredictionResult,
        save_name: str = "thermal_comparison",
    ) -> plt.Figure:
        """
        2×3 grid: [Baseline | Optimized | Difference] for PCB (row 0) and DIE (row 1).
        All maps in real °C.
        """
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))

        b_pcb, o_pcb = baseline.pcb_map,  optimized.pcb_map
        b_die, o_die = baseline.die_map,   optimized.die_map

        pcb_vmin = min(b_pcb.min(), o_pcb.min())
        pcb_vmax = max(b_pcb.max(), o_pcb.max())
        die_vmin = min(b_die.min(), o_die.min())
        die_vmax = max(b_die.max(), o_die.max())

        def _imshow(ax, data, vmin, vmax, title):
            im = ax.imshow(data, cmap=THERMAL_CMAP, origin="lower",
                           interpolation="bilinear", aspect="equal",
                           vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            return im

        def _diff_imshow(ax, base, opt, title):
            diff = opt - base
            lim  = max(abs(diff.min()), abs(diff.max())) + 1e-9
            im   = ax.imshow(diff, cmap="RdBu_r", origin="lower",
                             interpolation="bilinear", aspect="equal",
                             vmin=-lim, vmax=lim)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            return im

        im_bp = _imshow(axes[0,0], b_pcb, pcb_vmin, pcb_vmax, "PCB — Baseline")
        im_op = _imshow(axes[0,1], o_pcb, pcb_vmin, pcb_vmax, "PCB — Optimized")
        im_dp = _diff_imshow(axes[0,2], b_pcb, o_pcb, "PCB — Δ (Opt − Base)")

        im_bd = _imshow(axes[1,0], b_die, die_vmin, die_vmax, "DIE — Baseline")
        im_od = _imshow(axes[1,1], o_die, die_vmin, die_vmax, "DIE — Optimized")
        im_dd = _diff_imshow(axes[1,2], b_die, o_die, "DIE — Δ (Opt − Base)")

        # Colorbars
        for row, im_main, im_diff, label in [
            (0, im_op, im_dp, "PCB Temp (°C)"),
            (1, im_od, im_dd, "DIE Temp (°C)"),
        ]:
            fig.colorbar(im_main,  ax=axes[row, :2], shrink=0.7, label=label,    pad=0.02)
            fig.colorbar(im_diff,  ax=axes[row, 2],  shrink=0.9, label="Δ (°C)", pad=0.02)

        # Metrics annotation
        b_cv_pcb = np.std(b_pcb) / np.mean(b_pcb)
        o_cv_pcb = np.std(o_pcb) / np.mean(o_pcb)
        b_cv_die = np.std(b_die) / np.mean(b_die)
        o_cv_die = np.std(o_die) / np.mean(o_die)

        metrics = (
            f"PCB Uniformity (CV): {b_cv_pcb:.5f} → {o_cv_pcb:.5f}  "
            f"({(b_cv_pcb-o_cv_pcb)/b_cv_pcb*100:+.1f}%)    |    "
            f"DIE Uniformity (CV): {b_cv_die:.6f} → {o_cv_die:.6f}  "
            f"({(b_cv_die-o_cv_die)/b_cv_die*100:+.1f}%)    |    "
            f"PCB range: {baseline.pcb_range:.2f} → {optimized.pcb_range:.2f} °C    |    "
            f"DIE range: {baseline.die_range:.4f} → {optimized.die_range:.4f} °C"
        )
        fig.text(0.5, 0.01, metrics, ha="center", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#f0f4f8", ec="#aaa"))

        fig.suptitle("Thermal Digital Twin — Baseline vs Optimized Reflow Profile",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        self._save(fig, save_name)
        return fig

    # ------------------------------------------------------------------
    # 3. Reflow T-t Profile Curve
    # ------------------------------------------------------------------

    def plot_reflow_profile(
        self,
        profile:         ReflowProfile,
        compare_profile: Optional[ReflowProfile] = None,
        save_name:       str = "reflow_profile",
    ) -> plt.Figure:
        """
        Temperature-time curve for SAC305 reflow profile.
        Optionally overlay a second profile for comparison.
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        t1, T1 = build_tt_curve(profile)
        ax.plot(t1, T1, "b-", lw=2.5,
                label="Baseline" if compare_profile else "Profile")

        if compare_profile is not None:
            t2, T2 = build_tt_curve(compare_profile)
            ax.plot(t2, T2, "r--", lw=2.5, label="Optimized")

        # SAC305 reference lines
        ax.axhline(217, color="orange", ls="--", lw=1.2, alpha=0.8,
                   label="SAC305 liquidus (217°C)")
        ax.axhline(260, color="red",    ls=":",  lw=1.0, alpha=0.7,
                   label="Max peak (260°C)")
        ax.axhline(235, color="green",  ls=":",  lw=1.0, alpha=0.7,
                   label="Min peak (235°C)")

        # Zone shading (approximate from first profile)
        preheat_end = (profile.soak_temp_C - profile.T_amb_C) / max(profile.ramp_rate_Cps, 0.01)
        soak_end    = preheat_end + profile.soak_time_s
        ax.axvspan(0,          preheat_end, alpha=0.10, color="blue",   label="Preheat")
        ax.axvspan(preheat_end, soak_end,   alpha=0.10, color="green",  label="Soak")
        ax.axvspan(soak_end,   soak_end +
                   (profile.peak_temp_C - profile.soak_temp_C) /
                   max(profile.ramp_rate_Cps, 0.01) +
                   profile.tal_s,            alpha=0.10, color="red", label="Reflow/TAL")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("SAC305 Reflow Temperature-Time Profile", fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0, 290)
        ax.set_xlim(0, max(t1) * 1.05)

        # Annotate key values
        ax.annotate(
            f"Peak = {profile.peak_temp_C:.1f}°C\nTAL = {profile.tal_s:.1f}s",
            xy=(t1[3], T1[3]),
            xytext=(t1[3] - 30, T1[3] - 60),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray"),
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray")
        )

        plt.tight_layout()
        self._save(fig, save_name)
        return fig

    # ------------------------------------------------------------------
    # 4. Pareto Front
    # ------------------------------------------------------------------

    def plot_pareto_front(
        self,
        pareto_solutions,
        baseline_score: Optional[Tuple[float, float]] = None,
        save_name:      str = "pareto_front",
    ) -> plt.Figure:
        """
        Pareto front scatter: PCB score vs DIE score.
        Right panel: peak_temp vs TAL colored by PCB score.
        """
        pcb_obj = [s.objectives[0] for s in pareto_solutions]
        die_obj = [s.objectives[1] for s in pareto_solutions]
        comp    = [s.composite_score for s in pareto_solutions]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Pareto scatter
        ax = axes[0]
        sc = ax.scatter(pcb_obj, die_obj, c=comp, cmap="plasma_r",
                        s=80, edgecolors="k", linewidths=0.5, zorder=5)
        plt.colorbar(sc, ax=ax, label="Composite Score (lower = better)")

        if baseline_score:
            ax.scatter([baseline_score[0]], [baseline_score[1]],
                       c="red", s=150, marker="*", zorder=10,
                       label="Baseline", edgecolors="k")
            ax.legend(fontsize=9)

        ax.set_xlabel("PCB Thermal Non-Uniformity Score (°C, lower = better)")
        ax.set_ylabel("DIE Thermal Non-Uniformity Score (°C, lower = better)")
        ax.set_title("NSGA-II Pareto Front\nPCB vs DIE Thermal Uniformity",
                     fontweight="bold")

        # Right: Profile space — peak temp vs TAL
        ax2 = axes[1]
        peaks = [s.profile.peak_temp_C for s in pareto_solutions]
        tals  = [s.profile.tal_s for s in pareto_solutions]
        sc2   = ax2.scatter(tals, peaks, c=pcb_obj, cmap="viridis",
                            s=80, edgecolors="k", linewidths=0.5)
        plt.colorbar(sc2, ax=ax2, label="PCB Uniformity Score")

        ax2.axhline(235, ls="--", c="green", lw=1.0, label="SAC305 T_min = 235°C")
        ax2.axhline(260, ls="--", c="red",   lw=1.0, label="SAC305 T_max = 260°C")
        ax2.axvline(30,  ls=":",  c="blue",  lw=1.0, label="TAL min = 30s")
        ax2.axvline(60,  ls=":",  c="orange",lw=1.0, label="TAL max = 60s")
        ax2.legend(fontsize=7, loc="upper right")

        ax2.set_xlabel("Time Above Liquidus (s)")
        ax2.set_ylabel("Peak Temperature (°C)")
        ax2.set_title("Pareto Profile Parameter Space", fontweight="bold")

        fig.suptitle("Multi-Objective NSGA-II Pareto Optimization",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, save_name)
        return fig

    # ------------------------------------------------------------------
    # 5. GA Convergence
    # ------------------------------------------------------------------

    def plot_convergence(
        self,
        history,
        save_name: str = "ga_convergence",
    ) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        gens  = history.generations
        best  = history.best_scores
        mean  = history.mean_scores
        worst = history.worst_scores

        ax1.plot(gens, best,  "b-",  lw=2,   label="Best",  zorder=5)
        ax1.plot(gens, mean,  "g--", lw=1.5, label="Mean",  zorder=4)
        ax1.plot(gens, worst, "r:",  lw=1.2, label="Worst", zorder=3)
        ax1.fill_between(gens, best, worst, alpha=0.08, color="steelblue")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Composite Score (lower = better)")
        ax1.set_title("GA Convergence Curve", fontweight="bold")
        ax1.legend()

        improvements = [0] + [best[i-1] - best[i] for i in range(1, len(best))]
        ax2.bar(gens, improvements, color="steelblue", alpha=0.7,
                edgecolor="k", linewidth=0.3)
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Δ Best Score (per generation)")
        ax2.set_title("Generation-wise Improvement", fontweight="bold")

        total = (best[0] - best[-1]) / best[0] * 100 if best[0] > 0 else 0
        ax2.text(0.95, 0.95, f"Total improvement: {total:.1f}%",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))

        plt.tight_layout()
        self._save(fig, save_name)
        return fig

    # ------------------------------------------------------------------
    # 6. Advisor Dashboard
    # ------------------------------------------------------------------

    def plot_dashboard(
        self,
        report:    AdvisorReport,
        save_name: str = "advisor_dashboard",
    ) -> plt.Figure:
        """Full-page dashboard: thermal maps + T-t curve + metrics table."""
        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
                                height_ratios=[1.2, 1.2, 0.6])

        # ── Row 0: Thermal maps ───────────────────────────────────────────
        ax_pcb = fig.add_subplot(gs[0, :2])
        ax_die = fig.add_subplot(gs[0, 2:])

        for ax, tmap, title in [
            (ax_pcb, report.prediction.pcb_map, "PCB Thermal Field (°C)"),
            (ax_die, report.prediction.die_map,  "DIE Thermal Field (°C)"),
        ]:
            im = ax.imshow(tmap, cmap=THERMAL_CMAP, origin="lower",
                           interpolation="bilinear", aspect="equal")
            ax.set_title(title, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.05)
            fig.colorbar(im, cax=cax, label="Temp (°C)")

        # ── Row 1: T-t curve + metrics table ─────────────────────────────
        ax_tt  = fig.add_subplot(gs[1, :2])
        ax_met = fig.add_subplot(gs[1, 2:])

        # T-t curve
        t_arr, T_arr = build_tt_curve(report.profile)
        ax_tt.plot(t_arr, T_arr, "b-", lw=2, label="Profile")
        ax_tt.axhline(217, c="orange", ls="--", lw=1, label="Liquidus 217°C")
        ax_tt.set_xlabel("Time (s)"); ax_tt.set_ylabel("Temp (°C)")
        ax_tt.set_title("Reflow T-t Profile", fontweight="bold")
        ax_tt.legend(fontsize=8); ax_tt.set_ylim(0, 290)

        # Metrics table
        ax_met.axis("off")
        pred = report.prediction
        data = [
            ["Overall Risk",       report.overall_risk.value],
            ["Composite Score",    f"{report.composite_score:.4f}"],
            ["PCB Range (°C)",     f"{pred.pcb_range:.3f}"],
            ["DIE Range (°C)",     f"{pred.die_range:.5f}"],
            ["PCB Uniformity CV",  f"{pred.pcb_uniformity:.5f}"],
            ["DIE Uniformity CV",  f"{pred.die_uniformity:.6f}"],
            ["PCB Max Gradient",   f"{pred.pcb_gradient_max:.3f}"],
            ["DIE Max Gradient",   f"{pred.die_gradient_max:.5f}"],
            ["Inference Time",     f"{pred.inference_time_ms:.1f} ms"],
            ["Critical Flags",     str(report.n_critical)],
            ["Warning Flags",      str(report.n_warning)],
        ]
        table = ax_met.table(cellText=data, colLabels=["Metric", "Value"],
                             loc="center", cellLoc="left")
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.1, 1.4)

        # Color risk row
        risk_col = {RiskLevel.OK: "#2e7d32",
                    RiskLevel.WARNING: "#f57f17",
                    RiskLevel.CRITICAL: "#c62828"}[report.overall_risk]
        table[1, 1].set_facecolor(risk_col)
        table[1, 1].get_text().set_color("white")
        table[1, 1].get_text().set_fontweight("bold")
        ax_met.set_title("Key Metrics", fontweight="bold", pad=10)

        # ── Row 2: Risk flags ─────────────────────────────────────────────
        ax_flags = fig.add_subplot(gs[2, :])
        ax_flags.axis("off")
        flag_text = "  |  ".join([
            f"{'⚠️' if f.level == RiskLevel.WARNING else '🔴'} {f.category}"
            for f in report.flags[:6]
        ]) or "✅ No risk flags detected"
        ax_flags.text(0.5, 0.5, flag_text, ha="center", va="center",
                      fontsize=9, transform=ax_flags.transAxes,
                      bbox=dict(boxstyle="round,pad=0.5", fc="#fafafa", ec="#bbb", lw=1.5))
        ax_flags.set_title("Active Risk Flags", fontweight="bold", pad=4)

        fig.suptitle("Intelligent Reflow Advisor — Full Dashboard",
                     fontsize=14, fontweight="bold", y=1.01)
        self._save(fig, save_name)
        return fig


# ---------------------------------------------------------------------------
# Quick Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from inference_engine import InferenceEngine
    from reflow_advisor   import ReflowAdvisor

    viz     = Visualizer("figures/")
    engine  = InferenceEngine("models/thermal_digital_twin.pth")
    advisor = ReflowAdvisor(engine)

    p1 = ReflowProfile()
    p2 = ReflowProfile(peak_temp_C=250.0, tal_s=55.0, cooling_rate_Cps=2.0)

    r1 = engine.predict(p1)
    r2 = engine.predict(p2)

    report = advisor.evaluate(p1, verbose=False)

    print("\nGenerating all figures...")
    viz.plot_thermal_comparison(r1, r2, save_name="comparison")
    viz.plot_reflow_profile(p1, compare_profile=p2, save_name="profiles")
    viz.plot_dashboard(report, save_name="dashboard")
    print(f"\nAll figures saved to: {viz.output_dir}")