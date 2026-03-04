"""
=============================================================================
Stage 2 — Main Pipeline Runner  (UPDATED for Phase 1 model)
PCB Reflow Thermal Digital Twin
=============================================================================
Full end-to-end demo of the intelligent reflow advisory system:

  1.  Load trained PyTorch digital twin
  2.  Define baseline SAC305 profile
  3.  Predict thermal fields (baseline)
  4.  Evaluate with Intelligent Advisor (risk flags + scores)
  5.  Run GA single-objective optimization
  6.  Run NSGA-II multi-objective Pareto search
  7.  Generate publication figures
  8.  Save results JSON

Usage:
    python main.py                                        # Full run
    python main.py --quick                                # 20-gen demo
    python main.py --model path/to/thermal_digital_twin.pth
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
import time
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "model_path":          r"D:\Dhaval Codes\reflow\saved_models_stage1\thermal_digital_twin (1).pth",
    "output_dir":          "figures/",
    "results_dir":         "results/",
    "ga_n_pop":            50,
    "ga_n_generations":    80,
    "nsga2_n_pop":         60,
    "nsga2_n_generations": 80,
    "seed":                42,
}

QUICK_CONFIG = {
    **CONFIG,
    "ga_n_pop":            20,
    "ga_n_generations":    20,
    "nsga2_n_pop":         20,
    "nsga2_n_generations": 20,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(title: str, width: int = 65):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def setup_dirs(config: dict):
    for d in [config["output_dir"], config["results_dir"], "models"]:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config: dict):
    t_total = time.time()

    from inference_engine import InferenceEngine, ReflowProfile
    from reflow_advisor   import ReflowAdvisor
    from optimizer        import ReflowOptimizer
    from visualizer       import Visualizer

    setup_dirs(config)
    viz = Visualizer(config["output_dir"])

    # ── Step 1: Load Model ──────────────────────────────────────────────────
    banner("Step 1 — Load Inference Engine")
    engine = InferenceEngine(config["model_path"], verbose=True)
    engine.summary()

    # ── Step 2: Baseline Profile ────────────────────────────────────────────
    banner("Step 2 — Define Baseline Profile")

    # Case 1 from training data — typical SAC305 profile
    baseline = ReflowProfile(
        peak_temp_C            = 245.0,
        tal_s                  = 47.0,
        soak_temp_C            = 165.0,
        soak_time_s            = 90.0,
        ramp_rate_Cps          = 1.5,
        cooling_rate_Cps       = 3.0,
        t_total_s              = 291.0,
        T_amb_C                = 25.0,
        copper_area_fraction   = 0.20,
        paste_coverage_fraction= 0.01,
        k_die_WmK              = 130.0,
        k_pcb_WmK              = 0.30,
    )

    print(f"  Peak temp      : {baseline.peak_temp_C}°C")
    print(f"  TAL            : {baseline.tal_s}s")
    print(f"  Soak temp      : {baseline.soak_temp_C}°C")
    print(f"  Cooling rate   : {baseline.cooling_rate_Cps}°C/s")
    print(f"  Ramp rate      : {baseline.ramp_rate_Cps}°C/s")

    # ── Step 3: Baseline Prediction ─────────────────────────────────────────
    banner("Step 3 — Baseline Thermal Prediction")
    baseline_result = engine.predict(baseline)

    print(f"  PCB map        : {baseline_result.pcb_map.shape}  "
          f"| {baseline_result.pcb_map.min():.2f} → {baseline_result.pcb_map.max():.2f} °C")
    print(f"  DIE map        : {baseline_result.die_map.shape}  "
          f"| {baseline_result.die_map.min():.4f} → {baseline_result.die_map.max():.4f} °C")
    print(f"  PCB range (°C) : {baseline_result.pcb_range:.3f}")
    print(f"  DIE range (°C) : {baseline_result.die_range:.5f}")
    print(f"  PCB uniformity : {baseline_result.pcb_uniformity:.5f}")
    print(f"  DIE uniformity : {baseline_result.die_uniformity:.6f}")
    print(f"  Inference time : {baseline_result.inference_time_ms:.2f} ms")

    # ── Step 4: Advisor Evaluation ──────────────────────────────────────────
    banner("Step 4 — Intelligent Advisor Evaluation")
    advisor = ReflowAdvisor(engine)
    baseline_report = advisor.evaluate(baseline, verbose=True)

    # ── Step 5: GA Single-Objective ─────────────────────────────────────────
    banner("Step 5 — GA Single-Objective Optimization")
    opt = ReflowOptimizer(advisor, seed=config["seed"])

    ga_best, ga_history = opt.run_ga(
        n_pop        = config["ga_n_pop"],
        n_generations= config["ga_n_generations"],
        initial_profile = baseline,
        verbose      = True,
    )

    ga_result = engine.predict(ga_best)
    ga_report = advisor.evaluate(ga_best, verbose=True)
    comparison = opt.compare_profiles(baseline, ga_best)

    # ── Step 6: NSGA-II Multi-Objective ─────────────────────────────────────
    banner("Step 6 — NSGA-II Multi-Objective Pareto Search")
    pareto = opt.run_nsga2(
        n_pop        = config["nsga2_n_pop"],
        n_generations= config["nsga2_n_generations"],
        verbose      = True,
    )

    if not pareto:
        print("\n  ⚠️  NSGA-II returned no solutions.")
        print("  → Install pymoo first: pip install pymoo")
        print("  → Falling back to GA best solution for figures.\n")
        from optimizer import ParetoSolution
        best_pareto  = ParetoSolution(
            profile         = ga_best,
            objectives      = advisor.score_multi(ga_best),
            composite_score = ga_report.composite_score,
        )
        pareto = [best_pareto]
    else:
        print(f"\n  Top 5 Pareto solutions:")
        print(f"  {'Rank':<5} {'PCB Score':>12} {'DIE Score':>12} {'PeakTemp':>10} {'TAL(s)':>8}")
        print("  " + "-" * 55)
        for i, sol in enumerate(pareto[:5]):
            print(
                f"  {i+1:<5} {sol.objectives[0]:>12.4f} {sol.objectives[1]:>12.4f} "
                f"{sol.profile.peak_temp_C:>10.1f} {sol.profile.tal_s:>8.1f}"
            )

    best_pareto  = pareto[0]
    nsga2_result = engine.predict(best_pareto.profile)

    # ── Step 7: Figures ─────────────────────────────────────────────────────
    banner("Step 7 — Generating Publication Figures")

    print("  [1/4] Thermal field comparison (baseline vs GA-optimized)...")
    viz.plot_thermal_comparison(
        baseline_result, ga_result,
        save_name="fig1_thermal_comparison"
    )

    print("  [2/4] Reflow profile T-t curves...")
    viz.plot_reflow_profile(
        baseline, compare_profile=ga_best,
        save_name="fig2_reflow_profiles"
    )

    print("  [3/4] GA convergence curve...")
    viz.plot_convergence(ga_history, save_name="fig3_ga_convergence")

    print("  [4/4] NSGA-II Pareto front...")
    baseline_multi = advisor.score_multi(baseline)
    viz.plot_pareto_front(
        pareto,
        baseline_score=baseline_multi,
        save_name="fig4_pareto_front"
    )

    # ── Step 8: Save Results ────────────────────────────────────────────────
    banner("Step 8 — Saving Numerical Results")

    results = {
        "baseline": {
            "pcb_range":       float(baseline_result.pcb_range),
            "die_range":       float(baseline_result.die_range),
            "pcb_uniformity":  float(baseline_result.pcb_uniformity),
            "die_uniformity":  float(baseline_result.die_uniformity),
            "composite_score": float(baseline_report.composite_score),
            "overall_risk":    baseline_report.overall_risk.value,
        },
        "ga_optimized": {
            "pcb_range":       float(ga_result.pcb_range),
            "die_range":       float(ga_result.die_range),
            "pcb_uniformity":  float(ga_result.pcb_uniformity),
            "die_uniformity":  float(ga_result.die_uniformity),
            "composite_score": float(ga_report.composite_score),
            "overall_risk":    ga_report.overall_risk.value,
            "improvement_pct": float(comparison["improvement_pct"]),
            "profile":         ga_best.to_dict(),
        },
        "nsga2_pareto": {
            "n_solutions":      len(pareto),
            "best_composite":   float(pareto[0].composite_score),
            "best_pcb_score":   float(pareto[0].objectives[0]),
            "best_die_score":   float(pareto[0].objectives[1]),
            "best_profile":     best_pareto.profile.to_dict(),
        },
        "config": {
            "ga_generations":    config["ga_n_generations"],
            "ga_pop_size":       config["ga_n_pop"],
            "nsga2_generations": config["nsga2_n_generations"],
            "nsga2_pop_size":    config["nsga2_n_pop"],
            "model":             config["model_path"],
        }
    }

    results_path = os.path.join(config["results_dir"], "stage2_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")

    # ── Final Summary ───────────────────────────────────────────────────────
    banner("STAGE 2 COMPLETE — SUMMARY")
    print(f"""
  ┌───────────────────────────────────────────────────────┐
  │  RESULTS SUMMARY                                      │
  ├─────────────────────────────┬──────────────┬──────────┤
  │  Metric                     │  Baseline    │  GA-Opt  │
  ├─────────────────────────────┼──────────────┼──────────┤
  │  PCB Range (°C)             │  {baseline_result.pcb_range:>9.3f}   │  {ga_result.pcb_range:>7.3f} │
  │  DIE Range (°C)             │  {baseline_result.die_range:>9.5f}   │  {ga_result.die_range:>7.5f} │
  │  PCB Uniformity (CV)        │  {baseline_result.pcb_uniformity:>9.5f}   │  {ga_result.pcb_uniformity:>7.5f} │
  │  DIE Uniformity (CV)        │  {baseline_result.die_uniformity:>9.6f}   │  {ga_result.die_uniformity:>7.6f} │
  │  Composite Score            │  {baseline_report.composite_score:>9.4f}   │  {ga_report.composite_score:>7.4f} │
  │  Overall Risk               │  {baseline_report.overall_risk.value:>12s}│  {ga_report.overall_risk.value:>8s}│
  ├─────────────────────────────┴──────────────┴──────────┤
  │  Score Improvement  : {comparison['improvement_pct']:>6.1f}%                        │
  │  Pareto solutions   : {len(pareto):>4d}                              │
  │  Total time         : {time.time()-t_total:>5.1f}s                            │
  └───────────────────────────────────────────────────────┘

  Figures  → {config['output_dir']}
  Results  → {results_path}
  Model    → {'Trained PyTorch digital twin' if engine._loaded else 'DEMO MODE'}
""")

    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 — PCB Reflow Digital Twin Pipeline")
    p.add_argument("--model",  type=str, default=CONFIG["model_path"],
                   help="Path to trained .pth model (default: models/thermal_digital_twin.pth)")
    p.add_argument("--output", type=str, default=CONFIG["output_dir"])
    p.add_argument("--quick",  action="store_true",
                   help="Run with reduced generations (rapid test)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = QUICK_CONFIG if args.quick else CONFIG
    cfg["model_path"] = args.model
    cfg["output_dir"] = args.output

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║   PCB Reflow Thermal Digital Twin — Stage 2 Pipeline        ║
║   Intelligent Reflow Advisory & Optimization System         ║
╚══════════════════════════════════════════════════════════════╝

  Mode     : {'QUICK DEMO' if args.quick else 'FULL RUN'}
  Model    : {cfg['model_path']}
  Figures  : {cfg['output_dir']}
  GA gens  : {cfg['ga_n_generations']}
  NSGA2    : {cfg['nsga2_n_generations']}
""")

    run_pipeline(cfg)