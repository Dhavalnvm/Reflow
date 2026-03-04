"""
app_pages/run_tab.py
Tab 2: pipeline execution — imports existing modules directly
"""

import streamlit as st
import io, copy
import numpy as np


def render_run():
    st.markdown("### 🚀 Run Pipeline")

    # ── Status summary ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    has_model  = bool(st.session_state.get("model_path"))
    has_fea    = "pcb_exp" in st.session_state
    has_train  = bool(st.session_state.get("train_xls_bytes"))
    has_params = "exp_profile_kwargs" in st.session_state

    with c1:
        if has_model:
            st.markdown('<p class="badge-ok">✅ Model loaded</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="badge-warn">⚠️ Demo mode</p>', unsafe_allow_html=True)
    with c2:
        if has_fea:
            st.markdown('<p class="badge-ok">✅ FEA maps loaded</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="badge-warn">⚠️ Will predict exp maps</p>', unsafe_allow_html=True)
    with c3:
        if has_train:
            st.markdown('<p class="badge-ok">✅ Training data loaded</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="badge-warn">⚠️ Using built-in defaults</p>', unsafe_allow_html=True)
    with c4:
        if has_params:
            pk = st.session_state.exp_profile_kwargs.get("peak_temp_C", "?")
            st.markdown(f'<p class="badge-ok">✅ Profile: peak={pk}°C</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="badge-warn">⚠️ Set profile in Step 1</p>', unsafe_allow_html=True)

    st.divider()

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        run_btn = st.button("▶  Run Full Pipeline", type="primary", use_container_width=True)
    with col_info:
        gens = st.session_state.get("opt_settings", {}).get("nsga_gens", 80)
        pop  = st.session_state.get("opt_settings", {}).get("nsga_pop",  60)
        st.caption(f"NSGA-II: {pop} pop × {gens} generations. Expect ~1–3 min on CPU.")

    if run_btn:
        if not has_params:
            st.error("Set your experimental profile parameters in Step 1 first.")
            return

        progress = st.progress(0, "Starting...")
        log_area = st.empty()
        logs     = []

        def _log(msg):
            logs.append(msg)
            log_area.code("\n".join(logs[-20:]))

        try:
            _log("Importing modules...")
            progress.progress(0.05, "Importing modules...")

            # ── Import existing modules ───────────────────────────────────────
            from inference_engine import InferenceEngine, ReflowProfile, PredictionResult
            from reflow_advisor   import ReflowAdvisor
            from optimizer        import ReflowOptimizer, BOUNDS_ARRAY, PARAM_KEYS
            from visualizer       import Visualizer, build_tt_curve
            from experimental_pipeline import (
                make_fig6_contour, make_comparison_figure,
                PCB_SIDE_MM, DIE_SIDE_MM, OUTPUT_DIR
            )

            kwargs = st.session_state.exp_profile_kwargs
            geo    = st.session_state.get("geometry",     {"l_pcb": 40.0, "l_die": 15.0})
            opt_s  = st.session_state.get("opt_settings", {"nsga_pop": 60, "nsga_gens": 80, "peak_ub": 244.9})

            l_pcb = geo["l_pcb"]
            l_die = geo["l_die"]

            # ── Load engine ───────────────────────────────────────────────────
            progress.progress(0.08, "Loading model...")
            _log(f"Loading model: {st.session_state.get('model_path', 'DEMO MODE')}")
            engine = InferenceEngine(
                st.session_state.get("model_path") or "NO_MODEL.pth",
                verbose=False
            )

            # ── Build ReflowProfile ───────────────────────────────────────────
            exp_profile = ReflowProfile(**kwargs)
            _log(f"Experimental profile: peak={exp_profile.peak_temp_C}°C  "
                 f"TAL={exp_profile.tal_s}s  cool={exp_profile.cooling_rate_Cps}°C/s")

            # ── Get experimental thermal maps ─────────────────────────────────
            progress.progress(0.15, "Getting experimental thermal maps...")
            if has_fea:
                pcb_exp = st.session_state.pcb_exp
                die_exp = st.session_state.die_exp
                _log("Using ground-truth FEA maps from CSV")
            else:
                res_exp = engine.predict(exp_profile)
                pcb_exp = res_exp.pcb_map
                die_exp = res_exp.die_map
                _log("Predicted experimental maps with model (no CSV supplied)")

            _log(f"PCB exp: {pcb_exp.min():.2f}→{pcb_exp.max():.2f}°C  "
                 f"ΔT={pcb_exp.max()-pcb_exp.min():.3f}°C")
            _log(f"DIE exp: {die_exp.min():.4f}→{die_exp.max():.4f}°C  "
                 f"ΔT={die_exp.max()-die_exp.min():.5f}°C")

            # ── Advisor on experimental ───────────────────────────────────────
            progress.progress(0.20, "Evaluating experimental profile...")
            advisor    = ReflowAdvisor(engine)
            exp_report = advisor.evaluate(exp_profile, verbose=False)
            _log(f"Exp advisor: composite={exp_report.composite_score:.4f}  "
                 f"risk={exp_report.overall_risk.value}")

            # ── Optimize ──────────────────────────────────────────────────────
            progress.progress(0.25, "Running NSGA-II optimization...")
            _log(f"NSGA-II: pop={opt_s['nsga_pop']}  gens={opt_s['nsga_gens']}  "
                 f"peak_ub={opt_s['peak_ub']:.1f}°C")

            tight_bounds = copy.deepcopy(BOUNDS_ARRAY)
            peak_idx = PARAM_KEYS.index("peak_temp_C")
            tight_bounds[peak_idx, 1] = min(opt_s["peak_ub"],
                                            exp_profile.peak_temp_C - 0.1)

            opt = ReflowOptimizer(advisor, bounds=tight_bounds, seed=42)
            pareto = opt.run_nsga2(
                n_pop=opt_s["nsga_pop"],
                n_generations=opt_s["nsga_gens"],
                verbose=False,
            )

            if not pareto:
                _log("pymoo not installed — running GA fallback...")
                opt_profile, _ = opt.run_ga(
                    n_pop=opt_s["nsga_pop"],
                    n_generations=opt_s["nsga_gens"],
                    initial_profile=exp_profile,
                    verbose=False,
                )
                # Hard-clamp peak_temp even on GA result
                if opt_profile.peak_temp_C >= exp_profile.peak_temp_C:
                    import dataclasses
                    opt_profile = dataclasses.replace(
                        opt_profile,
                        peak_temp_C=exp_profile.peak_temp_C - 1.0
                    )
                pareto = None
            else:
                # ── Post-filter: keep only solutions that satisfy peak constraint ──
                valid = [s for s in pareto
                         if s.profile.peak_temp_C < exp_profile.peak_temp_C]
                if not valid:
                    # Fallback: clamp best solution's peak_temp
                    _log("⚠️  No Pareto solution satisfied peak constraint — "
                         "clamping best solution.")
                    import dataclasses
                    best = pareto[0]
                    clamped = dataclasses.replace(
                        best.profile,
                        peak_temp_C=min(best.profile.peak_temp_C,
                                        exp_profile.peak_temp_C - 1.0)
                    )
                    opt_profile = clamped
                else:
                    pareto = valid          # replace with only compliant solutions
                    opt_profile = pareto[0].profile
                _log(f"Pareto solutions (constraint-compliant): {len(pareto) if pareto else 'clamped'}")

            # Final hard-clamp as absolute safety net
            if opt_profile.peak_temp_C >= exp_profile.peak_temp_C:
                import dataclasses
                opt_profile = dataclasses.replace(
                    opt_profile,
                    peak_temp_C=exp_profile.peak_temp_C - 1.0
                )
                _log("⚠️  Applied hard-clamp to peak_temp (last resort).")

            _log(f"Optimized peak: {opt_profile.peak_temp_C:.2f}°C  "
                 f"(Δ = {opt_profile.peak_temp_C - exp_profile.peak_temp_C:+.1f}°C)  ✅")

            # ── Predict optimized maps ────────────────────────────────────────
            progress.progress(0.72, "Predicting optimized thermal maps...")
            opt_result = engine.predict(opt_profile)
            pcb_opt    = opt_result.pcb_map
            die_opt    = opt_result.die_map
            _log(f"PCB opt: {pcb_opt.min():.2f}→{pcb_opt.max():.2f}°C  "
                 f"ΔT={pcb_opt.max()-pcb_opt.min():.3f}°C")

            # ── Advisor on optimized ──────────────────────────────────────────
            opt_report = advisor.evaluate(opt_profile, verbose=False)
            _log(f"Opt advisor: composite={opt_report.composite_score:.4f}  "
                 f"risk={opt_report.overall_risk.value}")

            # ── % ΔT ──────────────────────────────────────────────────────────
            progress.progress(0.80, "Computing % ΔT...")
            pct_pcb = (pcb_opt - pcb_exp) / np.abs(pcb_exp) * 100.0
            pct_die = (die_opt - die_exp) / np.abs(die_exp) * 100.0
            _log(f"PCB mean % ΔT = {pct_pcb.mean():.3f}%")
            _log(f"DIE mean % ΔT = {pct_die.mean():.4f}%")

            # ── Generate figures (bytes) ──────────────────────────────────────
            progress.progress(0.85, "Generating figures...")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            def _fig6_bytes(tmap, L, title):
                import io as _io
                x = np.linspace(-L/2, L/2, 50)
                z = np.linspace(-L/2, L/2, 50)
                X, Z = np.meshgrid(x, z)
                lvls = np.linspace(tmap.min(), tmap.max(), 23)
                fig, ax = plt.subplots(figsize=(6, 5.2))
                cf = ax.contourf(X, Z, tmap, levels=lvls, cmap="viridis")
                ax.contour(X, Z, tmap, levels=lvls[::4], colors="k", lw=0.4, alpha=0.45)
                cbar = fig.colorbar(cf, ax=ax, pad=0.02)
                cbar.set_label("Temperature (°C)", fontsize=10)
                ax.set_xlabel("X (mm)", fontsize=10); ax.set_ylabel("Z (mm)", fontsize=10)
                ax.set_title(title, fontsize=10, fontweight="bold", pad=7)
                ax.set_aspect("equal")
                stats = (f"ΔT={tmap.max()-tmap.min():.3f}°C   "
                         f"CV={tmap.std()/tmap.mean():.5f}")
                ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=7.5,
                        va="bottom", bbox=dict(boxstyle="round,pad=0.3",
                                               fc="white", alpha=0.8, ec="gray"))
                plt.tight_layout()
                buf = _io.BytesIO()
                fig.savefig(buf, dpi=200, bbox_inches="tight", format="png")
                plt.close(fig); buf.seek(0)
                return buf.read()

            def _comparison_bytes():
                import io as _io
                x_pcb = np.linspace(-l_pcb/2, l_pcb/2, 50)
                z_pcb = np.linspace(-l_pcb/2, l_pcb/2, 50)
                x_die = np.linspace(-l_die/2, l_die/2, 50)
                z_die = np.linspace(-l_die/2, l_die/2, 50)
                XP, ZP = np.meshgrid(x_pcb, z_pcb)
                XD, ZD = np.meshgrid(x_die, z_die)
                N = 22
                pcb_lvls = np.linspace(min(pcb_exp.min(),pcb_opt.min()),
                                       max(pcb_exp.max(),pcb_opt.max()), N+1)
                die_lvls = np.linspace(min(die_exp.min(),die_opt.min()),
                                       max(die_exp.max(),die_opt.max()), N+1)

                fig = plt.figure(figsize=(22, 17))
                gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.42)

                def _cp(ax, X, Z, data, lvls, lbl, title):
                    cf = ax.contourf(X, Z, data, levels=lvls, cmap="viridis")
                    ax.contour(X, Z, data, levels=lvls[::4], colors="k", lw=0.35, alpha=0.45)
                    fig.colorbar(cf, ax=ax, label=lbl, pad=0.02)
                    ax.set_title(title, fontsize=9.5, fontweight="bold")
                    ax.set_xlabel("X (mm)", fontsize=9); ax.set_ylabel("Z (mm)", fontsize=9)
                    ax.set_aspect("equal")

                _cp(fig.add_subplot(gs[0,0]), XP, ZP, pcb_exp, pcb_lvls, "T (°C)",
                    f"PCB — Experimental\npeak={exp_profile.peak_temp_C:.0f}°C  "
                    f"cool={exp_profile.cooling_rate_Cps:.1f}°C/s")
                _cp(fig.add_subplot(gs[0,1]), XP, ZP, pcb_opt, pcb_lvls, "T (°C)",
                    f"PCB — Optimized\npeak={opt_profile.peak_temp_C:.1f}°C  "
                    f"cool={opt_profile.cooling_rate_Cps:.2f}°C/s")

                ax02 = fig.add_subplot(gs[0,2])
                lim  = max(abs(pct_pcb.min()), abs(pct_pcb.max()))
                cf02 = ax02.contourf(XP, ZP, pct_pcb,
                                     levels=np.linspace(-lim,lim,N+1), cmap="RdBu_r")
                fig.colorbar(cf02, ax=ax02, label="% ΔT", pad=0.02)
                ax02.set_title(f"PCB % ΔT\nmean={pct_pcb.mean():.3f}%",
                               fontsize=9.5, fontweight="bold")
                ax02.set_xlabel("X (mm)", fontsize=9); ax02.set_ylabel("Z (mm)", fontsize=9)
                ax02.set_aspect("equal")

                # T-t curve
                ax03 = fig.add_subplot(gs[0,3])
                for prof, lbl, col in [
                    (exp_profile, f"Exp ({exp_profile.peak_temp_C:.0f}°C)", "#1565c0"),
                    (opt_profile, f"Opt ({opt_profile.peak_temp_C:.1f}°C)", "#c62828"),
                ]:
                    t_a, T_a = build_tt_curve(prof)
                    ax03.plot(t_a, T_a, lw=2, color=col, label=lbl)
                ax03.axhline(217, ls="--", lw=1.2, color="orange", label="Liquidus")
                ax03.set_xlabel("Time (s)", fontsize=9); ax03.set_ylabel("Temp (°C)", fontsize=9)
                ax03.set_title("Reflow T-t Profile", fontsize=9.5, fontweight="bold")
                ax03.legend(fontsize=8); ax03.set_ylim(0, 285); ax03.grid(alpha=0.3)

                # DIE row
                _cp(fig.add_subplot(gs[1,0]), XD, ZD, die_exp, die_lvls, "T (°C)",
                    f"DIE — Experimental  peak={exp_profile.peak_temp_C:.0f}°C")
                _cp(fig.add_subplot(gs[1,1]), XD, ZD, die_opt, die_lvls, "T (°C)",
                    f"DIE — Optimized  peak={opt_profile.peak_temp_C:.1f}°C")

                ax12 = fig.add_subplot(gs[1,2])
                lim2 = max(abs(pct_die.min()), abs(pct_die.max()))
                cf12 = ax12.contourf(XD, ZD, pct_die,
                                     levels=np.linspace(-lim2,lim2,N+1), cmap="RdBu_r")
                fig.colorbar(cf12, ax=ax12, label="% ΔT", pad=0.02)
                ax12.set_title(f"DIE % ΔT\nmean={pct_die.mean():.4f}%",
                               fontsize=9.5, fontweight="bold")
                ax12.set_xlabel("X (mm)", fontsize=9); ax12.set_ylabel("Z (mm)", fontsize=9)
                ax12.set_aspect("equal")

                # Metrics table
                ax13 = fig.add_subplot(gs[1,3]); ax13.axis("off")
                pcb_cv_e = pcb_exp.std()/pcb_exp.mean()
                pcb_cv_o = pcb_opt.std()/pcb_opt.mean()
                die_cv_e = die_exp.std()/die_exp.mean()
                die_cv_o = die_opt.std()/die_opt.mean()
                rows = [
                    ["Peak Temp (°C)",  f"{exp_profile.peak_temp_C:.1f}", f"{opt_profile.peak_temp_C:.2f}",
                     f"{opt_profile.peak_temp_C-exp_profile.peak_temp_C:+.1f}"],
                    ["TAL (s)",         f"{exp_profile.tal_s:.1f}",  f"{opt_profile.tal_s:.1f}",
                     f"{opt_profile.tal_s-exp_profile.tal_s:+.1f}"],
                    ["Cooling (°C/s)",  f"{exp_profile.cooling_rate_Cps:.2f}", f"{opt_profile.cooling_rate_Cps:.2f}",
                     f"{opt_profile.cooling_rate_Cps-exp_profile.cooling_rate_Cps:+.2f}"],
                    ["PCB ΔT (°C)",
                     f"{pcb_exp.max()-pcb_exp.min():.3f}", f"{pcb_opt.max()-pcb_opt.min():.3f}",
                     f"{(pcb_opt.max()-pcb_opt.min())-(pcb_exp.max()-pcb_exp.min()):+.3f}"],
                    ["DIE ΔT (°C)",
                     f"{die_exp.max()-die_exp.min():.5f}", f"{die_opt.max()-die_opt.min():.5f}",
                     f"{(die_opt.max()-die_opt.min())-(die_exp.max()-die_exp.min()):+.5f}"],
                    ["PCB CV",  f"{pcb_cv_e:.5f}", f"{pcb_cv_o:.5f}", f"{(pcb_cv_e-pcb_cv_o)/pcb_cv_e*100:+.1f}%"],
                    ["DIE CV",  f"{die_cv_e:.6f}", f"{die_cv_o:.6f}", f"{(die_cv_e-die_cv_o)/die_cv_e*100:+.1f}%"],
                    ["PCB mean % ΔT", "—", "—", f"{pct_pcb.mean():.3f}%"],
                    ["DIE mean % ΔT", "—", "—", f"{pct_die.mean():.4f}%"],
                ]
                tbl = ax13.table(cellText=rows,
                                 colLabels=["Metric","Experimental","Optimized","Δ"],
                                 loc="center", cellLoc="center")
                tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.05, 1.65)
                for j in range(4):
                    tbl[0,j].set_facecolor("#1565c0")
                    tbl[0,j].get_text().set_color("white")
                    tbl[0,j].get_text().set_fontweight("bold")
                for j in range(4): tbl[1,j].set_facecolor("#e8f5e9")
                ax13.set_title("Metrics Summary", fontweight="bold", fontsize=10, pad=8)

                # Histograms
                ax20 = fig.add_subplot(gs[2,:2])
                ax20.hist(pct_pcb.ravel(), bins=60, color="#1565c0", edgecolor="k",
                          linewidth=0.3, alpha=0.8)
                ax20.axvline(pct_pcb.mean(), ls="--", lw=1.8, color="#c62828",
                             label=f"Mean={pct_pcb.mean():.3f}%")
                ax20.axvline(0, ls="-", lw=1, color="k", alpha=0.4)
                ax20.set_xlabel("% ΔT", fontsize=10); ax20.set_ylabel("Pixel count", fontsize=10)
                ax20.set_title("PCB — % ΔT Distribution (2D Surface)", fontweight="bold")
                ax20.legend(fontsize=9); ax20.grid(alpha=0.3)

                ax21 = fig.add_subplot(gs[2,2:])
                ax21.hist(pct_die.ravel(), bins=60, color="#00897b", edgecolor="k",
                          linewidth=0.3, alpha=0.8)
                ax21.axvline(pct_die.mean(), ls="--", lw=1.8, color="#c62828",
                             label=f"Mean={pct_die.mean():.4f}%")
                ax21.axvline(0, ls="-", lw=1, color="k", alpha=0.4)
                ax21.set_xlabel("% ΔT", fontsize=10); ax21.set_ylabel("Pixel count", fontsize=10)
                ax21.set_title("DIE — % ΔT Distribution (2D Surface)", fontweight="bold")
                ax21.legend(fontsize=9); ax21.grid(alpha=0.3)

                fig.suptitle(
                    f"Experimental vs Optimized  |  "
                    f"Peak: {exp_profile.peak_temp_C:.0f}°C → {opt_profile.peak_temp_C:.1f}°C  "
                    f"(Δ={opt_profile.peak_temp_C-exp_profile.peak_temp_C:+.1f}°C)",
                    fontsize=13, fontweight="bold", y=1.01
                )
                plt.tight_layout()
                buf = _io.BytesIO()
                fig.savefig(buf, dpi=200, bbox_inches="tight", format="png")
                plt.close(fig); buf.seek(0)
                return buf.read()

            _log("Rendering Fig 6 plots...")
            fig6a = _fig6_bytes(pcb_exp, l_pcb,
                f"Experimental PCB\npeak={exp_profile.peak_temp_C:.0f}°C  "
                f"TAL={exp_profile.tal_s:.0f}s")
            fig6b = _fig6_bytes(die_exp, l_die,
                f"Experimental DIE\npeak={exp_profile.peak_temp_C:.0f}°C")
            fig6c = _fig6_bytes(pcb_opt, l_pcb,
                f"Optimized PCB\npeak={opt_profile.peak_temp_C:.1f}°C  "
                f"TAL={opt_profile.tal_s:.1f}s")
            fig6d = _fig6_bytes(die_opt, l_die,
                f"Optimized DIE\npeak={opt_profile.peak_temp_C:.1f}°C")

            _log("Rendering comparison figure...")
            fig_cmp = _comparison_bytes()

            # Pareto front figure
            fig_pareto = None
            if pareto:
                import io as _io
                pareto_F = np.array([s.objectives for s in pareto])
                fig_p, ax_p = plt.subplots(figsize=(7, 5))
                sc = ax_p.scatter(pareto_F[:,0], pareto_F[:,1],
                                  c=pareto_F[:,0]+pareto_F[:,1],
                                  cmap="plasma_r", s=60, edgecolors="k", lw=0.4)
                plt.colorbar(sc, ax=ax_p, label="Combined objective")
                ax_p.set_xlabel("PCB Non-Uniformity Score"); ax_p.set_ylabel("DIE Non-Uniformity Score")
                ax_p.set_title("NSGA-II Pareto Front", fontweight="bold")
                ax_p.grid(alpha=0.3)
                plt.tight_layout()
                buf_p = _io.BytesIO()
                fig_p.savefig(buf_p, dpi=180, bbox_inches="tight", format="png")
                plt.close(fig_p); buf_p.seek(0)
                fig_pareto = buf_p.read()

            # ── Build metrics dict ────────────────────────────────────────────
            pcb_cv_e = float(pcb_exp.std()/pcb_exp.mean())
            pcb_cv_o = float(pcb_opt.std()/pcb_opt.mean())
            die_cv_e = float(die_exp.std()/die_exp.mean())
            die_cv_o = float(die_opt.std()/die_opt.mean())

            metrics = dict(
                pcb_range_exp=float(pcb_exp.max()-pcb_exp.min()),
                pcb_range_opt=float(pcb_opt.max()-pcb_opt.min()),
                die_range_exp=float(die_exp.max()-die_exp.min()),
                die_range_opt=float(die_opt.max()-die_opt.min()),
                pcb_cv_exp=pcb_cv_e, pcb_cv_opt=pcb_cv_o,
                die_cv_exp=die_cv_e, die_cv_opt=die_cv_o,
                pcb_cv_improvement_pct=float((pcb_cv_e-pcb_cv_o)/pcb_cv_e*100),
                die_cv_improvement_pct=float((die_cv_e-die_cv_o)/die_cv_e*100),
                pcb_mean_pct_dT=float(pct_pcb.mean()),
                pcb_min_pct_dT=float(pct_pcb.min()),
                pcb_max_pct_dT=float(pct_pcb.max()),
                die_mean_pct_dT=float(pct_die.mean()),
                die_min_pct_dT=float(pct_die.min()),
                die_max_pct_dT=float(pct_die.max()),
                exp_composite=float(exp_report.composite_score),
                opt_composite=float(opt_report.composite_score),
                exp_risk=exp_report.overall_risk.value,
                opt_risk=opt_report.overall_risk.value,
                pareto_solutions=len(pareto) if pareto else 0,
            )

            # ── Store results ─────────────────────────────────────────────────
            st.session_state.results = dict(
                exp_profile=exp_profile,
                opt_profile=opt_profile,
                metrics=metrics,
                figures=dict(
                    fig6a=fig6a, fig6b=fig6b,
                    fig6c=fig6c, fig6d=fig6d,
                    comparison=fig_cmp,
                    pareto=fig_pareto,
                ),
            )
            st.session_state.ran = True

            progress.progress(1.0, "✅ Done!")
            _log("✅ Pipeline complete! Go to Step 3 — Results tab.")
            st.success("✅ Pipeline complete! Switch to the **📊 Step 3 — Results** tab.")

        except Exception as e:
            import traceback
            progress.progress(0.0, "❌ Failed")
            st.error(f"Pipeline error: {e}")
            st.code(traceback.format_exc())