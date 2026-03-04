"""
utils/pipeline.py
Core computation pipeline — called by the Streamlit UI.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import io, warnings, tempfile, os, time
from pathlib import Path
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

# ── Feature columns (must match notebook exactly) ─────────────────────────────
FEATURE_COLS = [
    'peak_temp_C', 'tal_s', 'soak_temp_C', 'soak_time_s',
    'ramp_rate_Cps', 'cooling_rate_Cps', 't_total_s', 'T_amb_C',
    'copper_area_fraction', 'paste_coverage_fraction',
    'k_die_WmK', 'k_pcb_WmK',
]

# Default training data (C1–C10 from your TRAINING_DATA.xlsx)
DEFAULT_TRAINING = np.array([
    [245,47,165, 90,1.5,3.0,291,25,0.20,0.010,130,0.30],
    [240,43,162.5,120,1.0,2.5,376,25,0.15,0.015,130,0.30],
    [245,34,167.5, 60,2.0,3.5,213,25,0.30,0.018,130,0.30],
    [245,46,170, 120,1.7,3.0,300,25,0.30,0.020,130,0.30],
    [250,44,165,  80,1.5,3.2,271,25,0.40,0.020,130,0.30],
    [238,44,162.5, 90,1.6,3.0,287,25,0.20,0.024,130,0.30],
    [235,44,155,  90,1.2,3.0,316,25,0.20,0.022,130,0.30],
    [245,55,165,  90,1.4,2.0,326,25,0.25,0.026,130,0.30],
    [245,51,165, 110,1.0,2.8,364,25,0.26,0.028,130,0.30],
    [242,32,162.5, 70,1.8,3.4,231,25,0.30,0.030,130,0.30],
], dtype=np.float32)


# =============================================================================
# Thermal model
# =============================================================================

def _build_model():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class ThermalDigitalTwin(nn.Module):
            def __init__(self, in_features=12, dropout=0.1):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(in_features, 256), nn.LayerNorm(256), nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 512), nn.LayerNorm(512), nn.GELU(),
                )
                self.project = nn.Sequential(nn.Linear(512, 64*8*8), nn.GELU())
                def _dec():
                    return nn.Sequential(
                        nn.ConvTranspose2d(64,32,4,stride=2,padding=1),
                        nn.InstanceNorm2d(32), nn.GELU(),
                        nn.ConvTranspose2d(32,16,4,stride=2,padding=1),
                        nn.InstanceNorm2d(16), nn.GELU(),
                        nn.ConvTranspose2d(16, 1,4,stride=2,padding=1),
                    )
                self.pcb_decoder = _dec()
                self.die_decoder = _dec()
            def forward(self, x):
                z   = self.project(self.encoder(x)).view(-1,64,8,8)
                pcb = F.interpolate(self.pcb_decoder(z),(50,50),mode='bilinear',align_corners=False)
                die = F.interpolate(self.die_decoder(z),(50,50),mode='bilinear',align_corners=False)
                return pcb, die
        return ThermalDigitalTwin
    except ImportError:
        return None


def load_model(model_path, scaler, norm_params):
    """Load .pth model. Returns (model, device) or (None, None)."""
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ModelClass = _build_model()
    if ModelClass is None or not os.path.exists(model_path):
        return None, None
    try:
        try:
            torch.serialization.add_safe_globals([StandardScaler])
            ckpt = torch.load(model_path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(model_path, map_location=device, weights_only=False)

        in_features = ckpt.get('in_features', 12)
        model = ModelClass(in_features).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        # Update norm params from checkpoint
        if 'norm_params' in ckpt:
            np_ = ckpt['norm_params']
            norm_params['pcb_mean'] = float(np_.get('pcb_mean', norm_params['pcb_mean']))
            norm_params['pcb_std']  = float(np_.get('pcb_std',  norm_params['pcb_std']))
            dm = np_.get('die_means', None)
            ds = np_.get('die_stds',  None)
            if dm is not None:
                norm_params['die_mean'] = float(np.mean(dm))
                norm_params['die_std']  = float(np.mean(ds))
        if 'feature_scaler' in ckpt:
            scaler.__dict__.update(ckpt['feature_scaler'].__dict__)

        return model, device
    except Exception as e:
        print(f"[load_model] Error: {e}")
        return None, None


# =============================================================================
# Physics-based demo prediction
# =============================================================================

def physics_predict(profile_dict, exp_profile, pcb_exp_ref=None, die_exp_ref=None):
    """
    Scale from experimental reference maps using parameter ratios.
    If reference maps not available, build from scratch.
    """
    np.random.seed(int(profile_dict['peak_temp_C'] * 100) % 10000)
    N = 50
    cx, cy = N//2, N//2
    x, y   = np.meshgrid(np.arange(N), np.arange(N))

    pk   = profile_dict['peak_temp_C']
    cool = profile_dict['cooling_rate_Cps']
    ramp = profile_dict['ramp_rate_Cps']
    ratio = pk / exp_profile['peak_temp_C']

    # PCB map
    sigma  = 12 + (3.5 - cool) * 1.5
    gauss  = np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2))
    pkg    = (np.abs(x-cx) <= 16) & (np.abs(y-cy) <= 16)
    shadow = np.where(pkg, 0.5, 0.0)
    pcb    = (pk - 13.0*(gauss + shadow*0.3) + (cool - 3.0)*0.6*gauss
              + np.random.normal(0, 0.2, (N,N)))

    # DIE map
    ds     = 8 + (2.0 - ramp) * 2.0
    dg     = np.exp(-((x-cx)**2 + (y-cy)**2) / (2*ds**2))
    die    = ((pk - 14.0) + 0.04*dg + np.random.normal(0, 0.004, (N,N)))

    return pcb.astype(np.float64), die.astype(np.float64)


# =============================================================================
# Model predict
# =============================================================================

def model_predict(profile_dict, model, device, scaler, norm_params):
    import torch
    x_raw    = np.array([[profile_dict[k] for k in FEATURE_COLS]], dtype=np.float32)
    x_scaled = scaler.transform(x_raw)
    x_t      = torch.tensor(x_scaled, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pcb_p, die_p = model(x_t)
    pcb = pcb_p.cpu().numpy()[0,0] * norm_params['pcb_std'] + norm_params['pcb_mean']
    die = die_p.cpu().numpy()[0,0] * norm_params['die_std'] + norm_params['die_mean']
    return pcb.astype(np.float64), die.astype(np.float64)


# =============================================================================
# NSGA-II Optimizer
# =============================================================================

def run_nsga2(exp_profile, opt_settings, predictor_fn, scaler):
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize as pymoo_minimize

    peak_ub = opt_settings['peak_ub']
    pop     = opt_settings['nsga_pop']
    gens    = opt_settings['nsga_gens']

    BOUNDS = np.array([
        [235.0, min(peak_ub, exp_profile['peak_temp_C'] - 0.1)],
        [30.0,  60.0],
        [150.0, 180.0],
        [60.0,  120.0],
        [1.0,   3.0],
        [1.5,   4.0],
        [200.0, 400.0],
        [20.0,  30.0],
        [0.15,  0.35],
        [0.01,  0.03],
        [130.0, 148.0],
        [0.30,  0.50],
    ])

    lb_s = scaler.transform(BOUNDS[:,0].reshape(1,-1))[0]
    ub_s = scaler.transform(BOUNDS[:,1].reshape(1,-1))[0]

    class _Problem(Problem):
        def __init__(self):
            super().__init__(
                n_var=len(FEATURE_COLS), n_obj=2, n_constr=0,
                xl=np.minimum(lb_s,ub_s), xu=np.maximum(lb_s,ub_s)
            )
        def _evaluate(self, X, out, *args, **kwargs):
            F = []
            for xv in X:
                real = scaler.inverse_transform(xv.reshape(1,-1))[0]
                pd_  = {FEATURE_COLS[i]: float(real[i]) for i in range(len(FEATURE_COLS))}
                pcb, die = predictor_fn(pd_)
                o1 = (pcb.max()-pcb.min()) + 0.5*np.mean(np.abs(np.gradient(pcb)))
                o2 = (die.max()-die.min()) + 0.5*np.mean(np.abs(np.gradient(die)))
                F.append([o1, o2])
            out['F'] = np.array(F)

    result = pymoo_minimize(
        _Problem(),
        NSGA2(pop_size=pop,
              crossover=SBX(prob=0.9, eta=20),
              mutation=PM(prob=1.0/len(FEATURE_COLS), eta=20),
              eliminate_duplicates=True),
        ('n_gen', gens), seed=42, verbose=False
    )
    pareto_F = result.F
    combined = pareto_F[:,0] + pareto_F[:,1]
    best_idx = np.argmin(combined)
    best_real = scaler.inverse_transform(result.X[best_idx].reshape(1,-1))[0]
    opt_profile = {FEATURE_COLS[i]: float(best_real[i]) for i in range(len(FEATURE_COLS))}

    pareto_info = {
        'n_solutions': len(pareto_F),
        'pcb_obj_range': [float(pareto_F[:,0].min()), float(pareto_F[:,0].max())],
        'die_obj_range': [float(pareto_F[:,1].min()), float(pareto_F[:,1].max())],
        'pareto_F': pareto_F,
        'pareto_X': result.X,
    }
    return opt_profile, pareto_info


# =============================================================================
# Fig 6 style contour plot → bytes
# =============================================================================

VIRIDIS = plt.cm.viridis

def _coords(L_mm, n=50):
    return np.linspace(-L_mm/2, L_mm/2, n)

def make_fig6(tmap, L_mm, title, n_contours=22):
    """Returns PNG bytes of a Fig 6 style contour plot."""
    x = _coords(L_mm); z = _coords(L_mm)
    X, Z = np.meshgrid(x, z)
    lvls = np.linspace(tmap.min(), tmap.max(), n_contours+1)

    fig, ax = plt.subplots(figsize=(6, 5.2))
    cf = ax.contourf(X, Z, tmap, levels=lvls, cmap='viridis')
    ax.contour(X, Z, tmap, levels=lvls[::4], colors='k', linewidths=0.4, alpha=0.45)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("Temperature (°C)", fontsize=10)
    ax.set_xlabel("X (mm)", fontsize=10); ax.set_ylabel("Z (mm)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_aspect('equal')
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, dpi=200, bbox_inches='tight', format='png')
    plt.close(fig); buf.seek(0)
    return buf.read()


def make_comparison_fig(pcb_exp, die_exp, pcb_opt, die_opt,
                        exp_profile, opt_profile,
                        pct_pcb, pct_die,
                        l_pcb, l_die, pareto_info=None):
    """Full 3×4 comparison figure → PNG bytes."""
    plt.rcParams.update({"font.family": "serif", "figure.dpi": 120})
    fig = plt.figure(figsize=(22, 17))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.40)
    N   = 22

    xp, zp = _coords(l_pcb), _coords(l_pcb)
    xd, zd = _coords(l_die), _coords(l_die)
    XP, ZP = np.meshgrid(xp, zp)
    XD, ZD = np.meshgrid(xd, zd)

    pcb_vmin = min(pcb_exp.min(), pcb_opt.min())
    pcb_vmax = max(pcb_exp.max(), pcb_opt.max())
    die_vmin = min(die_exp.min(), die_opt.min())
    die_vmax = max(die_exp.max(), die_opt.max())
    pcb_lvls = np.linspace(pcb_vmin, pcb_vmax, N+1)
    die_lvls = np.linspace(die_vmin, die_vmax, N+1)

    def _cmap_ax(ax, X, Z, data, lvls, cbar_label, title):
        cf = ax.contourf(X, Z, data, levels=lvls, cmap='viridis')
        ax.contour(X, Z, data, levels=lvls[::4], colors='k', lw=0.35, alpha=0.45)
        fig.colorbar(cf, ax=ax, label=cbar_label, pad=0.02)
        ax.set_title(title, fontsize=9.5, fontweight='bold')
        ax.set_xlabel("X (mm)", fontsize=9); ax.set_ylabel("Z (mm)", fontsize=9)
        ax.set_aspect('equal')

    # Row 0: PCB
    _cmap_ax(fig.add_subplot(gs[0,0]), XP, ZP, pcb_exp, pcb_lvls, "T (°C)",
             f"PCB — Experimental\npeak={exp_profile['peak_temp_C']:.0f}°C  "
             f"cool={exp_profile['cooling_rate_Cps']:.1f}°C/s")
    _cmap_ax(fig.add_subplot(gs[0,1]), XP, ZP, pcb_opt, pcb_lvls, "T (°C)",
             f"PCB — Optimized\npeak={opt_profile['peak_temp_C']:.1f}°C  "
             f"cool={opt_profile['cooling_rate_Cps']:.2f}°C/s")

    ax02 = fig.add_subplot(gs[0,2])
    lim = max(abs(pct_pcb.min()), abs(pct_pcb.max()))
    cf02 = ax02.contourf(XP, ZP, pct_pcb, levels=np.linspace(-lim,lim,N+1), cmap='RdBu_r')
    ax02.contour(XP, ZP, pct_pcb, levels=np.linspace(-lim,lim,N+1)[::4],
                 colors='k', lw=0.3, alpha=0.4)
    fig.colorbar(cf02, ax=ax02, label="% ΔT", pad=0.02)
    ax02.set_title(f"PCB — % ΔT  (Opt − Exp)\nmean={pct_pcb.mean():.3f}%",
                   fontsize=9.5, fontweight='bold')
    ax02.set_xlabel("X (mm)", fontsize=9); ax02.set_ylabel("Z (mm)", fontsize=9)
    ax02.set_aspect('equal')

    # T-t curve
    ax03 = fig.add_subplot(gs[0,3])
    for prof, lbl, col in [(exp_profile, f"Exp ({exp_profile['peak_temp_C']:.0f}°C)", "#1565c0"),
                           (opt_profile, f"Opt ({opt_profile['peak_temp_C']:.1f}°C)", "#c62828")]:
        t, T = _build_tt(prof)
        ax03.plot(t, T, lw=2, color=col, label=lbl)
    ax03.axhline(217, ls='--', lw=1, color='orange', label='Liquidus 217°C')
    ax03.set_xlabel("Time (s)", fontsize=9); ax03.set_ylabel("Temp (°C)", fontsize=9)
    ax03.set_title("Reflow T-t Profile\nExperimental vs Optimized", fontsize=9.5, fontweight='bold')
    ax03.legend(fontsize=8); ax03.set_ylim(0, 290); ax03.grid(alpha=0.3)

    # Row 1: DIE
    _cmap_ax(fig.add_subplot(gs[1,0]), XD, ZD, die_exp, die_lvls, "T (°C)",
             f"DIE — Experimental\npeak={exp_profile['peak_temp_C']:.0f}°C")
    _cmap_ax(fig.add_subplot(gs[1,1]), XD, ZD, die_opt, die_lvls, "T (°C)",
             f"DIE — Optimized\npeak={opt_profile['peak_temp_C']:.1f}°C")

    ax12 = fig.add_subplot(gs[1,2])
    lim2 = max(abs(pct_die.min()), abs(pct_die.max()))
    cf12 = ax12.contourf(XD, ZD, pct_die, levels=np.linspace(-lim2,lim2,N+1), cmap='RdBu_r')
    ax12.contour(XD, ZD, pct_die, levels=np.linspace(-lim2,lim2,N+1)[::4],
                 colors='k', lw=0.3, alpha=0.4)
    fig.colorbar(cf12, ax=ax12, label="% ΔT", pad=0.02)
    ax12.set_title(f"DIE — % ΔT  (Opt − Exp)\nmean={pct_die.mean():.4f}%",
                   fontsize=9.5, fontweight='bold')
    ax12.set_xlabel("X (mm)", fontsize=9); ax12.set_ylabel("Z (mm)", fontsize=9)
    ax12.set_aspect('equal')

    # Metrics table
    ax13 = fig.add_subplot(gs[1,3]); ax13.axis('off')
    pcb_cv_e = pcb_exp.std()/pcb_exp.mean()
    pcb_cv_o = pcb_opt.std()/pcb_opt.mean()
    die_cv_e = die_exp.std()/die_exp.mean()
    die_cv_o = die_opt.std()/die_opt.mean()
    rows = [
        ["Peak Temp (°C)", f"{exp_profile['peak_temp_C']:.1f}", f"{opt_profile['peak_temp_C']:.2f}",
         f"{opt_profile['peak_temp_C']-exp_profile['peak_temp_C']:+.1f}"],
        ["TAL (s)", f"{exp_profile['tal_s']:.1f}", f"{opt_profile['tal_s']:.1f}",
         f"{opt_profile['tal_s']-exp_profile['tal_s']:+.1f}"],
        ["Cooling (°C/s)", f"{exp_profile['cooling_rate_Cps']:.2f}", f"{opt_profile['cooling_rate_Cps']:.2f}",
         f"{opt_profile['cooling_rate_Cps']-exp_profile['cooling_rate_Cps']:+.2f}"],
        ["PCB ΔT (°C)",
         f"{pcb_exp.max()-pcb_exp.min():.3f}", f"{pcb_opt.max()-pcb_opt.min():.3f}",
         f"{(pcb_opt.max()-pcb_opt.min())-(pcb_exp.max()-pcb_exp.min()):+.3f}"],
        ["DIE ΔT (°C)",
         f"{die_exp.max()-die_exp.min():.5f}", f"{die_opt.max()-die_opt.min():.5f}",
         f"{(die_opt.max()-die_opt.min())-(die_exp.max()-die_exp.min()):+.5f}"],
        ["PCB CV", f"{pcb_cv_e:.5f}", f"{pcb_cv_o:.5f}",
         f"{(pcb_cv_e-pcb_cv_o)/pcb_cv_e*100:+.1f}%"],
        ["DIE CV", f"{die_cv_e:.6f}", f"{die_cv_o:.6f}",
         f"{(die_cv_e-die_cv_o)/die_cv_e*100:+.1f}%"],
        ["PCB mean % ΔT", "—", "—", f"{pct_pcb.mean():.3f}%"],
        ["DIE mean % ΔT", "—", "—", f"{pct_die.mean():.4f}%"],
    ]
    tbl = ax13.table(cellText=rows, colLabels=["Metric","Experimental","Optimized","Δ"],
                     loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.05, 1.65)
    for j in range(4):
        tbl[0,j].set_facecolor('#1565c0')
        tbl[0,j].get_text().set_color('white')
        tbl[0,j].get_text().set_fontweight('bold')
    ax13.set_title("Metrics Summary", fontweight='bold', fontsize=10, pad=8)

    # Row 2: histograms
    ax20 = fig.add_subplot(gs[2,:2])
    ax20.hist(pct_pcb.ravel(), bins=60, color='#1565c0', edgecolor='k',
              linewidth=0.3, alpha=0.8)
    ax20.axvline(pct_pcb.mean(), ls='--', lw=1.8, color='#c62828',
                 label=f"Mean = {pct_pcb.mean():.3f}%")
    ax20.axvline(0, ls='-', lw=1, color='k', alpha=0.4)
    ax20.set_xlabel("% ΔT  =  (T_opt − T_exp) / T_exp × 100", fontsize=10)
    ax20.set_ylabel("Pixel count", fontsize=10)
    ax20.set_title("PCB — % Temperature Change Distribution (2D Surface)", fontsize=10, fontweight='bold')
    ax20.legend(fontsize=9); ax20.grid(alpha=0.3)

    ax21 = fig.add_subplot(gs[2,2:])
    ax21.hist(pct_die.ravel(), bins=60, color='#00897b', edgecolor='k',
              linewidth=0.3, alpha=0.8)
    ax21.axvline(pct_die.mean(), ls='--', lw=1.8, color='#c62828',
                 label=f"Mean = {pct_die.mean():.4f}%")
    ax21.axvline(0, ls='-', lw=1, color='k', alpha=0.4)
    ax21.set_xlabel("% ΔT  =  (T_opt − T_exp) / T_exp × 100", fontsize=10)
    ax21.set_ylabel("Pixel count", fontsize=10)
    ax21.set_title("DIE — % Temperature Change Distribution (2D Surface)", fontsize=10, fontweight='bold')
    ax21.legend(fontsize=9); ax21.grid(alpha=0.3)

    fig.suptitle(
        "PCB Reflow Thermal Digital Twin — Experimental vs Optimized Profile\n"
        f"Peak: {exp_profile['peak_temp_C']:.0f}°C → {opt_profile['peak_temp_C']:.1f}°C  "
        f"(Δ = {opt_profile['peak_temp_C']-exp_profile['peak_temp_C']:+.1f}°C)",
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, dpi=200, bbox_inches='tight', format='png')
    plt.close(fig); buf.seek(0)
    return buf.read()


def _build_tt(p):
    t, T = [0], [p['T_amb_C']]
    ph = (p['soak_temp_C']-p['T_amb_C'])/max(p['ramp_rate_Cps'],0.01)
    t.append(t[-1]+ph); T.append(p['soak_temp_C'])
    t.append(t[-1]+p['soak_time_s']); T.append(p['soak_temp_C'])
    rt = (p['peak_temp_C']-p['soak_temp_C'])/max(p['ramp_rate_Cps'],0.01)
    t.append(t[-1]+rt); T.append(p['peak_temp_C'])
    t.append(t[-1]+p['tal_s']); T.append(p['peak_temp_C'])
    ct = (p['peak_temp_C']-p['T_amb_C'])/max(p['cooling_rate_Cps'],0.01)
    t.append(t[-1]+ct); T.append(p['T_amb_C'])
    return np.array(t), np.array(T)


def make_pareto_fig(pareto_info, exp_pcb_obj, exp_die_obj, opt_profile):
    """Pareto front figure → PNG bytes."""
    if pareto_info is None:
        return None
    F = pareto_info['pareto_F']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    sc = ax.scatter(F[:,0], F[:,1], c=F[:,0]+F[:,1], cmap='plasma_r',
                    s=60, edgecolors='k', lw=0.4, zorder=4)
    ax.scatter([exp_pcb_obj], [exp_die_obj], c='red', s=150, marker='*',
               zorder=8, label='Experimental baseline')
    plt.colorbar(sc, ax=ax, label='Combined objective')
    ax.set_xlabel("PCB Thermal Non-Uniformity Score", fontsize=10)
    ax.set_ylabel("DIE Thermal Non-Uniformity Score", fontsize=10)
    ax.set_title("NSGA-II Pareto Front\nPCB vs DIE Thermal Uniformity", fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax2 = axes[1]
    from sklearn.preprocessing import StandardScaler as _SS
    _sc = _SS().fit(DEFAULT_TRAINING)
    peaks = _sc.inverse_transform(pareto_info['pareto_X'])[:,0]
    tals  = _sc.inverse_transform(pareto_info['pareto_X'])[:,1]
    sc2 = ax2.scatter(tals, peaks, c=F[:,0], cmap='viridis', s=60, edgecolors='k', lw=0.4)
    plt.colorbar(sc2, ax=ax2, label='PCB Uniformity Score')
    ax2.axhline(235, ls='--', c='green', lw=1, label='SAC305 T_min=235°C')
    ax2.axhline(260, ls='--', c='red',   lw=1, label='SAC305 T_max=260°C')
    ax2.axvline(30,  ls=':',  c='blue',  lw=1, label='TAL min=30s')
    ax2.axvline(60,  ls=':',  c='orange',lw=1, label='TAL max=60s')
    ax2.legend(fontsize=7)
    ax2.set_xlabel("Time Above Liquidus (s)", fontsize=10)
    ax2.set_ylabel("Peak Temperature (°C)", fontsize=10)
    ax2.set_title("Pareto Profile Parameter Space", fontweight='bold')

    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, dpi=200, bbox_inches='tight', format='png')
    plt.close(fig); buf.seek(0)
    return buf.read()


# =============================================================================
# Main pipeline entry point
# =============================================================================

def run_full_pipeline(exp_profile, opt_settings, geometry,
                      model_path=None, pcb_exp_arr=None, die_exp_arr=None,
                      train_xls_bytes=None, progress_cb=None):
    """
    Full pipeline. Returns results dict with all images as bytes and metrics.
    """
    prog = [0]
    def _prog(v, msg=""):
        prog[0] = v
        if progress_cb:
            progress_cb(v, msg)

    _prog(0.05, "Setting up normalization...")

    # ── Scaler ────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    if train_xls_bytes:
        import io as _io
        raw = pd.read_excel(_io.BytesIO(train_xls_bytes), header=None)
        hr  = raw[raw.eq('peak_temp_C').any(axis=1)].index[0]
        df  = raw.iloc[hr+1:].copy()
        df.columns = raw.iloc[hr]
        df  = df.reset_index(drop=True).dropna(how='all')
        for col in df.columns:
            try: df[col] = pd.to_numeric(df[col])
            except: pass
        X_raw = df[FEATURE_COLS].values.astype(np.float32)
    else:
        X_raw = DEFAULT_TRAINING

    scaler.fit(X_raw)

    # Norm params
    norm_params = {
        'pcb_mean': float(np.mean([pcb_exp_arr.mean() if pcb_exp_arr is not None else 242.0])),
        'pcb_std':  float(np.mean([pcb_exp_arr.std()  if pcb_exp_arr is not None else 4.5])),
        'die_mean': float(np.mean([die_exp_arr.mean() if die_exp_arr is not None else 231.32])),
        'die_std':  float(np.mean([die_exp_arr.std()  if die_exp_arr is not None else 0.009])),
    }

    _prog(0.10, "Loading model...")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, device = None, None
    if model_path:
        model, device = load_model(model_path, scaler, norm_params)

    model_loaded = model is not None

    def _predict(profile_dict):
        if model_loaded:
            return model_predict(profile_dict, model, device, scaler, norm_params)
        else:
            return physics_predict(profile_dict, exp_profile, pcb_exp_arr, die_exp_arr)

    _prog(0.15, "Predicting experimental thermal maps...")

    # ── Experimental maps ─────────────────────────────────────────────────────
    if pcb_exp_arr is not None and die_exp_arr is not None:
        pcb_exp = pcb_exp_arr.astype(np.float64)
        die_exp = die_exp_arr.astype(np.float64)
        used_fea = True
    else:
        pcb_exp, die_exp = _predict(exp_profile)
        used_fea = False

    _prog(0.25, "Running NSGA-II optimization (this takes ~1–2 min)...")

    # ── Optimize ──────────────────────────────────────────────────────────────
    try:
        opt_profile, pareto_info = run_nsga2(exp_profile, opt_settings, _predict, scaler)
    except Exception as e:
        print(f"[NSGA-II] Failed: {e}. Falling back to simple DE.")
        opt_profile  = _de_fallback(exp_profile, opt_settings, _predict, scaler)
        pareto_info  = None

    _prog(0.70, "Predicting optimized thermal maps...")

    # ── Optimized maps ────────────────────────────────────────────────────────
    pcb_opt, die_opt = _predict(opt_profile)

    _prog(0.80, "Computing % ΔT...")

    # ── % ΔT ──────────────────────────────────────────────────────────────────
    pct_pcb = (pcb_opt - pcb_exp) / np.abs(pcb_exp) * 100.0
    pct_die = (die_opt - die_exp) / np.abs(die_exp) * 100.0

    l_pcb = geometry['l_pcb']
    l_die = geometry['l_die']

    _prog(0.85, "Generating figures...")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig6a = make_fig6(pcb_exp, l_pcb,
        f"Experimental PCB Temperature Contour\n"
        f"peak={exp_profile['peak_temp_C']:.0f}°C  "
        f"cool={exp_profile['cooling_rate_Cps']:.1f}°C/s  "
        f"TAL={exp_profile['tal_s']:.0f}s")
    fig6b = make_fig6(die_exp, l_die,
        f"Experimental DIE Temperature Contour\n"
        f"peak={exp_profile['peak_temp_C']:.0f}°C  "
        f"cool={exp_profile['cooling_rate_Cps']:.1f}°C/s")
    fig6c = make_fig6(pcb_opt, l_pcb,
        f"Optimized PCB Temperature Contour\n"
        f"peak={opt_profile['peak_temp_C']:.1f}°C  "
        f"cool={opt_profile['cooling_rate_Cps']:.2f}°C/s  "
        f"TAL={opt_profile['tal_s']:.1f}s")
    fig6d = make_fig6(die_opt, l_die,
        f"Optimized DIE Temperature Contour\n"
        f"peak={opt_profile['peak_temp_C']:.1f}°C  "
        f"cool={opt_profile['cooling_rate_Cps']:.2f}°C/s")

    # Compute exp objectives for Pareto plot
    exp_pcb_obj = (pcb_exp.max()-pcb_exp.min()) + 0.5*np.mean(np.abs(np.gradient(pcb_exp)))
    exp_die_obj = (die_exp.max()-die_exp.min()) + 0.5*np.mean(np.abs(np.gradient(die_exp)))

    fig_cmp = make_comparison_fig(pcb_exp, die_exp, pcb_opt, die_opt,
                                  exp_profile, opt_profile,
                                  pct_pcb, pct_die, l_pcb, l_die, pareto_info)
    fig_pareto = make_pareto_fig(pareto_info, exp_pcb_obj, exp_die_obj, opt_profile)

    _prog(0.98, "Done.")

    # ── Metrics ───────────────────────────────────────────────────────────────
    pcb_cv_e = float(pcb_exp.std()/pcb_exp.mean())
    pcb_cv_o = float(pcb_opt.std()/pcb_opt.mean())
    die_cv_e = float(die_exp.std()/die_exp.mean())
    die_cv_o = float(die_opt.std()/die_opt.mean())

    metrics = {
        'pcb_range_exp':    float(pcb_exp.max()-pcb_exp.min()),
        'pcb_range_opt':    float(pcb_opt.max()-pcb_opt.min()),
        'die_range_exp':    float(die_exp.max()-die_exp.min()),
        'die_range_opt':    float(die_opt.max()-die_opt.min()),
        'pcb_cv_exp':       pcb_cv_e,
        'pcb_cv_opt':       pcb_cv_o,
        'die_cv_exp':       die_cv_e,
        'die_cv_opt':       die_cv_o,
        'pcb_mean_pct_dT':  float(pct_pcb.mean()),
        'pcb_min_pct_dT':   float(pct_pcb.min()),
        'pcb_max_pct_dT':   float(pct_pcb.max()),
        'die_mean_pct_dT':  float(pct_die.mean()),
        'die_min_pct_dT':   float(pct_die.min()),
        'die_max_pct_dT':   float(pct_die.max()),
        'pcb_cv_improvement_pct': (pcb_cv_e - pcb_cv_o)/pcb_cv_e*100,
        'die_cv_improvement_pct': (die_cv_e - die_cv_o)/die_cv_e*100,
    }

    return {
        'exp_profile':   exp_profile,
        'opt_profile':   opt_profile,
        'pareto_info':   pareto_info,
        'metrics':       metrics,
        'model_loaded':  model_loaded,
        'used_fea_maps': used_fea,
        'figures': {
            'fig6a_exp_pcb':   fig6a,
            'fig6b_exp_die':   fig6b,
            'fig6c_opt_pcb':   fig6c,
            'fig6d_opt_die':   fig6d,
            'comparison':      fig_cmp,
            'pareto':          fig_pareto,
        },
    }


def _de_fallback(exp_profile, opt_settings, predictor_fn, scaler):
    """Simple differential evolution fallback when pymoo not available."""
    np.random.seed(42)
    BOUNDS = np.array([
        [235.0, min(opt_settings['peak_ub'], exp_profile['peak_temp_C']-0.1)],
        [30.0, 60.0],[150.0,180.0],[60.0,120.0],[1.0,3.0],[1.5,4.0],
        [200.0,400.0],[20.0,30.0],[0.15,0.35],[0.01,0.03],[130.0,148.0],[0.3,0.5],
    ])
    def _fit(x):
        pd_ = {FEATURE_COLS[i]: float(np.clip(x[i],BOUNDS[i,0],BOUNDS[i,1]))
               for i in range(len(FEATURE_COLS))}
        pcb, die = predictor_fn(pd_)
        return (pcb.max()-pcb.min()) + 0.5*np.mean(np.abs(np.gradient(pcb))) + \
               (die.max()-die.min()) + 0.5*np.mean(np.abs(np.gradient(die)))

    pop = np.random.uniform(BOUNDS[:,0], BOUNDS[:,1], (30, len(FEATURE_COLS)))
    scores = np.array([_fit(x) for x in pop])
    for _ in range(40):
        for i in range(len(pop)):
            a,b,c = np.random.choice([j for j in range(len(pop)) if j!=i], 3, False)
            trial = np.clip(pop[a]+0.5*(pop[b]-pop[c]), BOUNDS[:,0], BOUNDS[:,1])
            trial[0] = np.clip(trial[0], BOUNDS[0,0], BOUNDS[0,1])
            ts = _fit(trial)
            if ts < scores[i]:
                pop[i] = trial; scores[i] = ts
    best = pop[np.argmin(scores)]
    return {FEATURE_COLS[i]: float(np.clip(best[i],BOUNDS[i,0],BOUNDS[i,1]))
            for i in range(len(FEATURE_COLS))}
