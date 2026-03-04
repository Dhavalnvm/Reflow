"""
=============================================================================
Stage 2 — Inference Engine  (UPDATED for Phase 1 PyTorch model)
PCB Reflow Thermal Digital Twin
=============================================================================
Loads the trained ThermalDigitalTwin (.pth) from Phase 1 and provides
fast thermal field predictions.

Usage:
    engine = InferenceEngine("models/thermal_digital_twin.pth")
    result = engine.predict(profile_params)
    print(result["pcb_map"].shape)   # (50, 50)
    print(result["die_map"].shape)   # (50, 50)
=============================================================================
"""

import numpy as np
import os
import warnings
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ReflowProfile:
    """
    Reflow profile + material parameters matching Phase 1 training features.
    Feature order MUST match FEATURE_COLS from the notebook exactly.
    """
    # ── Reflow profile ──────────────────────────────────────────────────────
    peak_temp_C:            float = 245.0   # Peak reflow temperature (°C)
    tal_s:                  float = 47.0    # Time above liquidus (s), SAC305: 30-60s
    soak_temp_C:            float = 165.0   # Soak zone temperature (°C)
    soak_time_s:            float = 90.0    # Soak duration (s)
    ramp_rate_Cps:          float = 1.5     # Heating ramp rate (°C/s)
    cooling_rate_Cps:       float = 3.0     # Cooling rate (°C/s), positive convention
    t_total_s:              float = 291.0   # Total profile duration (s)
    T_amb_C:                float = 25.0    # Ambient temperature (°C)

    # ── Material parameters ─────────────────────────────────────────────────
    copper_area_fraction:       float = 0.20    # PCB copper coverage fraction
    paste_coverage_fraction:    float = 0.01    # Solder paste coverage fraction
    k_die_WmK:                  float = 130.0   # Die thermal conductivity (W/m·K)
    k_pcb_WmK:                  float = 0.30    # PCB thermal conductivity (W/m·K)

    # ── FEATURE_COLS order (must match Phase 1 training exactly) ────────────
    FEATURE_COLS = [
        'peak_temp_C', 'tal_s', 'soak_temp_C', 'soak_time_s',
        'ramp_rate_Cps', 'cooling_rate_Cps', 't_total_s', 'T_amb_C',
        'copper_area_fraction', 'paste_coverage_fraction',
        'k_die_WmK', 'k_pcb_WmK',
    ]

    def to_array(self) -> np.ndarray:
        """Return feature vector in correct order for model input."""
        return np.array([getattr(self, k) for k in self.FEATURE_COLS], dtype=np.float32)

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.FEATURE_COLS}


@dataclass
class PredictionResult:
    """Full output from one inference call — all temperatures in real °C."""
    pcb_map:            np.ndarray      # (50, 50) in °C
    die_map:            np.ndarray      # (50, 50) in °C
    pcb_uniformity:     float           # CV = std/mean (lower = more uniform)
    die_uniformity:     float
    pcb_gradient_max:   float           # Max spatial gradient
    die_gradient_max:   float
    pcb_range:          float           # max - min (°C)
    die_range:          float
    inference_time_ms:  float


# ---------------------------------------------------------------------------
# ThermalDigitalTwin Architecture (must match Phase 1 exactly)
# ---------------------------------------------------------------------------

def _build_model(in_features: int = 12):
    """Reconstruct ThermalDigitalTwin architecture from Phase 1."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class ThermalDigitalTwin(nn.Module):
            def __init__(self, in_features=12, dropout=0.1):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(in_features, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                )
                self.project = nn.Sequential(
                    nn.Linear(512, 64 * 8 * 8),
                    nn.GELU(),
                )
                self.pcb_decoder = nn.Sequential(
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(32), nn.GELU(),
                    nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(16), nn.GELU(),
                    nn.ConvTranspose2d(16,  1, 4, stride=2, padding=1),
                )
                self.die_decoder = nn.Sequential(
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(32), nn.GELU(),
                    nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(16), nn.GELU(),
                    nn.ConvTranspose2d(16,  1, 4, stride=2, padding=1),
                )

            def forward(self, x):
                z   = self.encoder(x)
                z   = self.project(z)
                z   = z.view(-1, 64, 8, 8)
                pcb = F.interpolate(self.pcb_decoder(z), size=(50, 50),
                                    mode='bilinear', align_corners=False)
                die = F.interpolate(self.die_decoder(z), size=(50, 50),
                                    mode='bilinear', align_corners=False)
                return pcb, die

        return ThermalDigitalTwin(in_features=in_features)

    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Inference engine for the PCB Reflow Thermal Digital Twin.

    Loads the Phase 1 trained model (thermal_digital_twin.pth) and exposes
    a clean predict() API returning full-field thermal maps in real °C.
    """

    MAP_SHAPE = (50, 50)

    def __init__(self, model_path: str, verbose: bool = True):
        self.model_path = model_path
        self.verbose    = verbose
        self.model      = None
        self.scaler     = None
        self.norm_params = {}
        self.feature_cols = ReflowProfile.FEATURE_COLS
        self._loaded    = False
        self._device    = None

        if os.path.exists(model_path):
            self._load_model()
        else:
            print(f"[InferenceEngine] WARNING: Model not found at '{model_path}'")
            print("  → Running in DEMO MODE with synthetic predictions.")
            print("  → Save your Phase 1 model to that path to enable real inference.")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load Phase 1 PyTorch checkpoint (.pth)."""
        try:
            import torch
            t0 = time.time()

            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # PyTorch 2.6+ requires explicit allowlist for sklearn objects in checkpoint
            try:
                from sklearn.preprocessing import StandardScaler
                torch.serialization.add_safe_globals([StandardScaler])
                checkpoint = torch.load(self.model_path, map_location=self._device,
                                        weights_only=True)
            except Exception:
                # Fallback for older PyTorch or if above fails
                checkpoint = torch.load(self.model_path, map_location=self._device,
                                        weights_only=False)

            # Reconstruct model
            in_features = checkpoint.get('in_features', 12)
            self.model  = _build_model(in_features).to(self._device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Load scaler and normalization params
            self.scaler      = checkpoint.get('feature_scaler')
            self.norm_params = checkpoint.get('norm_params', {})
            self.feature_cols = checkpoint.get('feature_cols', ReflowProfile.FEATURE_COLS)

            self._loaded = True
            elapsed = (time.time() - t0) * 1000

            if self.verbose:
                print(f"[InferenceEngine] ✅ Model loaded in {elapsed:.1f} ms")
                print(f"  Path       : {self.model_path}")
                print(f"  Device     : {self._device}")
                print(f"  Features   : {in_features}")
                print(f"  PCB norm   : mean={self.norm_params.get('pcb_mean', 'N/A'):.2f} "
                      f"std={self.norm_params.get('pcb_std', 'N/A'):.3f}")
                print(f"  DIE norm   : mean={self.norm_params.get('die_mean', 'N/A'):.4f} "
                      f"std={self.norm_params.get('die_std', 'N/A'):.6f}")

        except Exception as e:
            print(f"[InferenceEngine] ERROR loading model: {e}")
            print("  → Falling back to DEMO MODE.")

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _denorm_pcb(self, x: np.ndarray) -> np.ndarray:
        m = self.norm_params.get('pcb_mean', 0)
        s = self.norm_params.get('pcb_std',  1)
        return x * s + m

    def _denorm_die(self, x: np.ndarray) -> np.ndarray:
        """Use global DIE mean/std (per-case norm not available at inference)."""
        # die_means and die_stds are lists — use their mean as global estimate
        die_means = self.norm_params.get('die_means', None)
        die_stds  = self.norm_params.get('die_stds',  None)
        if die_means is not None:
            m = float(np.mean(die_means))
            s = float(np.mean(die_stds))
        else:
            m = self.norm_params.get('die_mean', 0)
            s = self.norm_params.get('die_std',  1)
        return x * s + m

    # ------------------------------------------------------------------
    # Thermal Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _uniformity(arr: np.ndarray) -> float:
        """Coefficient of variation (std/mean). Lower = more uniform."""
        mean = np.mean(arr)
        return float(np.std(arr) / mean) if mean != 0 else 0.0

    @staticmethod
    def _max_gradient(arr: np.ndarray) -> float:
        gy, gx = np.gradient(arr)
        return float(np.max(np.sqrt(gx**2 + gy**2)))

    # ------------------------------------------------------------------
    # Demo (no model)
    # ------------------------------------------------------------------

    def _synthetic_predict(self, profile: ReflowProfile) -> Tuple[np.ndarray, np.ndarray]:
        """Physics-plausible synthetic maps for demo mode."""
        np.random.seed(42)
        size = self.MAP_SHAPE
        cx, cy = size[0] // 2, size[1] // 2
        x, y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))

        sigma = 15 + (profile.peak_temp_C - 245) * 0.2
        base  = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

        # PCB: broader field with die shadow
        pcb_map = profile.peak_temp_C - 13.0 * base + np.random.normal(0, 0.5, size)

        # DIE: nearly uniform with tiny variation
        die_map = (profile.peak_temp_C - 14.0) + 0.04 * base + np.random.normal(0, 0.005, size)

        return pcb_map, die_map

    # ------------------------------------------------------------------
    # Core Predict
    # ------------------------------------------------------------------

    def predict(self, profile: ReflowProfile) -> PredictionResult:
        """
        Run one inference pass.

        Parameters
        ----------
        profile : ReflowProfile dataclass with all 12 features

        Returns
        -------
        PredictionResult with full thermal maps in real °C and derived metrics.
        """
        t0 = time.time()

        if self._loaded and self.model is not None:
            try:
                import torch

                # Scale input features
                x_raw = profile.to_array().reshape(1, -1)
                if self.scaler is not None:
                    x_scaled = self.scaler.transform(x_raw).astype(np.float32)
                else:
                    x_scaled = x_raw

                x_t = torch.tensor(x_scaled, dtype=torch.float32).to(self._device)

                self.model.eval()
                with torch.no_grad():
                    pcb_pred, die_pred = self.model(x_t)

                # Denormalize to real °C
                pcb_map = self._denorm_pcb(pcb_pred.cpu().numpy()[0, 0])
                die_map = self._denorm_die(die_pred.cpu().numpy()[0, 0])

            except Exception as e:
                print(f"[InferenceEngine] Inference error: {e}. Using demo mode.")
                pcb_map, die_map = self._synthetic_predict(profile)
        else:
            pcb_map, die_map = self._synthetic_predict(profile)

        elapsed_ms = (time.time() - t0) * 1000

        return PredictionResult(
            pcb_map          = pcb_map,
            die_map          = die_map,
            pcb_uniformity   = self._uniformity(pcb_map),
            die_uniformity   = self._uniformity(die_map),
            pcb_gradient_max = self._max_gradient(pcb_map),
            die_gradient_max = self._max_gradient(die_map),
            pcb_range        = float(pcb_map.max() - pcb_map.min()),
            die_range        = float(die_map.max() - die_map.min()),
            inference_time_ms = elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Batch Predict
    # ------------------------------------------------------------------

    def predict_batch(self, profiles: list) -> list:
        """Predict for a list of ReflowProfile objects."""
        return [self.predict(p) for p in profiles]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self):
        print("=" * 55)
        print("  InferenceEngine — PCB Reflow Thermal Digital Twin")
        print("=" * 55)
        print(f"  Model path : {self.model_path}")
        print(f"  Loaded     : {self._loaded}")
        print(f"  Device     : {self._device}")
        print(f"  Map shape  : {self.MAP_SHAPE}")
        print(f"  Features   : {len(self.feature_cols)}")
        print(f"  Scaler     : {'active' if self.scaler else 'none'}")
        print("=" * 55)


# ---------------------------------------------------------------------------
# Quick Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running InferenceEngine self-test...\n")

    engine = InferenceEngine("models/thermal_digital_twin.pth")
    engine.summary()

    # Default SAC305 profile (matches Case 1 from training data)
    profile = ReflowProfile(
        peak_temp_C=245.0, tal_s=47.0, soak_temp_C=165.0, soak_time_s=90.0,
        ramp_rate_Cps=1.5, cooling_rate_Cps=3.0, t_total_s=291.0, T_amb_C=25.0,
        copper_area_fraction=0.20, paste_coverage_fraction=0.01,
        k_die_WmK=130.0, k_pcb_WmK=0.30,
    )

    result = engine.predict(profile)

    print(f"\nPrediction complete in {result.inference_time_ms:.2f} ms")
    print(f"  PCB map  : {result.pcb_map.shape}  |  {result.pcb_map.min():.2f} → {result.pcb_map.max():.2f} °C")
    print(f"  DIE map  : {result.die_map.shape}  |  {result.die_map.min():.4f} → {result.die_map.max():.4f} °C")
    print(f"  PCB range       : {result.pcb_range:.3f} °C")
    print(f"  DIE range       : {result.die_range:.5f} °C")
    print(f"  PCB uniformity  : {result.pcb_uniformity:.5f}")
    print(f"  DIE uniformity  : {result.die_uniformity:.6f}")
    print(f"  PCB max gradient: {result.pcb_gradient_max:.4f}")
    print(f"  DIE max gradient: {result.die_gradient_max:.6f}")