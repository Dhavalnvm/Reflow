"""
=============================================================================
Stage 2 — Intelligent Reflow Advisor  (UPDATED for Phase 1 model)
PCB Reflow Thermal Digital Twin
=============================================================================
Wraps InferenceEngine and adds:
  • SAC305 process rule validation
  • Physics-based risk flagging
  • Thermal uniformity assessment
  • Actionable improvement recommendations
  • Composite optimization score

Usage:
    advisor = ReflowAdvisor(engine)
    report  = advisor.evaluate(profile)
    report.print_summary()
=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum

from inference_engine import InferenceEngine, ReflowProfile, PredictionResult


# ---------------------------------------------------------------------------
# Risk Classification
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    OK       = "OK"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class RiskFlag:
    level:           RiskLevel
    category:        str
    parameter:       str
    observed_value:  float
    limit_value:     float
    message:         str
    recommendation:  str

    def __str__(self):
        icon = {"OK": "✅", "WARNING": "⚠️ ", "CRITICAL": "🔴"}[self.level.value]
        return (
            f"{icon} [{self.level.value:8s}] {self.category}\n"
            f"   {self.message}\n"
            f"   → {self.recommendation}"
        )


# ---------------------------------------------------------------------------
# SAC305 Process Limits
# ---------------------------------------------------------------------------

SAC305_LIMITS = {
    # Peak temperature (°C)
    "peak_temp_min":          235.0,
    "peak_temp_max":          260.0,
    "peak_temp_critical_max": 265.0,

    # Time above liquidus (s)  SAC305 liquidus = 217°C
    "tal_min_s":              30.0,
    "tal_max_s":              60.0,
    "tal_critical_max_s":     90.0,

    # Ramp rate (°C/s)
    "ramp_rate_max":          3.0,
    "ramp_rate_critical":     4.0,

    # Cooling rate (°C/s, positive convention)
    "cooling_rate_max":       4.0,    # SAC305 recommendation
    "cooling_rate_critical":  6.0,    # board/joint damage risk

    # Soak zone (°C)
    "soak_temp_min":          150.0,
    "soak_temp_max":          180.0,

    # Thermal uniformity (CV = std/mean) — thresholds for real °C maps (~230-250°C)
    "uniformity_warning":     0.020,   # CV > 2% for real °C PCB map
    "uniformity_critical":    0.040,   # CV > 4%

    # Spatial gradient (°C/pixel) — for real °C maps
    "gradient_warning":       3.0,
    "gradient_critical":      6.0,
}


# ---------------------------------------------------------------------------
# Advisor Report
# ---------------------------------------------------------------------------

@dataclass
class AdvisorReport:
    profile:           ReflowProfile
    prediction:        PredictionResult
    flags:             List[RiskFlag] = field(default_factory=list)
    overall_risk:      RiskLevel      = RiskLevel.OK
    uniformity_score:  float          = 0.0
    process_score:     float          = 0.0
    composite_score:   float          = 0.0
    suggestions:       List[str]      = field(default_factory=list)

    @property
    def n_critical(self) -> int:
        return sum(1 for f in self.flags if f.level == RiskLevel.CRITICAL)

    @property
    def n_warning(self) -> int:
        return sum(1 for f in self.flags if f.level == RiskLevel.WARNING)

    def print_summary(self):
        icons = {RiskLevel.OK: "✅", RiskLevel.WARNING: "⚠️ ", RiskLevel.CRITICAL: "🔴"}
        print("\n" + "=" * 62)
        print("  INTELLIGENT REFLOW ADVISOR REPORT")
        print("=" * 62)
        print(f"\n  Overall Risk    : {icons[self.overall_risk]} {self.overall_risk.value}")
        print(f"  Composite Score : {self.composite_score:.4f}  (lower = better)")
        print(f"  Process Score   : {self.process_score:.4f}")
        print(f"  Uniformity Score: {self.uniformity_score:.4f}")
        print(f"  Inference Time  : {self.prediction.inference_time_ms:.1f} ms")

        print("\n  ── Thermal Field Metrics (real °C) ────────────────────")
        print(f"  PCB range (°C)      : {self.prediction.pcb_range:.3f}")
        print(f"  DIE range (°C)      : {self.prediction.die_range:.5f}")
        print(f"  PCB uniformity (CV) : {self.prediction.pcb_uniformity:.5f}")
        print(f"  DIE uniformity (CV) : {self.prediction.die_uniformity:.6f}")
        print(f"  PCB peak temp (°C)  : {self.prediction.pcb_map.max():.2f}")
        print(f"  DIE peak temp (°C)  : {self.prediction.die_map.max():.4f}")

        print(f"\n  ── Risk Flags ({len(self.flags)} total: "
              f"{self.n_critical} critical, {self.n_warning} warnings) ──")
        if not self.flags:
            print("  No issues detected. ✅")
        else:
            for flag in self.flags:
                print(f"\n  {flag}")

        if self.suggestions:
            print("\n  ── Recommendations ────────────────────────────────────")
            for i, s in enumerate(self.suggestions, 1):
                print(f"  {i}. {s}")

        print("\n" + "=" * 62)


# ---------------------------------------------------------------------------
# Reflow Advisor
# ---------------------------------------------------------------------------

class ReflowAdvisor:
    """
    Intelligent reflow advisory system.

    1. Validates profile against SAC305 process limits
    2. Evaluates thermal field quality from model predictions
    3. Generates risk flags and actionable recommendations
    4. Computes composite score for optimization
    """

    def __init__(
        self,
        engine:             InferenceEngine,
        limits:             Dict = None,
        pcb_weight:         float = 0.5,
        die_weight:         float = 0.5,
        uniformity_weight:  float = 0.7,
        gradient_weight:    float = 0.3,
    ):
        self.engine           = engine
        self.limits           = limits or SAC305_LIMITS
        self.pcb_weight       = pcb_weight
        self.die_weight       = die_weight
        self.uniformity_weight = uniformity_weight
        self.gradient_weight  = gradient_weight

    # ------------------------------------------------------------------
    # Profile Validation
    # ------------------------------------------------------------------

    def _validate_profile(self, profile: ReflowProfile) -> List[RiskFlag]:
        flags = []
        L = self.limits

        # ── Peak temperature ─────────────────────────────────────────────
        pk = profile.peak_temp_C
        if pk > L["peak_temp_critical_max"]:
            flags.append(RiskFlag(
                level=RiskLevel.CRITICAL, category="Peak Temperature",
                parameter="peak_temp_C", observed_value=pk,
                limit_value=L["peak_temp_critical_max"],
                message=f"Peak temp {pk:.1f}°C exceeds critical limit {L['peak_temp_critical_max']:.0f}°C.",
                recommendation="Reduce peak temperature to ≤260°C to prevent component damage."
            ))
        elif pk > L["peak_temp_max"]:
            flags.append(RiskFlag(
                level=RiskLevel.WARNING, category="Peak Temperature",
                parameter="peak_temp_C", observed_value=pk,
                limit_value=L["peak_temp_max"],
                message=f"Peak temp {pk:.1f}°C above nominal SAC305 limit {L['peak_temp_max']:.0f}°C.",
                recommendation="Consider reducing peak temperature to 245–255°C."
            ))
        elif pk < L["peak_temp_min"]:
            flags.append(RiskFlag(
                level=RiskLevel.CRITICAL, category="Peak Temperature",
                parameter="peak_temp_C", observed_value=pk,
                limit_value=L["peak_temp_min"],
                message=f"Peak temp {pk:.1f}°C below SAC305 minimum {L['peak_temp_min']:.0f}°C.",
                recommendation="Increase peak temperature — insufficient for complete SAC305 reflow."
            ))
        else:
            flags.append(RiskFlag(
                level=RiskLevel.OK, category="Peak Temperature",
                parameter="peak_temp_C", observed_value=pk,
                limit_value=L["peak_temp_max"],
                message=f"Peak temp {pk:.1f}°C within SAC305 nominal range.",
                recommendation="No action required."
            ))

        # ── TAL ──────────────────────────────────────────────────────────
        tal = profile.tal_s
        if tal > L["tal_critical_max_s"]:
            flags.append(RiskFlag(
                level=RiskLevel.CRITICAL, category="Time Above Liquidus",
                parameter="tal_s", observed_value=tal,
                limit_value=L["tal_critical_max_s"],
                message=f"TAL {tal:.0f}s exceeds critical limit {L['tal_critical_max_s']:.0f}s.",
                recommendation="Reduce TAL — excessive IMC growth degrades joint reliability."
            ))
        elif tal > L["tal_max_s"]:
            flags.append(RiskFlag(
                level=RiskLevel.WARNING, category="Time Above Liquidus",
                parameter="tal_s", observed_value=tal,
                limit_value=L["tal_max_s"],
                message=f"TAL {tal:.0f}s exceeds recommended max {L['tal_max_s']:.0f}s.",
                recommendation="Reduce TAL to 30–60s for optimal SAC305 joint quality."
            ))
        elif tal < L["tal_min_s"]:
            flags.append(RiskFlag(
                level=RiskLevel.WARNING, category="Time Above Liquidus",
                parameter="tal_s", observed_value=tal,
                limit_value=L["tal_min_s"],
                message=f"TAL {tal:.0f}s may be insufficient for complete wetting.",
                recommendation="Extend TAL to ≥30s for complete solder joint formation."
            ))
        else:
            flags.append(RiskFlag(
                level=RiskLevel.OK, category="Time Above Liquidus",
                parameter="tal_s", observed_value=tal,
                limit_value=L["tal_max_s"],
                message=f"TAL {tal:.0f}s within SAC305 target window.",
                recommendation="No action required."
            ))

        # ── Ramp rate ────────────────────────────────────────────────────
        rr = profile.ramp_rate_Cps
        if rr > L["ramp_rate_critical"]:
            flags.append(RiskFlag(
                level=RiskLevel.CRITICAL, category="Ramp Rate",
                parameter="ramp_rate_Cps", observed_value=rr,
                limit_value=L["ramp_rate_critical"],
                message=f"Ramp rate {rr:.2f}°C/s exceeds critical limit {L['ramp_rate_critical']:.1f}°C/s.",
                recommendation="Reduce ramp rate — thermal shock risk to components."
            ))
        elif rr > L["ramp_rate_max"]:
            flags.append(RiskFlag(
                level=RiskLevel.WARNING, category="Ramp Rate",
                parameter="ramp_rate_Cps", observed_value=rr,
                limit_value=L["ramp_rate_max"],
                message=f"Ramp rate {rr:.2f}°C/s slightly above recommended {L['ramp_rate_max']:.1f}°C/s.",
                recommendation="Target ≤3°C/s ramp rate."
            ))

        # ── Cooling rate ─────────────────────────────────────────────────
        cr = profile.cooling_rate_Cps
        if cr > L["cooling_rate_critical"]:
            flags.append(RiskFlag(
                level=RiskLevel.CRITICAL, category="Cooling Rate",
                parameter="cooling_rate_Cps", observed_value=cr,
                limit_value=L["cooling_rate_critical"],
                message=f"Cooling rate {cr:.1f}°C/s exceeds critical limit {L['cooling_rate_critical']:.1f}°C/s.",
                recommendation="Reduce cooling rate — solder joint cracking risk."
            ))
        elif cr > L["cooling_rate_max"]:
            flags.append(RiskFlag(
                level=RiskLevel.WARNING, category="Cooling Rate",
                parameter="cooling_rate_Cps", observed_value=cr,
                limit_value=L["cooling_rate_max"],
                message=f"Cooling rate {cr:.1f}°C/s is aggressive (recommended ≤{L['cooling_rate_max']:.1f}°C/s).",
                recommendation="Target cooling rate of 2–4°C/s for joint reliability."
            ))
        else:
            flags.append(RiskFlag(
                level=RiskLevel.OK, category="Cooling Rate",
                parameter="cooling_rate_Cps", observed_value=cr,
                limit_value=L["cooling_rate_max"],
                message=f"Cooling rate {cr:.1f}°C/s within acceptable range.",
                recommendation="No action required."
            ))

        # ── Soak temperature ─────────────────────────────────────────────
        st = profile.soak_temp_C
        if not (L["soak_temp_min"] <= st <= L["soak_temp_max"]):
            flags.append(RiskFlag(
                level=RiskLevel.WARNING, category="Soak Temperature",
                parameter="soak_temp_C", observed_value=st,
                limit_value=L["soak_temp_max"],
                message=f"Soak temp {st:.1f}°C outside recommended range "
                        f"{L['soak_temp_min']:.0f}–{L['soak_temp_max']:.0f}°C.",
                recommendation="Adjust soak temperature to 150–180°C to preserve flux activity."
            ))

        return flags

    # ------------------------------------------------------------------
    # Thermal Field Assessment
    # ------------------------------------------------------------------

    def _assess_thermal_field(self, prediction: PredictionResult) -> List[RiskFlag]:
        flags = []
        L = self.limits

        # PCB uniformity
        pcb_cv = prediction.pcb_uniformity
        if pcb_cv > L["uniformity_critical"]:
            flags.append(RiskFlag(
                level=RiskLevel.CRITICAL, category="PCB Thermal Uniformity",
                parameter="pcb_uniformity", observed_value=pcb_cv,
                limit_value=L["uniformity_critical"],
                message=f"PCB non-uniformity (CV={pcb_cv:.5f}) critically high.",
                recommendation="Investigate board layout and oven zone balancing."
            ))
        elif pcb_cv > L["uniformity_warning"]:
            flags.append(RiskFlag(
                level=RiskLevel.WARNING, category="PCB Thermal Uniformity",
                parameter="pcb_uniformity", observed_value=pcb_cv,
                limit_value=L["uniformity_warning"],
                message=f"PCB non-uniformity (CV={pcb_cv:.5f}) elevated.",
                recommendation="Extend soak zone or reduce preheat ramp rate."
            ))

        # DIE uniformity
        die_cv = prediction.die_uniformity
        if die_cv > L["uniformity_critical"]:
            flags.append(RiskFlag(
                level=RiskLevel.CRITICAL, category="DIE Thermal Uniformity",
                parameter="die_uniformity", observed_value=die_cv,
                limit_value=L["uniformity_critical"],
                message=f"Die non-uniformity (CV={die_cv:.6f}) critically high — die cracking risk.",
                recommendation="Reduce ramp rates. Verify die attach coverage."
            ))
        elif die_cv > L["uniformity_warning"]:
            flags.append(RiskFlag(
                level=RiskLevel.WARNING, category="DIE Thermal Uniformity",
                parameter="die_uniformity", observed_value=die_cv,
                limit_value=L["uniformity_warning"],
                message=f"Die non-uniformity (CV={die_cv:.6f}) elevated.",
                recommendation="Reduce soak ramp rate or extend soak time."
            ))

        # Spatial gradients
        for label, grad in [("PCB", prediction.pcb_gradient_max),
                            ("DIE", prediction.die_gradient_max)]:
            if grad > L["gradient_critical"]:
                flags.append(RiskFlag(
                    level=RiskLevel.CRITICAL, category=f"{label} Spatial Gradient",
                    parameter=f"{label.lower()}_gradient_max", observed_value=grad,
                    limit_value=L["gradient_critical"],
                    message=f"{label} max gradient {grad:.3f} critically high (warpage risk).",
                    recommendation=f"Significantly reduce {label} ramp rate and extend soak."
                ))
            elif grad > L["gradient_warning"]:
                flags.append(RiskFlag(
                    level=RiskLevel.WARNING, category=f"{label} Spatial Gradient",
                    parameter=f"{label.lower()}_gradient_max", observed_value=grad,
                    limit_value=L["gradient_warning"],
                    message=f"{label} max gradient {grad:.3f} elevated.",
                    recommendation=f"Consider adjusting {label} thermal boundary conditions."
                ))

        return flags

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_scores(self, prediction: PredictionResult) -> Dict[str, float]:
        """
        Compute objective scores for optimization.
        All scores: 0 = ideal, higher = worse.
        These directly use real °C ranges and gradients.
        """
        # PCB: use range (°C) + gradient as uniformity measure
        pcb_score = prediction.pcb_range + 0.5 * prediction.pcb_gradient_max

        # DIE: same but scaled up since range is ~100x smaller
        die_score = prediction.die_range + 0.5 * prediction.die_gradient_max

        # Combined field score
        field_score = (self.pcb_weight * pcb_score +
                       self.die_weight * die_score)

        return {
            'pcb_score':   pcb_score,
            'die_score':   die_score,
            'field_score': field_score,
        }

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------

    def _generate_suggestions(self, flags: List[RiskFlag]) -> List[str]:
        suggestions = []
        seen = set()
        for flag in flags:
            if (flag.level in (RiskLevel.CRITICAL, RiskLevel.WARNING)
                    and flag.recommendation != "No action required."
                    and flag.recommendation not in seen):
                suggestions.append(flag.recommendation)
                seen.add(flag.recommendation)

        if not suggestions:
            suggestions.append(
                "Profile is SAC305-compliant. Run NSGA-II Pareto optimization "
                "to further improve PCB/DIE uniformity tradeoff."
            )
        return suggestions[:6]

    # ------------------------------------------------------------------
    # Main Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        profile: ReflowProfile,
        verbose: bool = True
    ) -> AdvisorReport:
        """
        Full evaluation: predict → validate → assess → score → report.

        Parameters
        ----------
        profile : ReflowProfile
        verbose : Print summary to console

        Returns
        -------
        AdvisorReport
        """
        # 1. Inference
        prediction = self.engine.predict(profile)

        # 2. Validate SAC305 limits
        profile_flags = self._validate_profile(profile)

        # 3. Assess thermal field
        field_flags = self._assess_thermal_field(prediction)

        all_flags = profile_flags + field_flags

        # 4. Overall risk
        if any(f.level == RiskLevel.CRITICAL for f in all_flags):
            overall = RiskLevel.CRITICAL
        elif any(f.level == RiskLevel.WARNING for f in all_flags):
            overall = RiskLevel.WARNING
        else:
            overall = RiskLevel.OK

        # 5. Scores
        scores = self._compute_scores(prediction)
        n_critical = sum(1 for f in all_flags if f.level == RiskLevel.CRITICAL)
        n_warning  = sum(1 for f in all_flags if f.level == RiskLevel.WARNING)
        process_score  = min(n_critical * 0.3 + n_warning * 0.1, 1.0)
        composite      = 0.7 * scores['field_score'] + 0.3 * process_score

        # 6. Suggestions
        non_ok_flags = [f for f in all_flags if f.level != RiskLevel.OK]
        suggestions  = self._generate_suggestions(non_ok_flags)

        report = AdvisorReport(
            profile          = profile,
            prediction       = prediction,
            flags            = non_ok_flags,
            overall_risk     = overall,
            uniformity_score = scores['field_score'],
            process_score    = process_score,
            composite_score  = composite,
            suggestions      = suggestions,
        )

        if verbose:
            report.print_summary()

        return report

    # ------------------------------------------------------------------
    # Optimizer-facing methods
    # ------------------------------------------------------------------

    def score_profile(self, profile: ReflowProfile) -> float:
        """Single-objective score for GA. Returns composite_score."""
        return self.evaluate(profile, verbose=False).composite_score

    def score_multi(self, profile: ReflowProfile) -> Tuple[float, float]:
        """Multi-objective scores for NSGA-II: (pcb_score, die_score)."""
        prediction = self.engine.predict(profile)
        scores     = self._compute_scores(prediction)
        return (scores['pcb_score'], scores['die_score'])


# ---------------------------------------------------------------------------
# Quick Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine  = InferenceEngine("models/thermal_digital_twin.pth")
    advisor = ReflowAdvisor(engine)

    print("\n>>> Test 1: Default SAC305 profile")
    profile = ReflowProfile()
    advisor.evaluate(profile)

    print("\n>>> Test 2: Aggressive profile (should show warnings)")
    bad = ReflowProfile(
        peak_temp_C=270.0,
        tal_s=95.0,
        cooling_rate_Cps=7.0,
        ramp_rate_Cps=4.5,
    )
    advisor.evaluate(bad)

    print("\n>>> Test 3: Optimizer scores")
    s1 = advisor.score_profile(ReflowProfile())
    s2 = advisor.score_profile(bad)
    print(f"  Default  composite score: {s1:.4f}")
    print(f"  Aggressive composite score: {s2:.4f}")