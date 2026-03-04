"""
=============================================================================
Stage 2 — Evolutionary Optimizer  (UPDATED for Phase 1 model)
PCB Reflow Thermal Digital Twin
=============================================================================
Two optimization modes:
  1. GA     — single-objective: minimize composite_score
  2. NSGA-II — multi-objective Pareto: minimize (pcb_score, die_score)

Both use ReflowAdvisor as fitness evaluator (surrogate-in-the-loop).

Usage:
    opt = ReflowOptimizer(advisor)

    # Single-objective
    best, history = opt.run_ga(n_generations=80)

    # Multi-objective
    pareto = opt.run_nsga2(n_generations=80)
=============================================================================
"""

import numpy as np
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from inference_engine import ReflowProfile
from reflow_advisor   import ReflowAdvisor


# ---------------------------------------------------------------------------
# SAC305 Parameter Bounds
# (min, max) for each feature in FEATURE_COLS order
# ---------------------------------------------------------------------------

PROFILE_BOUNDS = {
    'peak_temp_C':            (235.0, 260.0),
    'tal_s':                  (30.0,   60.0),
    'soak_temp_C':            (150.0, 180.0),
    'soak_time_s':            (60.0,  120.0),
    'ramp_rate_Cps':          (1.0,    3.0),
    'cooling_rate_Cps':       (1.5,    4.0),
    't_total_s':              (200.0, 400.0),
    'T_amb_C':                (20.0,   30.0),
    'copper_area_fraction':   (0.15,   0.35),
    'paste_coverage_fraction':(0.01,   0.03),
    'k_die_WmK':              (130.0, 148.0),
    'k_pcb_WmK':              (0.30,   0.50),
}

PARAM_KEYS   = list(PROFILE_BOUNDS.keys())
BOUNDS_ARRAY = np.array([PROFILE_BOUNDS[k] for k in PARAM_KEYS])  # (12, 2)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class GAHistory:
    best_scores:   List[float] = field(default_factory=list)
    mean_scores:   List[float] = field(default_factory=list)
    worst_scores:  List[float] = field(default_factory=list)
    generations:   List[int]   = field(default_factory=list)
    elapsed_times: List[float] = field(default_factory=list)

    def log(self, gen, scores, elapsed):
        self.generations.append(gen)
        self.best_scores.append(float(np.min(scores)))
        self.mean_scores.append(float(np.mean(scores)))
        self.worst_scores.append(float(np.max(scores)))
        self.elapsed_times.append(elapsed)

    def print_progress(self, gen, n_gen):
        print(
            f"  Gen {gen:4d}/{n_gen} | "
            f"Best: {self.best_scores[-1]:.4f} | "
            f"Mean: {self.mean_scores[-1]:.4f} | "
            f"Time: {self.elapsed_times[-1]:.1f}s"
        )


@dataclass
class ParetoSolution:
    profile:          ReflowProfile
    objectives:       Tuple[float, float]   # (pcb_score, die_score)
    composite_score:  float
    rank:             int   = 0
    crowding_distance: float = 0.0


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class ReflowOptimizer:
    """
    Evolutionary optimizer for SAC305 reflow profile improvement.

    Bounds are enforced to keep all candidates SAC305-compliant.
    Fitness uses ReflowAdvisor → InferenceEngine (surrogate-in-the-loop).
    """

    def __init__(
        self,
        advisor: ReflowAdvisor,
        bounds:  np.ndarray = BOUNDS_ARRAY,
        seed:    int = 42,
    ):
        self.advisor  = advisor
        self.bounds   = bounds
        self.n_params = bounds.shape[0]
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def _decode(self, x: np.ndarray) -> ReflowProfile:
        """Raw parameter vector → ReflowProfile."""
        kwargs = {PARAM_KEYS[i]: float(x[i]) for i in range(self.n_params)}
        return ReflowProfile(**kwargs)

    def _encode(self, profile: ReflowProfile) -> np.ndarray:
        """ReflowProfile → raw parameter vector."""
        return np.array([getattr(profile, k) for k in PARAM_KEYS], dtype=np.float64)

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def _random_individual(self) -> np.ndarray:
        return (
            np.random.rand(self.n_params) *
            (self.bounds[:, 1] - self.bounds[:, 0]) +
            self.bounds[:, 0]
        )

    # ------------------------------------------------------------------
    # SBX Crossover + Polynomial Mutation
    # ------------------------------------------------------------------

    def _sbx_crossover(self, p1: np.ndarray, p2: np.ndarray, eta: float = 20.0) -> Tuple:
        """Simulated Binary Crossover."""
        c1, c2 = p1.copy(), p2.copy()
        for i in range(self.n_params):
            if np.random.rand() < 0.5:
                u = np.random.rand()
                beta = (2*u)**(1/(eta+1)) if u <= 0.5 else (1/(2*(1-u)))**(1/(eta+1))
                c1[i] = 0.5 * ((1+beta)*p1[i] + (1-beta)*p2[i])
                c2[i] = 0.5 * ((1-beta)*p1[i] + (1+beta)*p2[i])
        return self._clip(c1), self._clip(c2)

    def _poly_mutation(self, x: np.ndarray, eta: float = 20.0, prob: float = None) -> np.ndarray:
        """Polynomial Mutation."""
        prob = prob or (1.0 / self.n_params)
        xm   = x.copy()
        for i in range(self.n_params):
            if np.random.rand() < prob:
                u   = np.random.rand()
                dq  = (2*u)**(1/(eta+1)) - 1 if u < 0.5 else 1 - (2*(1-u))**(1/(eta+1))
                rng = self.bounds[i, 1] - self.bounds[i, 0]
                xm[i] = np.clip(x[i] + dq * rng, self.bounds[i, 0], self.bounds[i, 1])
        return xm

    # ------------------------------------------------------------------
    # Single-Objective GA
    # ------------------------------------------------------------------

    def run_ga(
        self,
        n_pop:           int = 50,
        n_generations:   int = 80,
        initial_profile: Optional[ReflowProfile] = None,
        verbose:         bool = True,
    ) -> Tuple[ReflowProfile, GAHistory]:
        """
        GA minimizing composite_score (lower = better).
        Returns best profile found + history object.
        """
        if verbose:
            print(f"\n🧬 GA Optimization  |  pop={n_pop}  gens={n_generations}")

        # Initialize population
        pop = [self._random_individual() for _ in range(n_pop)]
        if initial_profile is not None:
            pop[0] = self._clip(self._encode(initial_profile))

        # Evaluate
        scores = np.array([
            self.advisor.score_profile(self._decode(x)) for x in pop
        ])

        history = GAHistory()
        t_start = time.time()

        for gen in range(1, n_generations + 1):
            t0 = time.time()
            new_pop = []

            # Elitism: keep best individual
            best_idx = np.argmin(scores)
            new_pop.append(pop[best_idx].copy())

            # Generate offspring
            while len(new_pop) < n_pop:
                # Tournament selection
                def tournament(k=3):
                    idx = np.random.choice(n_pop, k, replace=False)
                    return pop[idx[np.argmin(scores[idx])]]

                p1, p2 = tournament(), tournament()
                c1, c2 = self._sbx_crossover(p1, p2)
                c1 = self._poly_mutation(c1)
                c2 = self._poly_mutation(c2)
                new_pop.extend([c1, c2])

            pop    = new_pop[:n_pop]
            scores = np.array([
                self.advisor.score_profile(self._decode(x)) for x in pop
            ])

            history.log(gen, scores, time.time() - t0)

            if verbose and gen % 10 == 0:
                history.print_progress(gen, n_generations)

        best_x = pop[np.argmin(scores)]
        best   = self._decode(best_x)

        if verbose:
            print(f"\n✅ GA complete | Best score: {np.min(scores):.4f} | "
                  f"Total time: {time.time()-t_start:.1f}s")
            print(f"   Best profile peak_temp={best.peak_temp_C:.1f}°C  "
                  f"TAL={best.tal_s:.1f}s  cooling={best.cooling_rate_Cps:.2f}°C/s")

        return best, history

    # ------------------------------------------------------------------
    # Multi-Objective NSGA-II (via pymoo)
    # ------------------------------------------------------------------

    def run_nsga2(
        self,
        n_pop:         int = 60,
        n_generations: int = 80,
        verbose:       bool = True,
    ) -> List[ParetoSolution]:
        """
        NSGA-II minimizing (pcb_score, die_score) simultaneously.
        Returns list of ParetoSolution objects.
        """
        try:
            from pymoo.core.problem import Problem
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.operators.crossover.sbx import SBX
            from pymoo.operators.mutation.pm import PM
            from pymoo.optimize import minimize as pymoo_minimize
        except ImportError:
            print("pymoo not installed. Run: pip install pymoo")
            return []

        advisor = self.advisor
        _bounds = self.bounds          # ← capture instance bounds (respects peak_ub constraint)

        class _ReflowProblem(Problem):
            def __init__(self_inner):
                super().__init__(
                    n_var=len(PARAM_KEYS), n_obj=2, n_constr=0,
                    xl=_bounds[:, 0], xu=_bounds[:, 1]   # ← was hardcoded BOUNDS_ARRAY
                )

            def _evaluate(self_inner, X, out, *args, **kwargs):
                F = []
                for x_vec in X:
                    profile = ReflowProfile(**{PARAM_KEYS[i]: float(x_vec[i])
                                               for i in range(len(PARAM_KEYS))})
                    obj1, obj2 = advisor.score_multi(profile)
                    F.append([obj1, obj2])
                out['F'] = np.array(F)

        if verbose:
            print(f"\n🚀 NSGA-II  |  pop={n_pop}  gens={n_generations}")

        result = pymoo_minimize(
            _ReflowProblem(),
            NSGA2(
                pop_size=n_pop,
                crossover=SBX(prob=0.9, eta=20),
                mutation=PM(prob=1.0/len(PARAM_KEYS), eta=20),
                eliminate_duplicates=True,
            ),
            ('n_gen', n_generations),
            seed=42,
            verbose=verbose,
        )

        # Convert to ParetoSolution objects
        solutions = []
        for x_vec, obj in zip(result.X, result.F):
            profile = ReflowProfile(**{PARAM_KEYS[i]: float(x_vec[i])
                                       for i in range(len(PARAM_KEYS))})
            solutions.append(ParetoSolution(
                profile         = profile,
                objectives      = (float(obj[0]), float(obj[1])),
                composite_score = float(obj[0] + obj[1]),
            ))

        # Sort by composite score
        solutions.sort(key=lambda s: s.composite_score)

        if verbose:
            print(f"\n✅ NSGA-II complete | Pareto solutions: {len(solutions)}")
            print(f"   PCB obj range: {result.F[:, 0].min():.4f} → {result.F[:, 0].max():.4f}")
            print(f"   DIE obj range: {result.F[:, 1].min():.4f} → {result.F[:, 1].max():.4f}")

        return solutions

    # ------------------------------------------------------------------
    # Comparison Utility
    # ------------------------------------------------------------------

    def compare_profiles(
        self,
        baseline: ReflowProfile,
        optimized: ReflowProfile,
    ) -> dict:
        """Print and return comparison dict."""
        b_score = self.advisor.score_profile(baseline)
        o_score = self.advisor.score_profile(optimized)
        improvement = (b_score - o_score) / b_score * 100 if b_score != 0 else 0.0

        print(f"\n  Baseline score   : {b_score:.4f}")
        print(f"  Optimized score  : {o_score:.4f}")
        print(f"  Improvement      : {improvement:+.1f}%")

        return {
            'baseline_score':   b_score,
            'optimized_score':  o_score,
            'improvement_pct':  improvement,
        }


# ---------------------------------------------------------------------------
# Quick Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from inference_engine import InferenceEngine
    from reflow_advisor   import ReflowAdvisor

    engine   = InferenceEngine("models/thermal_digital_twin.pth")
    advisor  = ReflowAdvisor(engine)
    opt      = ReflowOptimizer(advisor)

    print("Running GA (20 gens quick test)...")
    best, history = opt.run_ga(n_pop=20, n_generations=20)

    print("\nRunning NSGA-II (20 gens quick test)...")
    pareto = opt.run_nsga2(n_pop=20, n_generations=20)

    print(f"\n✅ Done | Pareto solutions: {len(pareto)}")
    print(f"   Best composite: {pareto[0].composite_score:.4f}")
    print(f"   Peak temp: {pareto[0].profile.peak_temp_C:.1f}°C  "
          f"TAL: {pareto[0].profile.tal_s:.1f}s")