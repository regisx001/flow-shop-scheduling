"""
Grey Wolf Optimizer (GWO)
==========================

A pure, generic implementation of the Grey Wolf Optimizer as described in:

    Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014).
    Grey Wolf Optimizer.
    Advances in Engineering Software, 69, 46-61.

This module provides a single class ``GreyWolfOptimizer`` that works on any
continuous optimisation problem (minimisation). Users supply a fitness
function and the algorithm evolves a population of candidate solutions.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass, field


# ====================================================================
# Result container
# ====================================================================

@dataclass
class GWOResult:
    """Holds the output of a single GWO run."""

    best_position: np.ndarray
    """Best position found in the continuous search space."""

    best_fitness: float
    """Fitness value of the best position (alpha wolf)."""

    convergence: np.ndarray
    """Convergence curve, shape (max_iter,); best fitness per iteration."""

    runtime_sec: float
    """Wall-clock execution time in seconds."""

    n_evaluations: int
    """Total number of fitness function evaluations performed."""

    params: dict = field(default_factory=dict)
    """Snapshot of the algorithm parameters used for this run."""

    def summary(self) -> str:
        return (f"f*={self.best_fitness:.6e}  "
                f"evals={self.n_evaluations}  "
                f"time={self.runtime_sec:.4f}s")


# ====================================================================
# Grey Wolf Optimizer
# ====================================================================

class GreyWolfOptimizer:
    """
    Grey Wolf Optimizer (GWO) — a population-based metaheuristic for
    continuous global optimisation (minimisation), inspired by the
    social hierarchy and hunting behaviour of grey wolves.

    Core mechanics (Mirjalili et al., 2014)
    ----------------------------------------
    The population is split into four social ranks:

    * **Alpha (α)** — the fittest solution (leader).
    * **Beta  (β)** — the second-best solution.
    * **Delta (δ)** — the third-best solution.
    * **Omega (ω)** — all remaining wolves, guided by α, β, δ.

    Each wolf updates its position using the three best solutions:

        D_α = |C₁·X_α - X|      X₁ = X_α - A₁·D_α
        D_β = |C₂·X_β - X|      X₂ = X_β - A₂·D_β
        D_δ = |C₃·X_δ - X|      X₃ = X_δ - A₃·D_δ
        X(t+1) = (X₁ + X₂ + X₃) / 3

    where ``A = 2·a·r₁ - a`` (│A│ < 1 → exploitation, │A│ > 1 → exploration)
    and   ``C = 2·r₂`` (stochastic weight on the prey position).

    The parameter ``a`` decreases linearly from 2 → 0 over iterations,
    balancing exploration and exploitation.

    Parameters
    ----------
    fitness_fn : callable
        Function ``f(x) -> float`` where ``x`` is a 1-D numpy array of
        shape ``(dim,)``. The algorithm **minimises** this function.
    dim : int
        Dimensionality of the search space.
    pop_size : int
        Number of wolves (population size). Default 30.
    max_iter : int
        Maximum number of iterations. Default 200.
    lb : float or np.ndarray
        Lower bound(s) of the search space. If float, same bound for all
        dimensions. Default ``0.0``.
    ub : float or np.ndarray
        Upper bound(s) of the search space. Default ``1.0``.
    seed : int or None
        Random seed for reproducibility. Default ``None``.

    Example
    -------
    >>> import numpy as np
    >>> def sphere(x):
    ...     return float(np.sum(x ** 2))
    ...
    >>> gwo = GreyWolfOptimizer(sphere, dim=10, pop_size=30, max_iter=100,
    ...                         lb=-5.0, ub=5.0, seed=42)
    >>> result = gwo.run()
    >>> print(f"{result.best_fitness:.4f}")
    """

    # ── Lifecycle ──────────────────────────────────────────────────

    def __init__(
        self,
        fitness_fn: callable,
        dim: int,
        pop_size: int = 30,
        max_iter: int = 200,
        lb: float = 0.0,
        ub: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = np.broadcast_to(lb, (dim,)).astype(float)
        self.ub = np.broadcast_to(ub, (dim,)).astype(float)
        self.seed = seed

        # Internal state (set by run())
        self._rng: np.random.Generator | None = None
        self._X: np.ndarray | None = None           # (pop_size, dim) positions
        self._fitness: np.ndarray | None = None      # (pop_size,) fitness values
        self._alpha_pos: np.ndarray | None = None    # best position
        self._alpha_fit: float = np.inf              # best fitness
        self._beta_pos: np.ndarray | None = None     # 2nd best
        self._beta_fit: float = np.inf
        self._delta_pos: np.ndarray | None = None    # 3rd best
        self._delta_fit: float = np.inf
        self._convergence: np.ndarray | None = None
        self._n_eval: int = 0

    # ── Public API ────────────────────────────────────────────────

    def run(
        self,
        init_positions: np.ndarray | None = None,
    ) -> GWOResult:
        """
        Execute the GWO optimisation.

        Parameters
        ----------
        init_positions : np.ndarray, shape (k, dim) with k ≤ pop_size, optional
            Pre-defined positions to inject into the initial population.

        Returns
        -------
        GWOResult
            Container with the best solution (alpha wolf), convergence
            history, and runtime statistics.
        """
        self._rng = np.random.default_rng(self.seed)
        t_start = time.time()

        # 1. Initialise population uniformly in [lb, ub]
        self._X = self._rng.uniform(
            self.lb, self.ub, size=(self.pop_size, self.dim)
        ).astype(float)

        if init_positions is not None:
            n = min(len(init_positions), self.pop_size)
            self._X[:n] = init_positions[:n]

        # 2. Evaluate initial population and find alpha, beta, delta
        self._evaluate_all()
        self._update_hierarchy()

        self._convergence = np.zeros(self.max_iter)
        self._convergence[0] = self._alpha_fit

        # Track best-ever separately from current hierarchy
        best_ever_fit = self._alpha_fit

        # 3. Main loop
        for t in range(1, self.max_iter):
            # Linearly decreasing a from 2 to 0
            a = 2.0 * (1 - t / self.max_iter)

            for i in range(self.pop_size):
                self._update_agent(i, a)

            # Evaluate and update hierarchy
            self._evaluate_all()
            self._update_hierarchy()

            # Track best-ever fitness for convergence reporting
            if self._alpha_fit < best_ever_fit:
                best_ever_fit = self._alpha_fit

            self._convergence[t] = best_ever_fit

        runtime = time.time() - t_start

        return GWOResult(
            best_position=self._alpha_pos.copy(),
            best_fitness=best_ever_fit,
            convergence=self._convergence.copy(),
            runtime_sec=runtime,
            n_evaluations=self._n_eval,
            params=self._param_snapshot(),
        )

    # ── Core GWO mechanics ────────────────────────────────────────

    def _update_agent(self, i: int, a: float) -> None:
        """
        Update the position of a single wolf using alpha, beta, delta.

        Mirjalili et al. 2014, Eqs. 3.1–3.7.
        """
        # --- Alpha component ---
        r1 = self._rng.random(self.dim)
        r2 = self._rng.random(self.dim)
        A1 = 2.0 * a * r1 - a
        C1 = 2.0 * r2
        D_alpha = np.abs(C1 * self._alpha_pos - self._X[i])
        X1 = self._alpha_pos - A1 * D_alpha

        # --- Beta component ---
        r1 = self._rng.random(self.dim)
        r2 = self._rng.random(self.dim)
        A2 = 2.0 * a * r1 - a
        C2 = 2.0 * r2
        D_beta = np.abs(C2 * self._beta_pos - self._X[i])
        X2 = self._beta_pos - A2 * D_beta

        # --- Delta component ---
        r1 = self._rng.random(self.dim)
        r2 = self._rng.random(self.dim)
        A3 = 2.0 * a * r1 - a
        C3 = 2.0 * r2
        D_delta = np.abs(C3 * self._delta_pos - self._X[i])
        X3 = self._delta_pos - A3 * D_delta

        # --- Final position: average of the three guides ---
        self._X[i] = (X1 + X2 + X3) / 3.0

        np.clip(self._X[i], self.lb, self.ub, out=self._X[i])

    # ── Evaluation ────────────────────────────────────────────────

    def _evaluate_all(self) -> None:
        """Evaluate fitness for all individuals."""
        if self._fitness is None:
            self._fitness = np.empty(self.pop_size)

        for i in range(self.pop_size):
            self._fitness[i] = self.fitness_fn(self._X[i])
            self._n_eval += 1

    def _update_hierarchy(self) -> None:
        """
        Identify alpha, beta, and delta as the current top 3 by fitness.
        These guide the search in the next iteration.
        """
        order = np.argsort(self._fitness)

        # Alpha: best
        idx_a = order[0]
        self._alpha_fit = self._fitness[idx_a]
        self._alpha_pos = self._X[idx_a].copy()

        # Beta: second best (or alpha if only 1 wolf)
        idx_b = order[1] if self.pop_size > 1 else order[0]
        self._beta_fit = self._fitness[idx_b]
        self._beta_pos = self._X[idx_b].copy()

        # Delta: third best (or alpha if < 3 wolves)
        idx_d = order[2] if self.pop_size > 2 else order[0]
        self._delta_fit = self._fitness[idx_d]
        self._delta_pos = self._X[idx_d].copy()

    # ── Helpers ───────────────────────────────────────────────────

    def _param_snapshot(self) -> dict:
        return {
            "pop_size": self.pop_size,
            "max_iter": self.max_iter,
            "dim": self.dim,
            "lb": self.lb.tolist() if hasattr(self.lb, "__len__") else self.lb,
            "ub": self.ub.tolist() if hasattr(self.ub, "__len__") else self.ub,
            "seed": self.seed,
        }

    def __repr__(self) -> str:
        return (
            f"GreyWolfOptimizer(dim={self.dim}, pop={self.pop_size}, "
            f"iter={self.max_iter}, seed={self.seed})"
        )


# ====================================================================
# Quick demo — sphere function
# ====================================================================

if __name__ == "__main__":

    def sphere(x: np.ndarray) -> float:
        """Quadratic sphere function (minimum = 0 at origin)."""
        return float(np.sum(x ** 2))

    gwo = GreyWolfOptimizer(
        fitness_fn=sphere,
        dim=30,
        pop_size=30,
        max_iter=200,
        lb=-100.0,
        ub=100.0,
        seed=42,
    )

    result = gwo.run()
    print("Grey Wolf Optimizer — demo on sphere function (dim=30)")
    print(f"  Best fitness: {result.best_fitness:.6e}")
    print(f"  Evaluations:  {result.n_evaluations}")
    print(f"  Runtime:      {result.runtime_sec:.4f}s")
    print(f"  Best pos (first 5 dims): {result.best_position[:5]}")
