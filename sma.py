"""
Slime Mould Algorithm (SMA)
============================

A pure, generic implementation of the Slime Mould Algorithm as described in:

    Li, S., Chen, H., Wang, M., Heidari, A. A., & Mirjalili, S. (2020).
    Slime mould algorithm: A new method for stochastic optimization.
    Future Generation Computer Systems, 111, 300-323.

This module provides a single class ``SlimeMouldAlgorithm`` that works on any
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
class SMAResult:
    """Holds the output of a single SMA run."""

    best_position: np.ndarray
    """Best position found in the continuous search space."""

    best_fitness: float
    """Fitness value of the best position."""

    convergence: np.ndarray
    """Convergence curve, shape (max_iter,); best fitness per iteration."""

    runtime_sec: float
    """Wall-clock execution time in seconds."""

    n_evaluations: int
    """Total number of fitness function evaluations performed."""

    n_restarts: int
    """Number of times the stagnation-restart mechanism was triggered."""

    params: dict = field(default_factory=dict)
    """Snapshot of the algorithm parameters used for this run."""

    def summary(self) -> str:
        return (f"f*={self.best_fitness:.6e}  "
                f"evals={self.n_evaluations}  "
                f"restarts={self.n_restarts}  "
                f"time={self.runtime_sec:.4f}s")


# ====================================================================
# Slime Mould Algorithm
# ====================================================================

class SlimeMouldAlgorithm:
    """
    Slime Mould Algorithm (SMA) — a population-based metaheuristic for
    continuous global optimisation (minimisation).

    Core mechanics (Li et al., 2020)
    ---------------------------------
    * Each agent represents a position in a D-dimensional continuous space.
    * Positions are updated using two modes controlled by a probabilistic
      switch ``p = tanh(|f_i - f_best|)``:

      - **Approach food** (``r < p``): move toward the best position,
        modulated by a weight W and a random pair difference.
      - **Wrap food**    (``r >= p``): contract the position around itself.

    * A small oscillation probability ``z`` injects random positions to
      maintain diversity.
    * Weights W favour better individuals (W > 1) and penalise worse ones
      (W < 1), mimicking the cytoplasmic flow of the slime mould.

    Enhancements over the baseline
    ------------------------------
    * **Adaptive z**: starts higher (exploration) then decays to the
      configured base value.
    * **Elitism**: the top ``elite_size`` individuals survive unchanged
      each generation.
    * **Stagnation restart**: if best fitness stalls for ``stall_limit``
      iterations, the worst half of the population is reinitialised
      around the current best with small perturbations.

    Parameters
    ----------
    fitness_fn : callable
        Function ``f(x) -> float`` where ``x`` is a 1-D numpy array of
        shape ``(dim,)``. The algorithm **minimises** this function.
    dim : int
        Dimensionality of the search space.
    pop_size : int
        Number of agents (population size). Default 30.
    max_iter : int
        Maximum number of iterations. Default 200.
    lb : float or np.ndarray
        Lower bound(s) of the search space. If float, same bound for all
        dimensions. Default ``0.0``.
    ub : float or np.ndarray
        Upper bound(s) of the search space. Default ``1.0``.
    z : float
        Base oscillation (randomisation) probability. Default ``0.03``.
    elite_size : int
        Number of top individuals preserved unchanged each generation.
        Default ``2``.
    stall_limit : int
        Number of iterations without fitness improvement before triggering
        a restart. Default ``50``.
    seed : int or None
        Random seed for reproducibility. Default ``None``.

    Example
    -------
    >>> import numpy as np
    >>> def sphere(x):
    ...     return float(np.sum(x ** 2))
    ...
    >>> sma = SlimeMouldAlgorithm(sphere, dim=10, pop_size=30, max_iter=100,
    ...                           lb=-5.0, ub=5.0, seed=42)
    >>> result = sma.run()
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
        z: float = 0.03,
        elite_size: int = 2,
        stall_limit: int = 50,
        seed: int | None = None,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = np.broadcast_to(lb, (dim,)).astype(float)
        self.ub = np.broadcast_to(ub, (dim,)).astype(float)
        self.z = z
        self.elite_size = min(elite_size, pop_size)
        self.stall_limit = stall_limit
        self.seed = seed

        # Internal state (set by run())
        self._rng: np.random.Generator | None = None
        self._X: np.ndarray | None = None         # (pop_size, dim) positions
        self._fitness: np.ndarray | None = None    # (pop_size,) fitness values
        self._best_pos: np.ndarray | None = None   # global best position
        self._best_fit: float = np.inf             # global best fitness
        self._convergence: np.ndarray | None = None
        self._n_eval: int = 0
        self._n_restarts: int = 0

    # ── Public API ────────────────────────────────────────────────

    def run(
        self,
        init_positions: np.ndarray | None = None,
    ) -> SMAResult:
        """
        Execute the SMA optimisation.

        Parameters
        ----------
        init_positions : np.ndarray, shape (k, dim) with k ≤ pop_size, optional
            Pre-defined positions to inject into the initial population.
            The first ``k`` rows of the population are set to these values;
            the remaining ``pop_size - k`` rows are randomly initialised.

        Returns
        -------
        SMAResult
            Container with the best solution found, convergence history,
            and runtime statistics.
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

        # 2. Evaluate initial population
        self._evaluate_all()

        # 3. Main loop
        stall_counter = 0
        self._convergence = np.zeros(self.max_iter)

        for t in range(1, self.max_iter + 1):
            # Sort by fitness and compute weights
            order = np.argsort(self._fitness)
            W = self._compute_weights(order)

            a = np.arctanh(np.clip(1 - t / self.max_iter, -0.999999, 0.999999))
            vc = 1 - t / self.max_iter
            z_eff = self._adaptive_z(t)

            # Preserve elite
            elite_indices = set(order[: self.elite_size])
            X_elite = (
                self._X[list(elite_indices)].copy()
                if self.elite_size
                else None
            )

            # Update each agent
            for i in range(self.pop_size):
                if self.elite_size and i in elite_indices:
                    continue
                self._update_agent(i, W[i], a, vc, z_eff)

            # Restore elite
            if self.elite_size and X_elite is not None:
                self._X[list(elite_indices)] = X_elite

            # Evaluate (skip elite — their fitness is unchanged)
            self._evaluate_all(skip_indices=elite_indices)

            # Track global best
            gen_best_idx = int(np.argmin(self._fitness))
            gen_best_fit = float(self._fitness[gen_best_idx])

            if gen_best_fit < self._best_fit:
                self._best_fit = gen_best_fit
                self._best_pos = self._X[gen_best_idx].copy()
                stall_counter = 0
            else:
                stall_counter += 1

            # Stagnation restart
            if stall_counter >= self.stall_limit:
                self._restart_worst_half(order)
                stall_counter = 0

            self._convergence[t - 1] = self._best_fit

        runtime = time.time() - t_start

        return SMAResult(
            best_position=self._best_pos.copy(),
            best_fitness=self._best_fit,
            convergence=self._convergence.copy(),
            runtime_sec=runtime,
            n_evaluations=self._n_eval,
            n_restarts=self._n_restarts,
            params=self._param_snapshot(),
        )

    # ── Core SMA mechanics ────────────────────────────────────────

    def _compute_weights(self, order: np.ndarray) -> np.ndarray:
        """
        Compute weight W for each agent (Eq. 2.5 in Li et al. 2020).

        Better individuals (first half)  → ``W > 1`` (amplified influence)
        Worse individuals  (second half) → ``W < 1`` (dampened influence)
        """
        bF = self._fitness[order[0]]
        wF = self._fitness[order[-1]]
        denom = (bF - wF) if (bF - wF) != 0 else 1e-12

        W = np.ones(self.pop_size)
        for rank, idx in enumerate(order):
            r = self._rng.random()
            if rank < self.pop_size / 2:
                W[idx] = 1 + r * \
                    np.log10((bF - self._fitness[idx]) / denom + 1)
            else:
                W[idx] = 1 - r * \
                    np.log10((bF - self._fitness[idx]) / denom + 1)
        return W

    def _adaptive_z(self, t: int) -> float:
        """
        Adaptive oscillation probability.

        Starts at a higher value to encourage exploration in early
        iterations, then decays linearly to the configured base ``self.z``.
        """
        z_adaptive = self.z + (0.15 - self.z) * \
            max(0, 1 - 2 * t / self.max_iter)
        return max(z_adaptive, 0.01)

    def _update_agent(
        self, i: int, w_i: float, a: float, vc: float, z_eff: float
    ) -> None:
        """
        Update the position of a single agent.

        Two possible update rules (Li et al. Eqs. 2.4, 2.6):

        * ``r < p`` → **approach food**  (exploitation toward best)
        * ``r >= p`` → **wrap food**      (contraction around self)

        With probability ``z_eff`` → random reinitialisation (exploration).
        """
        if self._rng.random() < z_eff:
            self._X[i] = self._rng.uniform(self.lb, self.ub, size=self.dim)
            return

        p = np.tanh(abs(self._fitness[i] - self._best_fit))

        if self._rng.random() < p:
            # Approach food
            idx_a, idx_b = self._rng.choice(
                self.pop_size, size=2, replace=False)
            vb = self._rng.uniform(-a, a, size=self.dim)
            self._X[i] = self._best_pos + vb * \
                (w_i * self._X[idx_a] - self._X[idx_b])
        else:
            # Wrap food
            vc_vec = self._rng.uniform(-vc, vc, size=self.dim)
            self._X[i] = vc_vec * self._X[i]

        np.clip(self._X[i], self.lb, self.ub, out=self._X[i])

    # ── Evaluation ────────────────────────────────────────────────

    def _evaluate_all(self, skip_indices: set[int] | None = None) -> None:
        """Evaluate fitness for all (or non-skipped) individuals."""
        if self._fitness is None:
            self._fitness = np.empty(self.pop_size)

        skip = skip_indices or set()
        for i in range(self.pop_size):
            if i in skip:
                continue
            self._fitness[i] = self.fitness_fn(self._X[i])
            self._n_eval += 1

        # Initialise best on first call
        if self._best_pos is None:
            best_idx = int(np.argmin(self._fitness))
            self._best_pos = self._X[best_idx].copy()
            self._best_fit = float(self._fitness[best_idx])

    # ── Restart mechanism ─────────────────────────────────────────

    def _restart_worst_half(self, order: np.ndarray) -> None:
        """
        Reinitialise the worst half of the population around the best
        solution with small random perturbations.
        """
        worst_indices = order[self.pop_size // 2:]
        for idx in worst_indices:
            noise = self._rng.uniform(-0.2, 0.2,
                                      size=self.dim) * (self.ub - self.lb)
            self._X[idx] = np.clip(self._best_pos + noise, self.lb, self.ub)
            self._fitness[idx] = self.fitness_fn(self._X[idx])
            self._n_eval += 1
        self._n_restarts += 1

    # ── Helpers ───────────────────────────────────────────────────

    def _param_snapshot(self) -> dict:
        return {
            "pop_size": self.pop_size,
            "max_iter": self.max_iter,
            "dim": self.dim,
            "lb": self.lb.tolist() if hasattr(self.lb, "__len__") else self.lb,
            "ub": self.ub.tolist() if hasattr(self.ub, "__len__") else self.ub,
            "elite_size": self.elite_size,
            "stall_limit": self.stall_limit,
            "z": self.z,
            "seed": self.seed,
        }

    def __repr__(self) -> str:
        return (
            f"SlimeMouldAlgorithm(dim={self.dim}, pop={self.pop_size}, "
            f"iter={self.max_iter}, seed={self.seed})"
        )


# ====================================================================
# Quick demo — sphere function
# ====================================================================

if __name__ == "__main__":

    def sphere(x: np.ndarray) -> float:
        """Quadratic sphere function (minimum = 0 at origin)."""
        return float(np.sum(x ** 2))

    sma = SlimeMouldAlgorithm(
        fitness_fn=sphere,
        dim=30,
        pop_size=30,
        max_iter=200,
        lb=-100.0,
        ub=100.0,
        seed=42,
    )

    result = sma.run()
    print("Slime Mould Algorithm — demo on sphere function (dim=30)")
    print(f"  Best fitness: {result.best_fitness:.6e}")
    print(f"  Evaluations:  {result.n_evaluations}")
    print(f"  Restarts:     {result.n_restarts}")
    print(f"  Runtime:      {result.runtime_sec:.4f}s")
    print(f"  Best pos (first 5 dims): {result.best_position[:5]}")
