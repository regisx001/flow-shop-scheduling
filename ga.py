"""
Genetic Algorithm (GA)
=======================

A pure, generic implementation of a real-coded Genetic Algorithm for
continuous optimisation (minimisation).

Operators:
    * Tournament selection (binary)
    * BLX-α crossover (blend crossover)
    * Gaussian mutation (adaptive)
    * Elitism

This module provides a single class ``GeneticAlgorithm`` that works on any
continuous optimisation problem.  Users supply a fitness function and the
algorithm evolves a population of candidate solutions.

Reference:
    Eshelman, L. J., & Schaffer, J. D. (1993). Real-coded genetic algorithms
    and interval-schemata. Foundations of Genetic Algorithms, 2, 187-202.

Author: ZARQI Ezzoubair
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass, field


# ====================================================================
# Result container
# ====================================================================

@dataclass
class GAResult:
    """Holds the output of a single GA run."""

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

    params: dict = field(default_factory=dict)
    """Snapshot of the algorithm parameters used for this run."""

    def summary(self) -> str:
        return (f"f*={self.best_fitness:.6e}  "
                f"evals={self.n_evaluations}  "
                f"time={self.runtime_sec:.4f}s")


# ====================================================================
# Genetic Algorithm
# ====================================================================

class GeneticAlgorithm:
    """
    Genetic Algorithm (GA) — a population-based metaheuristic for
    continuous global optimisation (minimisation).

    Core mechanics
    --------------
    Each generation:

    1. **Elitism** — the top ``elite_size`` individuals survive unchanged.
    2. **Selection** — binary tournament selects parents.
    3. **Crossover** — BLX-α blends two parents into two children.
    4. **Mutation** — Gaussian noise is added with adaptive probability.
    5. **Replacement** — children replace the non-elite part of the population.

    Parameters
    ----------
    fitness_fn : callable
        Function ``f(x) -> float`` where ``x`` is a 1-D numpy array of
        shape ``(dim,)``. The algorithm **minimises** this function.
    dim : int
        Dimensionality of the search space.
    pop_size : int
        Number of individuals (population size). Default 30.
    max_iter : int
        Maximum number of generations. Default 200.
    lb : float or np.ndarray
        Lower bound(s) of the search space. If float, same bound for all
        dimensions. Default ``0.0``.
    ub : float or np.ndarray
        Upper bound(s) of the search space. Default ``1.0``.
    crossover_rate : float
        Probability of crossover for each selected pair. Default ``0.9``.
    mutation_rate : float
        Base mutation probability. Default ``0.1``.
    elite_size : int
        Number of top individuals preserved unchanged each generation.
        Default ``2``.
    tournament_size : int
        Number of individuals competing in each tournament. Default ``2``.
    alpha : float
        BLX-α expansion factor (0 = no expansion, 0.5 = moderate).
        Default ``0.3``.
    seed : int or None
        Random seed for reproducibility. Default ``None``.

    Example
    -------
    >>> import numpy as np
    >>> def sphere(x):
    ...     return float(np.sum(x ** 2))
    ...
    >>> ga = GeneticAlgorithm(sphere, dim=10, pop_size=30, max_iter=100,
    ...                       lb=-5.0, ub=5.0, seed=42)
    >>> result = ga.run()
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
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        elite_size: int = 2,
        tournament_size: int = 2,
        alpha: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = np.broadcast_to(lb, (dim,)).astype(float)
        self.ub = np.broadcast_to(ub, (dim,)).astype(float)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = min(elite_size, pop_size)
        self.tournament_size = tournament_size
        self.alpha_blx = alpha
        self.seed = seed

        # Internal state (set by run())
        self._rng: np.random.Generator | None = None
        self._X: np.ndarray | None = None           # (pop_size, dim)
        self._fitness: np.ndarray | None = None      # (pop_size,)
        self._best_pos: np.ndarray | None = None
        self._best_fit: float = np.inf
        self._convergence: np.ndarray | None = None
        self._n_eval: int = 0

    # ── Public API ────────────────────────────────────────────────

    def run(self) -> GAResult:
        """
        Execute the GA optimisation.

        Returns
        -------
        GAResult
            Container with the best solution, convergence history,
            and runtime statistics.
        """
        self._rng = np.random.default_rng(self.seed)
        t_start = time.time()

        # 1. Initialise population
        self._X = self._rng.uniform(
            self.lb, self.ub, size=(self.pop_size, self.dim)
        ).astype(float)

        # 2. Evaluate initial population
        self._fitness = np.empty(self.pop_size)
        for i in range(self.pop_size):
            self._fitness[i] = self.fitness_fn(self._X[i])
            self._n_eval += 1

        # Track best
        best_idx = int(np.argmin(self._fitness))
        self._best_fit = float(self._fitness[best_idx])
        self._best_pos = self._X[best_idx].copy()

        self._convergence = np.zeros(self.max_iter)
        self._convergence[0] = self._best_fit

        # 3. Main loop
        for gen in range(1, self.max_iter):
            # --- Build next generation ---
            children = self._make_children()

            # --- Evaluate children ---
            child_fitness = np.empty(len(children))
            for i, child in enumerate(children):
                child_fitness[i] = self.fitness_fn(child)
                self._n_eval += 1

            # --- Replace population ---
            self._replace_population(children, child_fitness)

            # --- Track best ---
            gen_best_idx = int(np.argmin(self._fitness))
            gen_best_fit = float(self._fitness[gen_best_idx])
            if gen_best_fit < self._best_fit:
                self._best_fit = gen_best_fit
                self._best_pos = self._X[gen_best_idx].copy()

            self._convergence[gen] = self._best_fit

        runtime = time.time() - t_start

        return GAResult(
            best_position=self._best_pos.copy(),
            best_fitness=self._best_fit,
            convergence=self._convergence.copy(),
            runtime_sec=runtime,
            n_evaluations=self._n_eval,
            params=self._param_snapshot(),
        )

    # ── GA operators ──────────────────────────────────────────────

    def _make_children(self) -> list[np.ndarray]:
        """Generate new offspring via selection, crossover, and mutation."""
        num_children = self.pop_size - self.elite_size
        children = []

        while len(children) < num_children:
            # Select two parents via tournament
            p1 = self._tournament_select()
            p2 = self._tournament_select()

            # Crossover (with probability crossover_rate)
            if self._rng.random() < self.crossover_rate:
                c1, c2 = self._blx_alpha_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            self._mutate(c1)
            self._mutate(c2)

            # Clip to bounds
            np.clip(c1, self.lb, self.ub, out=c1)
            np.clip(c2, self.lb, self.ub, out=c2)

            children.append(c1)
            if len(children) < num_children:
                children.append(c2)

        return children

    def _tournament_select(self) -> np.ndarray:
        """Binary tournament selection: pick the best of k random individuals."""
        indices = self._rng.choice(
            self.pop_size, size=self.tournament_size, replace=False)
        best_idx = indices[np.argmin(self._fitness[indices])]
        return self._X[best_idx].copy()

    def _blx_alpha_crossover(
        self, p1: np.ndarray, p2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        BLX-α crossover: blend two parents with an expansion factor α.

        For each dimension d:
            low  = min(p1[d], p2[d]) - α * |p1[d] - p2[d]|
            high = max(p1[d], p2[d]) + α * |p1[d] - p2[d]|
            child[d] ~ U(low, high)

        Returns two children.
        """
        diff = np.abs(p1 - p2)
        low = np.minimum(p1, p2) - self.alpha_blx * diff
        high = np.maximum(p1, p2) + self.alpha_blx * diff

        c1 = self._rng.uniform(low, high, size=self.dim)
        c2 = self._rng.uniform(low, high, size=self.dim)
        return c1, c2

    def _mutate(self, individual: np.ndarray) -> None:
        """
        Gaussian mutation with adaptive step size.

        Each gene is perturbed with probability mutation_rate.
        The step size scales with the search range.
        """
        sigma = 0.1 * (self.ub - self.lb)  # 10% of range
        mask = self._rng.random(self.dim) < self.mutation_rate
        noise = self._rng.normal(0, sigma)
        individual[mask] += noise[mask]

    def _replace_population(
        self, children: list[np.ndarray], child_fitness: np.ndarray
    ) -> None:
        """
        Replace the non-elite portion of the population with children.

        Elite individuals (best fitness) are preserved unchanged.
        """
        # Sort current population by fitness
        order = np.argsort(self._fitness)

        # Preserve elite
        elite_indices = order[:self.elite_size]
        X_elite = self._X[elite_indices].copy()
        F_elite = self._fitness[elite_indices].copy()

        # Replace non-elite with children
        non_elite_indices = order[self.elite_size:]
        for i, idx in enumerate(non_elite_indices):
            if i < len(children):
                self._X[idx] = children[i]
                self._fitness[idx] = child_fitness[i]

        # Restore elite
        self._X[elite_indices] = X_elite
        self._fitness[elite_indices] = F_elite

    # ── Helpers ───────────────────────────────────────────────────

    def _param_snapshot(self) -> dict:
        return {
            "pop_size": self.pop_size,
            "max_iter": self.max_iter,
            "dim": self.dim,
            "lb": self.lb.tolist() if hasattr(self.lb, "__len__") else self.lb,
            "ub": self.ub.tolist() if hasattr(self.ub, "__len__") else self.ub,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "elite_size": self.elite_size,
            "seed": self.seed,
        }

    def __repr__(self) -> str:
        return (
            f"GeneticAlgorithm(dim={self.dim}, pop={self.pop_size}, "
            f"gen={self.max_iter}, seed={self.seed})"
        )


# ====================================================================
# Quick demo — sphere function
# ====================================================================

if __name__ == "__main__":

    def sphere(x: np.ndarray) -> float:
        """Quadratic sphere function (minimum = 0 at origin)."""
        return float(np.sum(x ** 2))

    ga = GeneticAlgorithm(
        fitness_fn=sphere,
        dim=30,
        pop_size=30,
        max_iter=200,
        lb=-100.0,
        ub=100.0,
        seed=42,
    )

    result = ga.run()
    print("Genetic Algorithm — demo on sphere function (dim=30)")
    print(f"  Best fitness: {result.best_fitness:.6e}")
    print(f"  Evaluations:  {result.n_evaluations}")
    print(f"  Runtime:      {result.runtime_sec:.4f}s")
    print(f"  Best pos (first 5 dims): {result.best_position[:5]}")
