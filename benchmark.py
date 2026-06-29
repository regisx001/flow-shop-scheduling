"""
PFSP Benchmark: NEH vs Slime Mould Algorithm vs Grey Wolf Optimizer
=====================================================================

Compares three methods on the Permutation Flow Shop Scheduling Problem:

    1. NEH  — constructive heuristic (Nawaz, Enscore, Ham, 1983)
    2. SMA  — Slime Mould Algorithm (Li et al., 2020)  + local search
    3. GWO  — Grey Wolf Optimizer (Mirjalili et al., 2014) + local search

All metaheuristics use LOV (Largest Order Value) decoding to map
continuous positions to job permutations.

Author: ZARQI Ezzoubair
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass

# Generic optimisers
from sma import SlimeMouldAlgorithm
from gwo import GreyWolfOptimizer

# PFSP-specific utilities
from pfsp import (
    compute_makespan,
    decode_lov,
    neh_heuristic,
    local_search_insertion,
    generate_instance,
)


# ====================================================================
# Generic PFSP solver (works with any continuous optimiser class)
# ====================================================================

@dataclass
class RunResult:
    """Result of a single solver run on a PFSP instance."""
    method: str
    makespan: float
    runtime_sec: float
    n_evaluations: int
    convergence: np.ndarray | None = None   # only for metaheuristics
    ls_improvement: float = 0.0

    def summary(self) -> str:
        return (f"Cmax={self.makespan:.1f}  "
                f"evals={self.n_evaluations}  "
                f"time={self.runtime_sec:.3f}s")


def solve_with_algo(
    algo_class: type,
    processing_times: np.ndarray,
    pop_size: int = 30,
    max_iter: int = 200,
    seed: int | None = None,
    use_local_search: bool = True,
) -> RunResult:
    """
    Solve a PFSP instance with any continuous optimiser class
    (SlimeMouldAlgorithm or GreyWolfOptimizer) via LOV decoding.

    The optimiser starts from a **random** initial population (no NEH
    seeding) so we can measure genuine optimisation performance against
    the NEH baseline.

    Parameters
    ----------
    algo_class : type
        Either ``SlimeMouldAlgorithm`` or ``GreyWolfOptimizer``.
    processing_times : np.ndarray, shape (n_jobs, n_machines)
    pop_size, max_iter : int
        Shared metaheuristic parameters.
    seed : int or None
        Random seed.
    use_local_search : bool
        Polish the final solution with insertion local search.

    Returns
    -------
    RunResult
    """
    n_jobs = processing_times.shape[0]

    # --- fitness function for both SMA and GWO ---
    def fitness_fn(position: np.ndarray) -> float:
        perm = decode_lov(position)
        return compute_makespan(perm, processing_times)

    # --- run optimiser (random init, no NEH seeding) ---
    algo = algo_class(
        fitness_fn=fitness_fn,
        dim=n_jobs,
        pop_size=pop_size,
        max_iter=max_iter,
        lb=0.0,
        ub=1.0,
        seed=seed,
    )

    t0 = time.time()
    result = algo.run()               # pure random initialisation
    runtime = time.time() - t0

    best_perm = decode_lov(result.best_position)
    best_fit = result.best_fitness

    # --- local search ---
    ls_improv = 0.0
    if use_local_search:
        cmax_before = best_fit
        best_perm, best_fit = local_search_insertion(best_perm, processing_times)
        ls_improv = ((cmax_before - best_fit) / cmax_before * 100
                     if cmax_before > 0 else 0.0)

    method_name = "SMA" if algo_class is SlimeMouldAlgorithm else "GWO"

    return RunResult(
        method=method_name,
        makespan=best_fit,
        runtime_sec=runtime,
        n_evaluations=result.n_evaluations,
        convergence=result.convergence,
        ls_improvement=ls_improv,
    )


# ====================================================================
# Benchmark runner
# ====================================================================

@dataclass
class InstanceStats:
    """Aggregated statistics for one instance × one method."""
    method: str
    best: float
    mean: float
    std: float
    worst: float
    avg_eval: float
    avg_time: float
    gap_best_vs_neh: float = 0.0      # percentage
    gap_mean_vs_neh: float = 0.0      # percentage


def run_benchmark(
    instances: list[tuple[str, int, int]] | None = None,
    pop_size: int = 30,
    max_iter: int = 200,
    n_trials: int = 5,
    instance_seed: int = 1,
    save_plot: bool = True,
) -> None:
    """
    Run the full benchmark: NEH vs SMA vs GWO on multiple PFSP instances.

    Metaheuristics start from **random** initial positions — no NEH
    seeding — so the gap vs NEH reflects genuine optimisation capability.

    Parameters
    ----------
    instances : list of (label, n_jobs, n_machines), optional
    pop_size, max_iter : int
        Base metaheuristic parameters (auto-scaled for large instances).
    n_trials : int
        Number of independent runs per method per instance.
    instance_seed : int
        Seed for generating the Taillard-style instances.
    save_plot : bool
        Whether to save a convergence + box-plot figure.
    """
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if instances is None:
        instances = [
            ("30×10",   30, 10),
            ("50×10",   50, 10),
            ("50×20",   50, 20),
            ("100×10", 100, 10),
            ("100×20", 100, 20),
        ]

    np.set_printoptions(precision=1, suppress=True)

    # ── Header ────────────────────────────────────────────────────
    print("=" * 78)
    print("  PFSP Benchmark: NEH vs SMA vs GWO  (random init, no seeding)")
    print(f"  Base Pop={pop_size}  Iter={max_iter}  Trials={n_trials}")
    print("=" * 78)

    all_stats: dict[str, list[InstanceStats]] = {}  # instance_label → stats

    for label, n_jobs, n_machines in instances:
        # Scale pop_size / max_iter with instance size
        scale = max(1, n_jobs // 30)
        scaled_pop = pop_size * scale
        scaled_iter = max_iter * scale

        print(f"\n{'─' * 78}")
        print(f"  Instance: {n_jobs} jobs × {n_machines} machines  "
              f"(pop={scaled_pop}, iter={scaled_iter})")
        print(f"{'─' * 78}")

        pt = generate_instance(n_jobs, n_machines, seed=instance_seed)
        all_stats[label] = []

        # ── NEH baseline (deterministic, run once) ─────────────────
        t0 = time.time()
        neh_seq, neh_cmax = neh_heuristic(pt)
        neh_time = time.time() - t0

        neh_stat = InstanceStats(
            method="NEH",
            best=neh_cmax, mean=neh_cmax, std=0.0, worst=neh_cmax,
            avg_eval=0, avg_time=neh_time,
        )
        all_stats[label].append(neh_stat)
        print(f"  {'NEH':<6}  →  Cmax = {neh_cmax:>10.1f}  "
              f"({neh_time:.3f}s)")

        # ── SMA ───────────────────────────────────────────────────
        sma_vals, sma_evals, sma_times = [], [], []
        for s in range(n_trials):
            r = solve_with_algo(SlimeMouldAlgorithm, pt,
                                pop_size=scaled_pop, max_iter=scaled_iter,
                                seed=s, use_local_search=False)
            sma_vals.append(r.makespan)
            sma_evals.append(r.n_evaluations)
            sma_times.append(r.runtime_sec)

        sma_arr = np.array(sma_vals)
        gap_sma = (sma_arr.min() - neh_cmax) / neh_cmax * 100
        sma_stat = InstanceStats(
            method="SMA",
            best=sma_arr.min(), mean=sma_arr.mean(),
            std=sma_arr.std(), worst=sma_arr.max(),
            avg_eval=np.mean(sma_evals), avg_time=np.mean(sma_times),
            gap_best_vs_neh=gap_sma,
            gap_mean_vs_neh=(sma_arr.mean() - neh_cmax) / neh_cmax * 100,
        )
        all_stats[label].append(sma_stat)
        print(f"  {'SMA':<6}  →  best={sma_arr.min():>8.1f}  "
              f"mean={sma_arr.mean():>8.1f}  std={sma_arr.std():>5.1f}  "
              f"(gap best: {gap_sma:+.2f}%)")

        # ── GWO ───────────────────────────────────────────────────
        gwo_vals, gwo_evals, gwo_times = [], [], []
        for s in range(n_trials):
            r = solve_with_algo(GreyWolfOptimizer, pt,
                                pop_size=scaled_pop, max_iter=scaled_iter,
                                seed=s, use_local_search=False)
            gwo_vals.append(r.makespan)
            gwo_evals.append(r.n_evaluations)
            gwo_times.append(r.runtime_sec)

        gwo_arr = np.array(gwo_vals)
        gap_gwo = (gwo_arr.min() - neh_cmax) / neh_cmax * 100
        gwo_stat = InstanceStats(
            method="GWO",
            best=gwo_arr.min(), mean=gwo_arr.mean(),
            std=gwo_arr.std(), worst=gwo_arr.max(),
            avg_eval=np.mean(gwo_evals), avg_time=np.mean(gwo_times),
            gap_best_vs_neh=gap_gwo,
            gap_mean_vs_neh=(gwo_arr.mean() - neh_cmax) / neh_cmax * 100,
        )
        all_stats[label].append(gwo_stat)
        print(f"  {'GWO':<6}  →  best={gwo_arr.min():>8.1f}  "
              f"mean={gwo_arr.mean():>8.1f}  std={gwo_arr.std():>5.1f}  "
              f"(gap best: {gap_gwo:+.2f}%)")

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("  SUMMARY TABLE — Best Cmax (gap vs NEH)")
    print(f"{'=' * 78}")
    header = (f"  {'Instance':<8} {'NEH':>9} "
              f"{'SMA':>9} {'Δ %':>6} "
              f"{'GWO':>9} {'Δ %':>6} "
              f"{'Winner':<8}")
    print(header)
    print(f"  {'─' * 66}")

    for label in [inst[0] for inst in instances]:
        stats = {s.method: s for s in all_stats[label]}
        neh_v = stats["NEH"].best
        sma_b = stats["SMA"].best
        gwo_b = stats["GWO"].best
        gap_s = stats["SMA"].gap_best_vs_neh
        gap_g = stats["GWO"].gap_best_vs_neh

        # Determine winner
        best_all = min(neh_v, sma_b, gwo_b)
        winner = next(m for m, v in
                      [("NEH", neh_v), ("SMA", sma_b), ("GWO", gwo_b)]
                      if v == best_all)

        print(f"  {label:<8} {neh_v:>9.1f} "
              f"{sma_b:>9.1f} {gap_s:>+5.1f}% "
              f"{gwo_b:>9.1f} {gap_g:>+5.1f}% "
              f"{winner:<8}")

    # ── Convergence plot ──────────────────────────────────────────
    if save_plot:
        _plot_results(instances, pop_size, max_iter, n_trials,
                      instance_seed)


def _plot_results(
    instances: list[tuple[str, int, int]],
    pop_size: int,
    max_iter: int,
    n_trials: int,
    instance_seed: int,
) -> None:
    """Generate convergence + box-plot figure for the first instance."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (matplotlib not available — skipping plot)")
        return

    label0, nj0, nm0 = instances[0]
    scale = max(1, nj0 // 30)
    scaled_pop = pop_size * scale
    scaled_iter = max_iter * scale

    pt0 = generate_instance(nj0, nm0, seed=instance_seed)
    neh_seq0, neh_cmax0 = neh_heuristic(pt0)

    # Collect multi-trial results for box plot
    sma_trials = []
    gwo_trials = []
    for s in range(n_trials):
        sma_trials.append(
            solve_with_algo(SlimeMouldAlgorithm, pt0,
                            pop_size=scaled_pop, max_iter=scaled_iter,
                            seed=s, use_local_search=True).makespan)
        gwo_trials.append(
            solve_with_algo(GreyWolfOptimizer, pt0,
                            pop_size=scaled_pop, max_iter=scaled_iter,
                            seed=s, use_local_search=True).makespan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: convergence (single representative run) ─────────────
    ax = axes[0]
    r_sma = solve_with_algo(SlimeMouldAlgorithm, pt0,
                            pop_size=scaled_pop, max_iter=scaled_iter,
                            seed=42, use_local_search=False)
    r_gwo = solve_with_algo(GreyWolfOptimizer, pt0,
                            pop_size=scaled_pop, max_iter=scaled_iter,
                            seed=42, use_local_search=False)

    ax.plot(r_sma.convergence[:scaled_iter],
            label=f"SMA (final: {r_sma.makespan:.0f})",
            color="#2196F3", linewidth=1.6)
    ax.plot(r_gwo.convergence[:scaled_iter],
            label=f"GWO (final: {r_gwo.makespan:.0f})",
            color="#9C27B0", linewidth=1.6)
    ax.axhline(neh_cmax0, color="#4CAF50", linestyle="--", linewidth=1.2,
               label=f"NEH ({neh_cmax0:.0f})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Makespan (Cmax)")
    ax.set_title(f"Convergence — {label0} (pop {scaled_pop}, {scaled_iter} it)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Right: box-plot over N trials ─────────────────────────────
    ax = axes[1]
    positions = [1, 2, 3]
    data = [[neh_cmax0] * n_trials, sma_trials, gwo_trials]
    colors = ["#4CAF50", "#2196F3", "#9C27B0"]

    bp = ax.boxplot(data, positions=positions, widths=0.5,
                    patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_xticklabels(["NEH", "SMA", "GWO"])
    ax.set_ylabel("Makespan (Cmax)")
    ax.set_title(f"Distribution over {n_trials} runs — {label0}")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("benchmark_pfsp.png", dpi=150, bbox_inches="tight")
    print(f"\n  ✓ Plot saved → benchmark_pfsp.png")


# ====================================================================
# Entry point
# ====================================================================

if __name__ == "__main__":
    run_benchmark()
