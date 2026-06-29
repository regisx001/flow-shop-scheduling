"""
PFSP Benchmark: NEH vs SMA vs GWO vs GA
=========================================

Compares four methods on the Permutation Flow Shop Scheduling Problem:

    1. NEH  — constructive heuristic (Nawaz, Enscore, Ham, 1983)
    2. SMA  — Slime Mould Algorithm (Li et al., 2020)
    3. GWO  — Grey Wolf Optimizer (Mirjalili et al., 2014)
    4. GA   — Genetic Algorithm (real-coded, BLX-α crossover)

All metaheuristics use LOV (Largest Order Value) decoding to map
continuous positions to job permutations.

Author: ZARQI Ezzoubair
"""

import numpy as np
import time
from dataclasses import dataclass

# Generic optimisers
from sma import SlimeMouldAlgorithm
from gwo import GreyWolfOptimizer
from ga import GeneticAlgorithm

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
    iter_to_neh: int = -1                    # iteration where NEH quality reached
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

    # --- Iteration where NEH-quality was reached ---
    neh_cmax = compute_makespan(neh_heuristic(
        processing_times)[0], processing_times)
    iter_to_neh = -1
    if result.convergence is not None and len(result.convergence) > 0:
        hit = np.where(result.convergence <= neh_cmax)[0]
        if len(hit) > 0:
            iter_to_neh = int(hit[0]) + 1  # 1-indexed

    # --- local search ---
    ls_improv = 0.0
    if use_local_search:
        cmax_before = best_fit
        best_perm, best_fit = local_search_insertion(
            best_perm, processing_times)
        ls_improv = ((cmax_before - best_fit) / cmax_before * 100
                     if cmax_before > 0 else 0.0)

    method_name = (
        "SMA" if algo_class is SlimeMouldAlgorithm
        else "GWO" if algo_class is GreyWolfOptimizer
        else "GA"
    )

    return RunResult(
        method=method_name,
        makespan=best_fit,
        runtime_sec=runtime,
        n_evaluations=result.n_evaluations,
        convergence=result.convergence,
        iter_to_neh=iter_to_neh,
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
    # avg iters to reach NEH quality (-1 = never)
    avg_iter_to_neh: float = 0.0
    gap_best_vs_neh: float = 0.0      # percentage
    gap_mean_vs_neh: float = 0.0      # percentage


def run_benchmark(
    instances: list[tuple[str, int, int]] | None = None,
    pop_size: int = 10,
    max_iter: int = 20,
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
    print("  PFSP Benchmark: NEH vs SMA vs GWO vs GA  (random init)")
    print(f"  Base Pop={pop_size}  Iter={max_iter}  Trials={n_trials}")
    print("=" * 78)

    all_stats: dict[str, list[InstanceStats]] = {}  # instance_label → stats

    for idx, (label, n_jobs, n_machines) in enumerate(instances):
        # Scale pop_size / max_iter with instance size
        scale = max(1, n_jobs // 30)
        scaled_pop = pop_size * scale
        scaled_iter = max_iter * scale

        # Use different seed per instance for diversity
        inst_seed = instance_seed + idx * 100

        print(f"\n{'─' * 78}")
        print(f"  Instance: {n_jobs} jobs × {n_machines} machines  "
              f"(seed={inst_seed}, pop={scaled_pop}, iter={scaled_iter})")
        print(f"{'─' * 78}")

        pt = generate_instance(n_jobs, n_machines, seed=inst_seed)
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

        # ── Random baseline (lower bound) ─────────────────────────
        rand_vals = []
        for _ in range(n_trials * 5):  # more samples for stable baseline
            rp = np.random.default_rng(_).permutation(n_jobs)
            rand_vals.append(compute_makespan(rp, pt))
        rand_best = min(rand_vals)
        rand_mean = np.mean(rand_vals)
        rand_stat = InstanceStats(
            method="RAND",
            best=rand_best, mean=rand_mean, std=np.std(rand_vals),
            worst=max(rand_vals), avg_eval=0, avg_time=0,
        )
        all_stats[label].append(rand_stat)
        print(f"  {'RAND':<6}  →  best={rand_best:>8.1f}  "
              f"mean={rand_mean:>8.1f}  (lower bound)")

        # ── SMA ───────────────────────────────────────────────────
        sma_vals, sma_evals, sma_times, sma_iters = [], [], [], []
        for s in range(n_trials):
            r = solve_with_algo(SlimeMouldAlgorithm, pt,
                                pop_size=scaled_pop, max_iter=scaled_iter,
                                seed=s, use_local_search=False)
            sma_vals.append(r.makespan)
            sma_evals.append(r.n_evaluations)
            sma_times.append(r.runtime_sec)
            sma_iters.append(r.iter_to_neh)

        sma_arr = np.array(sma_vals)
        gap_sma = (sma_arr.min() - neh_cmax) / neh_cmax * 100
        sma_stat = InstanceStats(
            method="SMA",
            best=sma_arr.min(), mean=sma_arr.mean(),
            std=sma_arr.std(), worst=sma_arr.max(),
            avg_eval=np.mean(sma_evals), avg_time=np.mean(sma_times),
            avg_iter_to_neh=np.mean(sma_iters),
            gap_best_vs_neh=gap_sma,
            gap_mean_vs_neh=(sma_arr.mean() - neh_cmax) / neh_cmax * 100,
        )
        all_stats[label].append(sma_stat)
        print(f"  {'SMA':<6}  →  best={sma_arr.min():>8.1f}  "
              f"mean={sma_arr.mean():>8.1f}  std={sma_arr.std():>5.1f}  "
              f"(gap best: {gap_sma:+.2f}%)")

        # ── GWO ───────────────────────────────────────────────────
        gwo_vals, gwo_evals, gwo_times, gwo_iters = [], [], [], []
        for s in range(n_trials):
            r = solve_with_algo(GreyWolfOptimizer, pt,
                                pop_size=scaled_pop, max_iter=scaled_iter,
                                seed=s, use_local_search=False)
            gwo_vals.append(r.makespan)
            gwo_evals.append(r.n_evaluations)
            gwo_times.append(r.runtime_sec)
            gwo_iters.append(r.iter_to_neh)

        gwo_arr = np.array(gwo_vals)
        gap_gwo = (gwo_arr.min() - neh_cmax) / neh_cmax * 100
        gwo_stat = InstanceStats(
            method="GWO",
            best=gwo_arr.min(), mean=gwo_arr.mean(),
            std=gwo_arr.std(), worst=gwo_arr.max(),
            avg_eval=np.mean(gwo_evals), avg_time=np.mean(gwo_times),
            avg_iter_to_neh=np.mean(gwo_iters),
            gap_best_vs_neh=gap_gwo,
            gap_mean_vs_neh=(gwo_arr.mean() - neh_cmax) / neh_cmax * 100,
        )
        all_stats[label].append(gwo_stat)
        print(f"  {'GWO':<6}  →  best={gwo_arr.min():>8.1f}  "
              f"mean={gwo_arr.mean():>8.1f}  std={gwo_arr.std():>5.1f}  "
              f"(gap best: {gap_gwo:+.2f}%)")

        # ── GA ────────────────────────────────────────────────────
        ga_vals, ga_evals, ga_times, ga_iters = [], [], [], []
        for s in range(n_trials):
            r = solve_with_algo(GeneticAlgorithm, pt,
                                pop_size=scaled_pop, max_iter=scaled_iter,
                                seed=s, use_local_search=False)
            ga_vals.append(r.makespan)
            ga_evals.append(r.n_evaluations)
            ga_times.append(r.runtime_sec)
            ga_iters.append(r.iter_to_neh)

        ga_arr = np.array(ga_vals)
        gap_ga = (ga_arr.min() - neh_cmax) / neh_cmax * 100
        ga_stat = InstanceStats(
            method="GA",
            best=ga_arr.min(), mean=ga_arr.mean(),
            std=ga_arr.std(), worst=ga_arr.max(),
            avg_eval=np.mean(ga_evals), avg_time=np.mean(ga_times),
            avg_iter_to_neh=np.mean(ga_iters),
            gap_best_vs_neh=gap_ga,
            gap_mean_vs_neh=(ga_arr.mean() - neh_cmax) / neh_cmax * 100,
        )
        all_stats[label].append(ga_stat)
        print(f"  {'GA':<6}  →  best={ga_arr.min():>8.1f}  "
              f"mean={ga_arr.mean():>8.1f}  std={ga_arr.std():>5.1f}  "
              f"(gap best: {gap_ga:+.2f}%)")

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print("  SUMMARY — NEH is the upper bound, RAND is the lower bound")
    print(f"  Metaheuristics start from random positions → convergence analysis")
    print(f"{'=' * 110}")
    header = (f"  {'Inst':<8} {'NEH':>7} {'RAND':>7}  "
              f"{'SMA':>7} {'±std':>5} {'Δ%':>6} {'hit':>4}  "
              f"{'GWO':>7} {'±std':>5} {'Δ%':>6} {'hit':>4}  "
              f"{'GA':>7} {'±std':>5} {'Δ%':>6} {'hit':>4}")
    print(header)
    print(f"  {'─' * 98}")

    for label in [inst[0] for inst in instances]:
        stats = {s.method: s for s in all_stats[label]}
        neh_v = stats["NEH"].best
        rand_v = stats["RAND"].best
        sma_m = stats["SMA"].mean
        sma_s = stats["SMA"].std
        gwo_m = stats["GWO"].mean
        gwo_s = stats["GWO"].std
        ga_m = stats["GA"].mean
        ga_s = stats["GA"].std
        gap_s = stats["SMA"].gap_mean_vs_neh
        gap_g = stats["GWO"].gap_mean_vs_neh
        gap_ga = stats["GA"].gap_mean_vs_neh
        sma_it = stats["SMA"].avg_iter_to_neh
        gwo_it = stats["GWO"].avg_iter_to_neh
        ga_it = stats["GA"].avg_iter_to_neh

        sma_hit = f"{sma_it:.0f}it" if sma_it > 0 else "—"
        gwo_hit = f"{gwo_it:.0f}it" if gwo_it > 0 else "—"
        ga_hit = f"{ga_it:.0f}it" if ga_it > 0 else "—"

        print(f"  {label:<8} {neh_v:>7.0f} {rand_v:>7.0f}  "
              f"{sma_m:>7.0f} {sma_s:>5.0f} {gap_s:>+5.1f}% {sma_hit:>4}  "
              f"{gwo_m:>7.0f} {gwo_s:>5.0f} {gap_g:>+5.1f}% {gwo_hit:>4}  "
              f"{ga_m:>7.0f} {ga_s:>5.0f} {gap_ga:>+5.1f}% {ga_hit:>4}")

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
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (matplotlib not available — skipping plot)")
        return

    label0, nj0, nm0 = instances[0]
    scale = max(1, nj0 // 30)
    scaled_pop = pop_size * scale
    scaled_iter = max_iter * scale

    inst_seed0 = instance_seed  # first instance uses base seed
    pt0 = generate_instance(nj0, nm0, seed=inst_seed0)
    neh_seq0, neh_cmax0 = neh_heuristic(pt0)

    # Collect multi-trial results for box plot
    sma_trials = []
    gwo_trials = []
    ga_trials = []
    rand_trials = []
    for s in range(n_trials):
        sma_trials.append(
            solve_with_algo(SlimeMouldAlgorithm, pt0,
                            pop_size=scaled_pop, max_iter=scaled_iter,
                            seed=s, use_local_search=False).makespan)
        gwo_trials.append(
            solve_with_algo(GreyWolfOptimizer, pt0,
                            pop_size=scaled_pop, max_iter=scaled_iter,
                            seed=s, use_local_search=False).makespan)
        ga_trials.append(
            solve_with_algo(GeneticAlgorithm, pt0,
                            pop_size=scaled_pop, max_iter=scaled_iter,
                            seed=s, use_local_search=False).makespan)
        rp = np.random.default_rng(s).permutation(nj0)
        rand_trials.append(compute_makespan(rp, pt0))

    rand_arr = np.array(rand_trials)
    rand_mean = rand_arr.mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: convergence (single representative run) ─────────────
    ax = axes[0]
    r_sma = solve_with_algo(SlimeMouldAlgorithm, pt0,
                            pop_size=scaled_pop, max_iter=scaled_iter,
                            seed=42, use_local_search=False)
    r_gwo = solve_with_algo(GreyWolfOptimizer, pt0,
                            pop_size=scaled_pop, max_iter=scaled_iter,
                            seed=42, use_local_search=False)
    r_ga = solve_with_algo(GeneticAlgorithm, pt0,
                           pop_size=scaled_pop, max_iter=scaled_iter,
                           seed=42, use_local_search=False)

    ax.plot(r_sma.convergence[:scaled_iter],
            label=f"SMA → {r_sma.makespan:.0f}",
            color="#2196F3", linewidth=1.6)
    ax.plot(r_gwo.convergence[:scaled_iter],
            label=f"GWO → {r_gwo.makespan:.0f}",
            color="#9C27B0", linewidth=1.6)
    ax.plot(r_ga.convergence[:scaled_iter],
            label=f"GA → {r_ga.makespan:.0f}",
            color="#FF5722", linewidth=1.6)
    ax.axhline(neh_cmax0, color="#4CAF50", linestyle="--", linewidth=1.2,
               label=f"NEH ({neh_cmax0:.0f})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Makespan (Cmax)")
    ax.set_title(
        f"Convergence — {label0} (pop {scaled_pop}, {scaled_iter} it)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Right: box-plot over N trials ─────────────────────────────
    ax = axes[1]
    positions = [1, 2, 3, 4, 5]
    data = [[neh_cmax0] * n_trials, sma_trials, gwo_trials,
            ga_trials, rand_trials]
    colors = ["#4CAF50", "#2196F3", "#9C27B0", "#FF5722", "#FF9800"]

    bp = ax.boxplot(data, positions=positions, widths=0.5,
                    patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_xticklabels(["NEH", "SMA", "GWO", "GA", "RAND"])
    ax.set_ylabel("Makespan (Cmax)")
    ax.set_title(f"Distribution over {n_trials} runs — {label0}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(neh_cmax0, color="#4CAF50",
               linestyle=":", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig("benchmark_pfsp.png", dpi=150, bbox_inches="tight")
    print(f"\n  ✓ Box-plot saved → benchmark_pfsp.png")

    # ── Convergence figure (one subplot per instance) ─────────────
    _plot_convergence(instances, pop_size, max_iter, n_trials,
                      instance_seed)


def _plot_convergence(
    instances: list[tuple[str, int, int]],
    pop_size: int,
    max_iter: int,
    n_trials: int,
    instance_seed: int,
) -> None:
    """Generate convergence curves for all instances (mean ± std)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.rcParams.update({"font.size": 9})
    n_inst = len(instances)
    cols = min(3, n_inst)
    rows = (n_inst + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows))
    if n_inst == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (label, nj, nm) in enumerate(instances):
        ax = axes[idx]
        scale = max(1, nj // 30)
        sp = pop_size * scale
        si = max_iter * scale
        inst_seed = instance_seed + idx * 100
        pt = generate_instance(nj, nm, seed=inst_seed)
        neh_seq, neh_cmax = neh_heuristic(pt)

        # Random baseline (use all trials)
        rand_vals = []
        for s in range(n_trials):
            rp = np.random.default_rng(s).permutation(nj)
            rand_vals.append(compute_makespan(rp, pt))
        rand_mean = np.mean(rand_vals)

        # Collect convergence curves (3 trials for speed)
        plot_trials = min(n_trials, 3)
        sma_curves = []
        for s in range(plot_trials):
            r = solve_with_algo(SlimeMouldAlgorithm, pt,
                                pop_size=sp, max_iter=si,
                                seed=s, use_local_search=False)
            if r.convergence is not None:
                sma_curves.append(r.convergence[:si])
        sma_curves = np.array(sma_curves)
        sma_mean = sma_curves.mean(axis=0)
        sma_std = sma_curves.std(axis=0)
        sma_best = sma_curves.min(axis=0)   # best-of-trials per iteration

        gwo_curves = []
        for s in range(plot_trials):
            r = solve_with_algo(GreyWolfOptimizer, pt,
                                pop_size=sp, max_iter=si,
                                seed=s, use_local_search=False)
            if r.convergence is not None:
                gwo_curves.append(r.convergence[:si])
        gwo_curves = np.array(gwo_curves)
        gwo_mean = gwo_curves.mean(axis=0)
        gwo_std = gwo_curves.std(axis=0)
        gwo_best = gwo_curves.min(axis=0)

        ga_curves = []
        for s in range(plot_trials):
            r = solve_with_algo(GeneticAlgorithm, pt,
                                pop_size=sp, max_iter=si,
                                seed=s, use_local_search=False)
            if r.convergence is not None:
                ga_curves.append(r.convergence[:si])
        ga_curves = np.array(ga_curves)
        ga_mean = ga_curves.mean(axis=0)
        ga_std = ga_curves.std(axis=0)
        ga_best = ga_curves.min(axis=0)

        iters = np.arange(1, si + 1)

        # Plot — solid = mean, dashed = best-of-trials
        ax.fill_between(iters, sma_mean - sma_std, sma_mean + sma_std,
                        alpha=0.12, color="#2196F3")
        ax.plot(iters, sma_mean, color="#2196F3", linewidth=1.6,
                label=f"SMA μ={sma_mean[-1]:.0f}")
        ax.plot(iters, sma_best, color="#2196F3", linewidth=0.8,
                linestyle="--", alpha=0.7,
                label=f"SMA best={sma_best[-1]:.0f}")

        ax.fill_between(iters, gwo_mean - gwo_std, gwo_mean + gwo_std,
                        alpha=0.12, color="#9C27B0")
        ax.plot(iters, gwo_mean, color="#9C27B0", linewidth=1.6,
                label=f"GWO μ={gwo_mean[-1]:.0f}")
        ax.plot(iters, gwo_best, color="#9C27B0", linewidth=0.8,
                linestyle="--", alpha=0.7,
                label=f"GWO best={gwo_best[-1]:.0f}")

        ax.fill_between(iters, ga_mean - ga_std, ga_mean + ga_std,
                        alpha=0.12, color="#FF5722")
        ax.plot(iters, ga_mean, color="#FF5722", linewidth=1.6,
                label=f"GA μ={ga_mean[-1]:.0f}")
        ax.plot(iters, ga_best, color="#FF5722", linewidth=0.8,
                linestyle="--", alpha=0.7,
                label=f"GA best={ga_best[-1]:.0f}")

        ax.axhline(neh_cmax, color="#4CAF50", linestyle="--", linewidth=1.2,
                   label=f"NEH ({neh_cmax:.0f})")
        ax.axhline(rand_mean, color="#FF9800", linestyle=":", linewidth=1.0,
                   alpha=0.6, label=f"RAND ({rand_mean:.0f})")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Makespan (Cmax)")
        ax.set_title(
            f"{label}  (pop={sp}, {plot_trials} trials — solid=mean, dash=best)")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_inst, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("benchmark_convergence.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Convergence plot saved → benchmark_convergence.png")


# ====================================================================
# Entry point
# ====================================================================

if __name__ == "__main__":
    run_benchmark()
