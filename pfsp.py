"""
Flow Shop Scheduling Problem (PFSP) — solved with the Slime Mould Algorithm
============================================================================

This module uses ``SlimeMouldAlgorithm`` (from ``sma.py``) to minimise the
makespan (Cmax) of a Permutation Flow Shop Scheduling Problem.

The SMA operates in a continuous space; the Largest Order Value (LOV) rule
maps each continuous vector to a valid job permutation via ``argsort``.

Reference for SMA:
    Li, S., Chen, H., Wang, M., Heidari, A. A., & Mirjalili, S. (2020).
    Slime mould algorithm: A new method for stochastic optimization.
    Future Generation Computer Systems, 111, 300-323.

Author: ZARQI Ezzoubair
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from sma import SlimeMouldAlgorithm, SMAResult


# ====================================================================
# 1. Makespan computation
# ====================================================================

def compute_makespan(permutation: np.ndarray,
                     processing_times: np.ndarray) -> float:
    """
    Compute the makespan (Cmax) of a job permutation on a flow shop.

    Parameters
    ----------
    permutation : np.ndarray, shape (n_jobs,)
        Order in which jobs are processed (same order on every machine).
    processing_times : np.ndarray, shape (n_jobs, n_machines)
        ``p[j, k]`` = processing time of job j on machine k.

    Returns
    -------
    float
        Makespan Cmax.
    """
    seq = processing_times[permutation]           # (n_jobs, n_machines)
    n_jobs, n_machines = seq.shape
    C = np.zeros((n_jobs, n_machines))

    C[0, 0] = seq[0, 0]
    for k in range(1, n_machines):
        C[0, k] = C[0, k - 1] + seq[0, k]
    for i in range(1, n_jobs):
        C[i, 0] = C[i - 1, 0] + seq[i, 0]
        C[i, 1:] = np.maximum(C[i - 1, 1:], C[i, :-1]) + seq[i, 1:]
    return float(C[-1, -1])


# ====================================================================
# 2. LOV decoding (continuous → permutation)
# ====================================================================

def decode_lov(position: np.ndarray) -> np.ndarray:
    """Convert a continuous vector into a permutation via argsort."""
    return np.argsort(position)


# ====================================================================
# 3. NEH constructive heuristic  (Nawaz, Enscore, Ham — 1983)
# ====================================================================

def neh_heuristic(processing_times: np.ndarray) -> tuple[np.ndarray, float]:
    """
    NEH heuristic: one of the best constructive methods for PFSP.

    Returns
    -------
    (permutation, makespan)
    """
    n_jobs = processing_times.shape[0]
    total_time = processing_times.sum(axis=1)
    order = np.argsort(-total_time)              # descending total work

    sequence = [order[0]]
    for job in order[1:]:
        best_cmax = np.inf
        best_seq = None
        for pos in range(len(sequence) + 1):
            trial = sequence[:pos] + [job] + sequence[pos:]
            cmax = compute_makespan(np.array(trial), processing_times)
            if cmax < best_cmax:
                best_cmax = cmax
                best_seq = trial
        sequence = best_seq

    seq = np.array(sequence)
    return seq, compute_makespan(seq, processing_times)


# ====================================================================
# 4. Local search (insertion neighbourhood)
# ====================================================================

def local_search_insertion(permutation: np.ndarray,
                           processing_times: np.ndarray,
                           max_iters: int = 20) -> tuple[np.ndarray, float]:
    """
    First-improvement local search on the insertion neighbourhood.

    Tries to move each job to every other position; stops when no
    improvement is found or ``max_iters`` is reached.
    """
    n = len(permutation)
    best_seq = permutation.copy()
    best_cmax = compute_makespan(best_seq, processing_times)

    for _ in range(max_iters):
        improved = False
        for i in range(n):
            job = best_seq[i]
            reduced = np.delete(best_seq, i)
            for j in range(n):
                if j == i:
                    continue
                trial = np.insert(reduced, j, job)
                cmax = compute_makespan(trial, processing_times)
                if cmax < best_cmax:
                    best_cmax = cmax
                    best_seq = trial
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best_seq, best_cmax


# ====================================================================
# 5. SMA-based PFSP solver
# ====================================================================

@dataclass
class PFSPResult:
    """Result of solving a PFSP instance."""

    permutation: np.ndarray
    """Best job sequence found."""

    makespan: float
    """Makespan (Cmax) of the best sequence."""

    sma_result: SMAResult
    """Raw SMA result with convergence and statistics."""

    ls_improvement: float = 0.0
    """Percentage improvement from the final local search."""

    def summary(self) -> str:
        return (f"Cmax = {self.makespan:.1f}    "
                f"evals = {self.sma_result.n_evaluations}    "
                f"time = {self.sma_result.runtime_sec:.3f}s    "
                f"LS+ = {self.ls_improvement:.2f}%")


def solve_pfsp(
    processing_times: np.ndarray,
    pop_size: int = 30,
    max_iter: int = 200,
    seed: int | None = None,
    use_local_search: bool = True,
) -> PFSPResult:
    """
    Solve a PFSP instance using the Slime Mould Algorithm.

    Parameters
    ----------
    processing_times : np.ndarray, shape (n_jobs, n_machines)
        The PFSP instance.
    pop_size : int
        SMA population size.
    max_iter : int
        SMA iterations.
    seed : int or None
        Random seed.
    use_local_search : bool
        Apply insertion local search to polish the final solution.

    Returns
    -------
    PFSPResult
    """
    n_jobs = processing_times.shape[0]

    # ── Build fitness function for SMA ────────────────────────────
    # SMA minimises continuous positions; we decode via LOV and
    # compute the makespan of the resulting permutation.
    def fitness_fn(position: np.ndarray) -> float:
        permutation = decode_lov(position)
        return compute_makespan(permutation, processing_times)

    # ── Seed initialisation with NEH ──────────────────────────────
    neh_seq, _ = neh_heuristic(processing_times)
    # Convert NEH permutation → continuous vector via inverse argsort
    neh_cont = np.argsort(neh_seq).astype(float)
    neh_cont = (neh_cont - neh_cont.min()) / (
        neh_cont.max() - neh_cont.min() + 1e-12
    )

    # Duplicate NEH seed with slight noise to inject diversity
    n_seed = max(1, pop_size // 4)
    rng_local = np.random.default_rng(seed)
    seeded = np.tile(neh_cont, (n_seed, 1))
    noise = rng_local.uniform(-0.05, 0.05, size=seeded.shape)
    init_pop = np.clip(seeded + noise, 0.0, 1.0)

    # ── Run SMA ───────────────────────────────────────────────────
    sma = SlimeMouldAlgorithm(
        fitness_fn=fitness_fn,
        dim=n_jobs,
        pop_size=pop_size,
        max_iter=max_iter,
        lb=0.0,
        ub=1.0,
        seed=seed,
    )

    sma_result = sma.run(init_positions=init_pop)

    best_perm = decode_lov(sma_result.best_position)
    best_fit = sma_result.best_fitness

    # ── Final local search ────────────────────────────────────────
    ls_improv = 0.0
    if use_local_search:
        cmax_before = best_fit
        best_perm, best_fit = local_search_insertion(
            best_perm, processing_times)
        ls_improv = ((cmax_before - best_fit) / cmax_before * 100
                     if cmax_before > 0 else 0.0)

    return PFSPResult(
        permutation=best_perm,
        makespan=best_fit,
        sma_result=sma_result,
        ls_improvement=ls_improv,
    )


# ====================================================================
# 6. Taillard-style instance generator
# ====================================================================

def generate_instance(n_jobs: int, n_machines: int, seed: int = 1) -> np.ndarray:
    """
    Generate a synthetic PFSP instance: processing times ~ U[1, 99].
    """
    rng = np.random.default_rng(seed)
    return rng.integers(1, 100, size=(n_jobs, n_machines)).astype(float)


# ====================================================================
# 7. Demo
# ====================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  SMA for Permutation Flow Shop Scheduling Problem (PFSP)")
    print("  ZARQI Ezzoubair")
    print("=" * 65)

    # ── Test on multiple instance sizes ───────────────────────────
    INSTANCES = [
        ("20×5",  20, 5),
        ("20×10", 20, 10),
        ("30×10", 30, 10),
        ("50×10", 50, 10),
    ]
    INSTANCE_SEED = 1

    for label, n_jobs, n_machines in INSTANCES:
        print(f"\n{'─' * 65}")
        print(f"  Instance: {n_jobs} jobs × {n_machines} machines")
        print(f"{'─' * 65}")

        pt = generate_instance(n_jobs, n_machines, seed=INSTANCE_SEED)

        # NEH baseline
        t0 = time.time()
        neh_seq, neh_cmax = neh_heuristic(pt)
        neh_time = time.time() - t0
        print(f"\n  NEH heuristic   →  Cmax = {neh_cmax:>8.1f}  "
              f"({neh_time:.3f}s)")

        # SMA — 5 runs
        runs = []
        for s in range(5):
            r = solve_pfsp(pt, pop_size=30, max_iter=200, seed=s)
            runs.append(r.makespan)

        runs = np.array(runs)
        gap_best = (runs.min() - neh_cmax) / neh_cmax * 100
        gap_mean = (runs.mean() - neh_cmax) / neh_cmax * 100

        print(f"  SMA (5× 30/200) →  best={runs.min():>8.1f}  "
              f"mean={runs.mean():>8.1f}  std={runs.std():.1f}")
        print(f"    Gap vs NEH:     best {gap_best:+.2f}%   "
              f"mean {gap_mean:+.2f}%")

    # ── Convergence plot (first instance) ────────────────────────
    try:
        import matplotlib.pyplot as plt

        label0, nj0, nm0 = INSTANCES[0]
        pt0 = generate_instance(nj0, nm0, seed=INSTANCE_SEED)
        neh_seq0, neh_cmax0 = neh_heuristic(pt0)

        # Runs for box plot
        all_runs = [solve_pfsp(pt0, pop_size=30, max_iter=200, seed=s).makespan
                    for s in range(5)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: convergence
        single = solve_pfsp(pt0, pop_size=30, max_iter=200, seed=42)
        ax1.plot(single.sma_result.convergence, color="#2196F3", linewidth=1.8)
        ax1.axhline(neh_cmax0, color="#4CAF50", linestyle="--", linewidth=1.2,
                    label=f"NEH ({neh_cmax0:.0f})")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Makespan (Cmax)")
        ax1.set_title(f"SMA Convergence — {label0}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right: boxplot
        ax2.boxplot([neh_cmax0], positions=[1], widths=0.5,
                    patch_artist=True,
                    boxprops=dict(facecolor="#4CAF50", alpha=0.5))
        ax2.boxplot(all_runs, positions=[2], widths=0.5,
                    patch_artist=True,
                    boxprops=dict(facecolor="#2196F3", alpha=0.5))
        ax2.set_xticklabels(["NEH", "SMA"])
        ax2.set_ylabel("Makespan (Cmax)")
        ax2.set_title(f"Distribution over 5 runs — {label0}")
        ax2.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig("pfsp_sma_results.png", dpi=150, bbox_inches="tight")
        print(f"\n  ✓ Plot saved → pfsp_sma_results.png")
    except ImportError:
        print("\n  (matplotlib not available — skipping plot)")

    print(f"\n{'=' * 65}")
    print("  Done.")
    print(f"{'=' * 65}")
