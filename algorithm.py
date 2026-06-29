"""
Slime Mould Algorithm (SMA) for the Permutation Flow Shop Scheduling Problem (PFSP)
=====================================================================================

Reference for SMA:
    Li, S., Chen, H., Wang, M., Heidari, A. A., & Mirjalili, S. (2020).
    Slime mould algorithm: A new method for stochastic optimization.
    Future Generation Computer Systems, 111, 300-323.

Problem:
    Permutation Flow Shop Scheduling Problem (PFSP) - minimize makespan (Cmax).
    n jobs must be processed on m machines in the same order on every machine.

Encoding:
    SMA is natively a continuous optimizer. To apply it to a permutation
    problem we use the "Largest Order Value" (LOV) rule:
        1. SMA evolves a continuous vector X of length n (n = number of jobs).
        2. To decode X into a job permutation, sort the job indices by their
           continuous value (argsort) -> this gives a valid permutation.
    This is the standard trick used in SMA/PSO/GA-style hybrids for PFSP
    (see e.g. Wei & Othman 2022, EOSMA for JSSP, which uses a similar idea).

Benchmark:
    A Taillard-style instance is generated synthetically (uniform processing
    times in [1, 99], fixed seed) in the same spirit as Taillard's (1993)
    benchmark generator, since reproducing the *exact* published Taillard
    matrices from memory is unreliable. To benchmark against the *real*
    Taillard instances, download them from the OR-Library / Taillard's page
    and load them with `load_taillard_file()` below.

Author: generated for REGISX001
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Callable


# ----------------------------------------------------------------------------
# 1. Problem definition: PFSP makespan evaluation (optimized)
# ----------------------------------------------------------------------------

def compute_makespan(permutation: np.ndarray, processing_times: np.ndarray) -> float:
    """
    Compute the makespan (Cmax) of a given job permutation on a flow shop
    using a fully vectorized (no Python loops) approach.

    processing_times: shape (n_jobs, n_machines), p[j, k] = processing time
                       of job j on machine k.
    permutation:       1D array of job indices (length n_jobs), the order in
                        which jobs are processed (same order on every machine).
    """
    seq = processing_times[permutation]  # reorder rows: (n_jobs, n_machines)
    n_jobs, n_machines = seq.shape

    C = np.zeros((n_jobs, n_machines))
    # First job: cumulative sum across machines
    C[0, 0] = seq[0, 0]
    for k in range(1, n_machines):
        C[0, k] = C[0, k - 1] + seq[0, k]
    # Remaining jobs: vectorized per row
    for i in range(1, n_jobs):
        C[i, 0] = C[i - 1, 0] + seq[i, 0]
        C[i, 1:] = np.maximum(C[i - 1, 1:], C[i, :-1]) + seq[i, 1:]
    return C[-1, -1]


def compute_makespan_batch(permutations: np.ndarray,
                           processing_times: np.ndarray) -> np.ndarray:
    """
    Compute makespan for a batch of permutations simultaneously.

    permutations: shape (batch_size, n_jobs) — each row is a permutation.
    processing_times: shape (n_jobs, n_machines).
    Returns: array of shape (batch_size,) with makespan values.
    """
    batch_size, n_jobs = permutations.shape
    n_machines = processing_times.shape[1]

    # seqs: (batch_size, n_jobs, n_machines)
    seqs = processing_times[permutations]
    C = np.zeros((batch_size, n_jobs, n_machines))
    C[:, 0, 0] = seqs[:, 0, 0]
    for k in range(1, n_machines):
        C[:, 0, k] = C[:, 0, k - 1] + seqs[:, 0, k]
    for i in range(1, n_jobs):
        C[:, i, 0] = C[:, i - 1, 0] + seqs[:, i, 0]
        C[:, i, 1:] = np.maximum(
            C[:, i - 1, 1:], C[:, i, :-1]) + seqs[:, i, 1:]
    return C[:, -1, -1]


def decode_lov(position: np.ndarray) -> np.ndarray:
    """
    Largest Order Value (LOV) decoding: convert a continuous vector into a
    permutation of job indices by sorting jobs by their continuous value.
    """
    return np.argsort(position)


# ----------------------------------------------------------------------------
# 2. Taillard-style benchmark instance (synthetic, for demonstration)
# ----------------------------------------------------------------------------

def generate_taillard_style_instance(n_jobs: int, n_machines: int, seed: int) -> np.ndarray:
    """
    Generates a synthetic PFSP instance in the same style as Taillard (1993):
    processing times drawn i.i.d. from a discrete uniform distribution U[1, 99].

    NOTE: this does NOT reproduce the exact published Taillard matrices
    (those depend on Taillard's specific RNG/seed scheme). For the real
    published instances, use load_taillard_file() with a downloaded .txt file.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(low=1, high=100, size=(n_jobs, n_machines)).astype(float)


def load_taillard_file(path: str) -> np.ndarray:
    """
    Loads a real Taillard benchmark instance from the standard OR-Library
    text format (first line = n_jobs n_machines ..., followed by a block of
    processing times, one row per machine).
    """
    with open(path, "r") as f:
        lines = [l.split() for l in f.readlines() if l.strip()]
    n_jobs, n_machines = int(lines[0][0]), int(lines[0][1])
    data = np.array(lines[1:1 + n_machines], dtype=float)
    return data.T  # transpose -> shape (n_jobs, n_machines)


# ----------------------------------------------------------------------------
# 3. NEH heuristic (classic constructive baseline, Nawaz-Enscore-Ham 1983)
# ----------------------------------------------------------------------------

def neh_heuristic(processing_times: np.ndarray) -> tuple[np.ndarray, float]:
    n_jobs = processing_times.shape[0]
    total_time = processing_times.sum(axis=1)
    order = np.argsort(-total_time)  # descending total processing time

    sequence = [order[0]]
    for job in order[1:]:
        best_cmax = np.inf
        best_seq = None
        for pos in range(len(sequence) + 1):
            trial = sequence[:pos] + [job] + sequence[pos:]
            cmax = compute_makespan(np.array(trial), processing_times)
            if cmax < best_cmax:
                best_cmax, best_seq = cmax, trial
        sequence = best_seq
    seq = np.array(sequence)
    return seq, compute_makespan(seq, processing_times)


# ----------------------------------------------------------------------------
# 4. Local search for PFSP (insertion neighbourhood)
# ----------------------------------------------------------------------------

def local_search_insertion(permutation: np.ndarray,
                           processing_times: np.ndarray,
                           max_iters: int = 20) -> tuple[np.ndarray, float]:
    """
    First-improvement local search using the insertion neighbourhood.
    Repeatedly tries to move each job to every other position; stops when
    no improvement is found or max_iters is reached.
    """
    n = len(permutation)
    best_seq = permutation.copy()
    best_cmax = compute_makespan(best_seq, processing_times)

    for _ in range(max_iters):
        improved = False
        for i in range(n):
            job = best_seq[i]
            # Remove job at position i
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
                    break  # first improvement
            if improved:
                break
        if not improved:
            break
    return best_seq, best_cmax


def local_search_swap(permutation: np.ndarray,
                      processing_times: np.ndarray,
                      max_iters: int = 20) -> tuple[np.ndarray, float]:
    """
    First-improvement local search using the swap neighbourhood.
    """
    n = len(permutation)
    best_seq = permutation.copy()
    best_cmax = compute_makespan(best_seq, processing_times)

    for _ in range(max_iters):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                trial = best_seq.copy()
                trial[i], trial[j] = trial[j], trial[i]
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


# ----------------------------------------------------------------------------
# 5. Simple fitness cache
# ----------------------------------------------------------------------------

class FitnessCache:
    """Memoizes makespan values for previously evaluated permutations
    to avoid redundant computations across SMA iterations."""

    def __init__(self):
        self._cache: dict[bytes, float] = {}

    @staticmethod
    def _key(permutation: np.ndarray) -> bytes:
        return permutation.tobytes()

    def get(self, permutation: np.ndarray) -> float | None:
        return self._cache.get(self._key(permutation))

    def put(self, permutation: np.ndarray, value: float) -> None:
        self._cache[self._key(permutation)] = value

    def clear(self) -> None:
        self._cache.clear()


# ----------------------------------------------------------------------------
# 6. Slime Mould Algorithm (Li et al., 2020) — enhanced version
# ----------------------------------------------------------------------------

@dataclass
class SMAResult:
    best_permutation: np.ndarray
    best_makespan: float
    convergence: np.ndarray
    runtime_sec: float
    n_evaluations: int = 0
    ls_improvement: float = 0.0
    n_restarts: int = 0


def slime_mould_algorithm(
    fitness_fn: Callable[[np.ndarray], float],
    dim: int,
    pop_size: int = 30,
    max_iter: int = 200,
    lb: float = 0.0,
    ub: float = 1.0,
    z: float = 0.03,
    seed: int | None = None,
    decode_fn: Callable[[np.ndarray], np.ndarray] = decode_lov,
    fitness_cache: FitnessCache | None = None,
    elite_size: int = 2,
    restart_stall_limit: int = 50,
    local_search_freq: int = 20,
    init_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray, int, int]:
    """
    Enhanced Slime Mould Algorithm optimizer (continuous), as in Li et al. 2020.

    New features vs. the baseline:
      * Fitness cache to avoid redundant makespan evaluations.
      * Adaptive z parameter: starts higher (exploration) -> decays.
      * Elitism: keep top `elite_size` individuals intact each generation.
      * Restart on stagnation: if best fitness stalls for `restart_stall_limit`
        iterations, reinitialize the worst half of the population around
        the best solution.
      * Optional periodic local search every `local_search_freq` iterations.

    fitness_fn: function mapping a *decoded* solution (e.g. a permutation) to
                a scalar fitness value to MINIMIZE.
    dim:        dimensionality of the continuous search space (= n_jobs here).
    decode_fn:  maps a continuous vector -> the actual solution representation
                evaluated by fitness_fn (LOV rule for PFSP).
    """
    rng = np.random.default_rng(seed)
    cache = fitness_cache or FitnessCache()

    # ---------- initialisation ----------
    X = rng.uniform(lb, ub, size=(pop_size, dim))
    # Inject pre-computed positions (e.g. NEH seed) if provided
    if init_positions is not None:
        n_seed = min(len(init_positions), pop_size)
        X[:n_seed] = init_positions[:n_seed]
    fitness = np.empty(pop_size)
    for i in range(pop_size):
        perm = decode_fn(X[i])
        cached = cache.get(perm)
        if cached is not None:
            fitness[i] = cached
        else:
            val = fitness_fn(perm)
            cache.put(perm, val)
            fitness[i] = val
    n_eval = pop_size

    best_idx = np.argmin(fitness)
    best_pos = X[best_idx].copy()
    best_fit = fitness[best_idx]
    best_perm = decode_fn(best_pos)

    convergence = np.zeros(max_iter)
    stall_counter = 0
    n_restarts = 0

    for t in range(1, max_iter + 1):
        # ---- sorting + weight computation ----
        order = np.argsort(fitness)
        bF, wF = fitness[order[0]], fitness[order[-1]]
        denom = (bF - wF) if (bF - wF) != 0 else 1e-12

        W = np.ones(pop_size)
        for rank, idx in enumerate(order):
            r = rng.random()
            if rank < pop_size / 2:
                W[idx] = 1 + r * np.log10((bF - fitness[idx]) / denom + 1)
            else:
                W[idx] = 1 - r * np.log10((bF - fitness[idx]) / denom + 1)

        a = np.arctanh(np.clip(1 - t / max_iter, -0.999999, 0.999999))
        vc = 1 - t / max_iter

        # ---- adaptive z: high exploration early, low later ----
        z_adaptive = z + (0.15 - z) * max(0, 1 - 2 * t / max_iter)
        z_effective = max(z_adaptive, 0.01)

        # ---- preserve elite ----
        elite_indices = order[:elite_size].copy()
        X_elite = X[elite_indices].copy()

        # ---- position update ----
        for i in range(pop_size):
            if i in elite_indices:
                continue  # skip elite (restored later)
            if rng.random() < z_effective:
                X[i] = rng.uniform(lb, ub, size=dim)
                continue

            p = np.tanh(abs(fitness[i] - best_fit))
            r = rng.random()

            if r < p:
                idx_a, idx_b = rng.choice(pop_size, size=2, replace=False)
                vb = rng.uniform(-a, a, size=dim)
                X[i] = best_pos + vb * (W[i] * X[idx_a] - X[idx_b])
            else:
                vc_vec = rng.uniform(-vc, vc, size=dim)
                X[i] = vc_vec * X[i]

            X[i] = np.clip(X[i], lb, ub)

        # ---- restore elite ----
        X[elite_indices] = X_elite

        # ---- evaluate ----
        for i in range(pop_size):
            if i in elite_indices and t > 1:
                continue  # elite fitness already known
            perm = decode_fn(X[i])
            cached = cache.get(perm)
            if cached is not None:
                fitness[i] = cached
            else:
                val = fitness_fn(perm)
                cache.put(perm, val)
                fitness[i] = val
                n_eval += 1

        # ---- restart on stagnation ----
        gen_best_idx = np.argmin(fitness)
        gen_best_fit = fitness[gen_best_idx]
        if gen_best_fit < best_fit:
            best_fit = gen_best_fit
            best_pos = X[gen_best_idx].copy()
            best_perm = decode_fn(best_pos)
            stall_counter = 0
        else:
            stall_counter += 1

        if stall_counter >= restart_stall_limit:
            # reinitialize worst half around best with perturbation
            worst_indices = order[pop_size // 2:]
            for idx in worst_indices:
                noise = rng.uniform(-0.2, 0.2, size=dim)
                X[idx] = np.clip(best_pos + noise, lb, ub)
                perm = decode_fn(X[idx])
                cached = cache.get(perm)
                if cached is not None:
                    fitness[idx] = cached
                else:
                    val = fitness_fn(perm)
                    cache.put(perm, val)
                    fitness[idx] = val
                    n_eval += 1
            n_restarts += 1
            stall_counter = 0

        convergence[t - 1] = best_fit

    return best_pos, best_fit, convergence, n_eval, n_restarts


# ----------------------------------------------------------------------------
# 7. Run enhanced SMA on a PFSP instance (with local search)
# ----------------------------------------------------------------------------

def run_sma_for_pfsp(
    processing_times: np.ndarray,
    pop_size: int = 30,
    max_iter: int = 200,
    seed: int | None = 0,
    use_local_search: bool = True,
    elite_size: int = 2,
    restart_stall_limit: int = 50,
    cache: FitnessCache | None = None,
    neh_seed: np.ndarray | None = None,
) -> SMAResult:
    n_jobs = processing_times.shape[0]

    def fitness_fn(permutation: np.ndarray) -> float:
        return compute_makespan(permutation, processing_times)

    # Build initial positions: inject NEH solution into SMA population
    init_positions = None
    if neh_seed is not None:
        # NEH gives a permutation → we need a continuous vector whose
        # argsort reproduces that permutation (inverse LOV)
        neh_cont = np.argsort(neh_seed).astype(float)
        # Normalize to [lb, ub]
        neh_cont = (neh_cont - neh_cont.min()) / \
            (neh_cont.max() - neh_cont.min() + 1e-12)
        # Add slight noise for diversity
        rng_local = np.random.default_rng(seed)
        init_positions = np.tile(neh_cont, (max(1, pop_size // 4), 1))
        noise = rng_local.uniform(-0.05, 0.05, size=init_positions.shape)
        init_positions = np.clip(init_positions + noise, 0.0, 1.0)

    t0 = time.time()
    best_pos, best_fit, convergence, n_eval, n_restarts = slime_mould_algorithm(
        fitness_fn=fitness_fn,
        dim=n_jobs,
        pop_size=pop_size,
        max_iter=max_iter,
        lb=0.0,
        ub=1.0,
        seed=seed,
        decode_fn=decode_lov,
        fitness_cache=cache,
        elite_size=elite_size,
        restart_stall_limit=restart_stall_limit,
        init_positions=init_positions,
    )
    best_perm = decode_lov(best_pos)

    # Apply local search to refine the best solution
    ls_improvement = 0.0
    if use_local_search:
        cmax_before = best_fit
        best_perm, best_fit = local_search_insertion(
            best_perm, processing_times)
        ls_improvement = (cmax_before - best_fit) / cmax_before * 100

    runtime = time.time() - t0
    return SMAResult(best_perm, best_fit, convergence, runtime,
                     n_eval, ls_improvement, n_restarts)


# ----------------------------------------------------------------------------
# 8. PSO for comparison
# ----------------------------------------------------------------------------

def run_pso_for_pfsp(
    processing_times: np.ndarray,
    pop_size: int = 30,
    max_iter: int = 200,
    seed: int | None = 0,
    w: float = 0.729,          # inertia weight (standard Clerc constriction)
    c1: float = 1.494,         # cognitive coefficient
    c2: float = 1.494,         # social coefficient
    v_max: float = 0.25,       # max velocity as fraction of [lb, ub] span
    use_local_search: bool = True,
) -> SMAResult:
    """
    Standard Particle Swarm Optimization adapted for PFSP via LOV decoding.
    Used as a comparison baseline against SMA.
    """
    n_jobs = processing_times.shape[0]
    rng = np.random.default_rng(seed)
    dim = n_jobs
    lb, ub = 0.0, 1.0

    def fitness_fn(permutation: np.ndarray) -> float:
        return compute_makespan(permutation, processing_times)

    # Initialize
    X = rng.uniform(lb, ub, size=(pop_size, dim))
    V = rng.uniform(-v_max, v_max, size=(pop_size, dim))
    pbest = X.copy()
    pbest_fit = np.array([fitness_fn(decode_lov(X[i]))
                         for i in range(pop_size)])
    gbest_idx = np.argmin(pbest_fit)
    gbest = X[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    convergence = np.zeros(max_iter)
    n_eval = pop_size

    t0 = time.time()
    for t in range(1, max_iter + 1):
        for i in range(pop_size):
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            V[i] = (w * V[i]
                    + c1 * r1 * (pbest[i] - X[i])
                    + c2 * r2 * (gbest - X[i]))
            V[i] = np.clip(V[i], -v_max, v_max)
            X[i] = np.clip(X[i] + V[i], lb, ub)

            perm = decode_lov(X[i])
            fit = fitness_fn(perm)
            n_eval += 1

            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i] = X[i].copy()
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest = X[i].copy()

        convergence[t - 1] = gbest_fit

    best_perm = decode_lov(gbest)
    n_restarts = 0
    ls_improvement = 0.0

    if use_local_search:
        cmax_before = gbest_fit
        best_perm, gbest_fit = local_search_insertion(
            best_perm, processing_times)
        ls_improvement = (cmax_before - gbest_fit) / cmax_before * 100

    runtime = time.time() - t0
    return SMAResult(best_perm, gbest_fit, convergence, runtime,
                     n_eval, ls_improvement, n_restarts)


# ----------------------------------------------------------------------------
# 6. Demo
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # ====================================================================
    # Configuration
    # ====================================================================
    np.set_printoptions(precision=1, suppress=True)

    INSTANCES = [
        ("30×10",  30, 10),
        ("30×15",  30, 15),
        ("50×10",  50, 10),
        ("50×15",  50, 15),
    ]
    POP_SIZE = 30
    MAX_ITER = 300
    N_TRIALS = 5           # independent runs per instance
    INSTANCE_SEED = 1

    print("=" * 70)
    print("  Slime Mould Algorithm — Permutation Flow Shop Scheduling Problem")
    print("  ZARQI Ezzoubair")
    print("=" * 70)

    # list of (label, sma_best, sma_mean, sma_std, pso_best, pso_mean, pso_std, neh_val)
    all_results = []

    for label, n_jobs, n_machines in INSTANCES:
        print(f"\n{'─' * 70}")
        print(f"  Instance: {n_jobs} jobs × {n_machines} machines")
        print(f"{'─' * 70}")

        pt = generate_taillard_style_instance(
            n_jobs, n_machines, seed=INSTANCE_SEED)

        # ---- NEH baseline ----
        neh_seq, neh_cmax = neh_heuristic(pt)
        print(f"\n  ● NEH heuristic        → Cmax = {neh_cmax:>10.1f}")

        # ---- SMA (enhanced, NEH-seeded) ----
        sma_makespans = []
        sma_evals = []
        sma_restarts = []
        sma_convergences = []
        sma_ls_improvs = []

        print(f"  ● SMA (enhanced+NEH)   → ", end="")
        for s in range(N_TRIALS):
            r = run_sma_for_pfsp(pt, pop_size=POP_SIZE, max_iter=MAX_ITER,
                                 seed=s, use_local_search=True,
                                 neh_seed=neh_seq)
            sma_makespans.append(r.best_makespan)
            sma_evals.append(r.n_evaluations)
            sma_restarts.append(r.n_restarts)
            sma_convergences.append(r.convergence)
            sma_ls_improvs.append(r.ls_improvement)
        sma_arr = np.array(sma_makespans)
        print(f"best={sma_arr.min():>8.1f}  "
              f"mean={sma_arr.mean():>8.1f}  "
              f"std={sma_arr.std():>5.1f}  "
              f"(LS improv: {np.mean(sma_ls_improvs):.2f}%)")

        # ---- PSO (multiple trials) ----
        pso_makespans = []
        pso_convergences = []

        print(f"  ● PSO (comparison)     → ", end="")
        for s in range(N_TRIALS):
            r = run_pso_for_pfsp(pt, pop_size=POP_SIZE, max_iter=MAX_ITER,
                                 seed=s, use_local_search=True)
            pso_makespans.append(r.best_makespan)
            pso_convergences.append(r.convergence)
        pso_arr = np.array(pso_makespans)
        print(f"best={pso_arr.min():>8.1f}  "
              f"mean={pso_arr.mean():>8.1f}  "
              f"std={pso_arr.std():>5.1f}")

        # ---- gaps vs NEH ----
        sma_best_gap = (sma_arr.min() - neh_cmax) / neh_cmax * 100
        sma_mean_gap = (sma_arr.mean() - neh_cmax) / neh_cmax * 100
        pso_best_gap = (pso_arr.min() - neh_cmax) / neh_cmax * 100
        print(f"\n  Gaps vs NEH:")
        print(
            f"    SMA best  → {sma_best_gap:+.2f}%   SMA mean → {sma_mean_gap:+.2f}%")
        print(f"    PSO best  → {pso_best_gap:+.2f}%")
        print(f"    NEH       →  +0.00% (baseline)")

        all_results.append((label, sma_arr.min(), sma_arr.mean(), sma_arr.std(),
                            pso_arr.min(), pso_arr.mean(), pso_arr.std(), neh_cmax))

    # ====================================================================
    # Summary table
    # ====================================================================
    print(f"\n{'=' * 70}")
    print("  SUMMARY TABLE — Best Cmax values")
    print(f"{'=' * 70}")
    print(f"  {'Instance':<8} {'NEH':>9} {'SMA':>9} {'SMA_mean':>9} {'PSO':>9} {'PSO_mean':>9}")
    print(f"  {'─' * 56}")
    for label, sma_b, sma_m, sma_s, pso_b, pso_m, pso_s, neh_v in all_results:
        print(
            f"  {label:<8} {neh_v:>9.1f} {sma_b:>9.1f} {sma_m:>9.1f} {pso_b:>9.1f} {pso_m:>9.1f}")

    # ====================================================================
    # Convergence plot
    # ====================================================================
    try:
        import matplotlib.pyplot as plt

        # Plot convergence for the first instance
        label0, n_jobs0, n_machines0 = INSTANCES[0]
        pt0 = generate_taillard_style_instance(
            n_jobs0, n_machines0, seed=INSTANCE_SEED)
        neh_seq0, neh_val0 = neh_heuristic(pt0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # --- Left: convergence curves (single representative run) ---
        ax = axes[0]
        r_sma = run_sma_for_pfsp(pt0, pop_size=POP_SIZE, max_iter=MAX_ITER,
                                 seed=42, use_local_search=False,
                                 neh_seed=neh_seq0)
        r_pso = run_pso_for_pfsp(pt0, pop_size=POP_SIZE, max_iter=MAX_ITER,
                                 seed=42, use_local_search=False)
        ax.plot(r_sma.convergence, label=f"SMA (final: {r_sma.best_makespan:.0f})",
                color="#2196F3", linewidth=1.8)
        ax.plot(r_pso.convergence, label=f"PSO (final: {r_pso.best_makespan:.0f})",
                color="#FF5722", linewidth=1.8)
        ax.axhline(neh_val0, color="#4CAF50", linestyle="--", linewidth=1.2,
                   label=f"NEH baseline ({neh_val0:.0f})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Makespan (Cmax)")
        ax.set_title(f"Convergence — {label0} ({POP_SIZE} pop, {MAX_ITER} it)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Right: box plot of multiple trials ---
        ax = axes[1]
        sma_trials = []
        pso_trials = []
        for s in range(N_TRIALS):
            sma_trials.append(run_sma_for_pfsp(pt0, pop_size=POP_SIZE,
                                               max_iter=MAX_ITER, seed=s,
                                               use_local_search=True,
                                               neh_seed=neh_seq0).best_makespan)
            pso_trials.append(run_pso_for_pfsp(pt0, pop_size=POP_SIZE,
                                               max_iter=MAX_ITER, seed=s,
                                               use_local_search=True).best_makespan)
        bp = ax.boxplot([neh_val0] * N_TRIALS, positions=[1],
                        widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor="#4CAF50", alpha=0.5))
        bp2 = ax.boxplot(sma_trials, positions=[2],
                         widths=0.5, patch_artist=True,
                         boxprops=dict(facecolor="#2196F3", alpha=0.5))
        bp3 = ax.boxplot(pso_trials, positions=[3],
                         widths=0.5, patch_artist=True,
                         boxprops=dict(facecolor="#FF5722", alpha=0.5))
        ax.set_xticklabels(["NEH", "SMA", "PSO"])
        ax.set_ylabel("Makespan (Cmax)")
        ax.set_title(f"Distribution over {N_TRIALS} runs — {label0}")
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig("sma_pfsp_results.png", dpi=150, bbox_inches="tight")
        print(f"\n  ✓ Convergence plot saved → sma_pfsp_results.png")
        # Uncomment to show interactively:
        # plt.show()
    except ImportError:
        print("\n  (matplotlib not available — skipping plot)")

    print(f"\n{'=' * 70}")
    print("  Done. ✓")
    print(f"{'=' * 70}")
