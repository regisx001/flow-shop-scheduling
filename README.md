# Metaheuristics for the Permutation Flow Shop Scheduling Problem

A Python framework for solving the **Permutation Flow Shop Scheduling Problem (PFSP)** using continuous metaheuristics with LOV (Largest Order Value) encoding.

Four algorithms are implemented and benchmarked:

| Algorithm | Type | Reference |
|-----------|------|-----------|
| **NEH** | Constructive heuristic | Nawaz, Enscore & Ham, 1983 |
| **SMA** | Slime Mould Algorithm | Li et al., 2020 |
| **GWO** | Grey Wolf Optimizer | Mirjalili et al., 2014 |
| **GA** | Genetic Algorithm | BLX-&alpha; crossover, tournament selection |

## Author

**ZARQI Ezzoubair** — June 2026

## Problem Description

In the PFSP, **n jobs** must be processed on **m machines** in the same order on every machine. The objective is to **minimize the makespan (Cmax)** — the completion time of the last job on the last machine.

The search space has **n!** possible permutations, making exact methods infeasible for large instances. This project explores metaheuristic approaches.

### LOV Encoding

All metaheuristics operate in a **continuous space** [0,1]^n. Each continuous vector is decoded into a permutation via `argsort` (Largest Order Value rule). This allows standard continuous optimizers to solve a discrete permutation problem.

## Project Structure

```
.
├── pfsp.py              # PFSP problem: makespan, NEH, local search, instance generation
├── sma.py               # Slime Mould Algorithm (continuous optimizer)
├── gwo.py               # Grey Wolf Optimizer (continuous optimizer)
├── ga.py                # Genetic Algorithm (BLX-alpha crossover)
├── benchmark.py          # CLI benchmark: NEH vs SMA vs GWO vs GA
├── benchmark.ipynb       # Jupyter notebook: interactive benchmark with inline plots
├── algorithm.py          # Original SMA+PFSP prototype
├── subject.txt           # Project metadata
├── article.pdf           # Reference article
└── venv/                 # Python virtual environment
```

## Quick Start

### 1. Clone and set up the environment

```bash
cd "Metaheuristique Project"
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib
```

### 2. Run the command-line benchmark

```bash
python benchmark.py
```

This runs NEH vs SMA vs GWO vs GA on 5 instances (20x5 to 100x10) and saves:
- `benchmark_pfsp.png` — box plot of makespan distributions
- `benchmark_convergence.png` — convergence curves (mean +/- std)

### 3. Open the interactive notebook

```bash
code benchmark.ipynb
```

Run all cells to get interactive tables and inline plots. Tweak parameters in **Section 3** (POP_SIZE, MAX_ITER, N_TRIALS) and re-run.

## Algorithms

### NEH (Nawaz-Enscore-Ham)

Constructive heuristic — the best-known polynomial-time method for PFSP:

1. Sort jobs by descending total processing time
2. Insert each job at the position that minimizes partial makespan

**Complexity:** O(n^3 m). **Deterministic** — always produces the same solution.

### SMA (Slime Mould Algorithm)

Population-based metaheuristic inspired by the foraging behavior of slime mould:

- Agents move toward food (best solution) using **adaptive weights** W
- Two update modes: **approach food** (exploitation) and **wrap food** (contraction)
- **Oscillation probability** z injects random positions for diversity
- **Elitism** + **stagnation restart** prevent premature convergence

### GWO (Grey Wolf Optimizer)

Social hierarchy-based metaheuristic:

- **Alpha, Beta, Delta** (top 3 solutions) guide the search
- All wolves update toward the average of the three leaders
- Parameter **a** decreases linearly from 2 to 0, balancing exploration/exploitation

### GA (Genetic Algorithm)

Real-coded GA with standard operators:

- **Selection:** Binary tournament
- **Crossover:** BLX-&alpha; (blend crossover, &alpha;=0.3) at 90% rate
- **Mutation:** Gaussian noise (&sigma;=10% of range) at 10% rate
- **Elitism:** Top 2 individuals preserved

## Benchmark Results (pop=10, iter=20, 5 trials)

| Instance | NEH | SMA Mean | GWO Mean | GA Mean |
|----------|-----|----------|----------|---------|
| 20x5 | 1156 | 1156 +/- 0 | 1156 +/- 0 | 1156 +/- 0 |
| 30x10 | 1625 | 1625 +/- 0 | 1625 +/- 0 | 1625 +/- 0 |
| 50x10 | 2480 | 2534 +/- 10 | 2511 +/- 22 | 2544 +/- 13 |
| 50x20 | 3335 | 3365 +/- 30 | 3349 +/- 20 | 3359 +/- 33 |
| 100x10 | 5180 | 5180 +/- 0 | 5180 +/- 0 | 5186 +/- 9 |

**Key findings:**

- NEH is remarkably strong — often near-optimal for Taillard-style instances
- SMA converges fastest (2-5 iterations) with near-zero variance
- GWO has more variance but sometimes outperforms SMA
- GA sits between SMA and GWO in reliability

## API Reference

### PFSP Module (`pfsp.py`)

```python
>>> from pfsp import generate_instance, neh_heuristic, compute_makespan, decode_lov

>>> pt = generate_instance(30, 10, seed=1)    # (n_jobs, n_machines)
>>> seq, cmax = neh_heuristic(pt)              # NEH solution
>>> perm = decode_lov(np.random.rand(30))       # LOV: continuous -> permutation
>>> cmax = compute_makespan(perm, pt)           # evaluate a permutation
```

### Optimizer Interface

All optimizers share the same interface:

```python
>>> from sma import SlimeMouldAlgorithm
>>> from gwo import GreyWolfOptimizer
>>> from ga import GeneticAlgorithm

>>> algo = SlimeMouldAlgorithm(
...     fitness_fn=my_function,
...     dim=30,
...     pop_size=30,
...     max_iter=200,
...     lb=0.0, ub=1.0,
...     seed=42
... )
>>> result = algo.run()
>>> print(result.best_fitness, result.best_position, result.convergence)
```

### Benchmark Module (`benchmark.py`)

```python
>>> from benchmark import run_benchmark
>>> run_benchmark(pop_size=10, max_iter=20, n_trials=5, save_plot=True)
```

## References

1. Nawaz, M., Enscore, E. E., & Ham, I. (1983). A heuristic algorithm for the m-machine, n-job flow-shop sequencing problem. *Omega*, 11(1), 91-95.

2. Li, S., Chen, H., Wang, M., Heidari, A. A., & Mirjalili, S. (2020). Slime mould algorithm: A new method for stochastic optimization. *Future Generation Computer Systems*, 111, 300-323.

3. Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey Wolf Optimizer. *Advances in Engineering Software*, 69, 46-61.

4. Eshelman, L. J., & Schaffer, J. D. (1993). Real-coded genetic algorithms and interval-schemata. *Foundations of Genetic Algorithms*, 2, 187-202.

## License

This project is created for academic/research purposes.
