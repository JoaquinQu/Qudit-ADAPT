# Multiway number partitioning (Sec. III B)

Data and analysis notebook for the multiway number partitioning results of the paper:
Qudit-ADAPT with the $l=1$ and $l=2$ operator pools against qudit QAOA, on balanced instances with
$n=5,\dots,12$ qutrits.

```
mwnp/
├── mwnp_results.ipynb               regenerates Figs. 6–7, the numbers of Sec. III B and the table of Appendix C
├── figures/                         the figures written by the notebook
└── data/
    ├── instances.json               the 160 instances (20 per size)
    ├── adapt_runs.json              1600 Qudit-ADAPT runs
    ├── qaoa_interp.json             qudit QAOA with INTERP initialization, p = 1..40
    └── qaoa_random_restarts_p40.json  qudit QAOA at p = 40 from 10 random initializations
```

Run it with

```bash
jupyter lab mwnp/mwnp_results.ipynb
```

The notebook only reads `data/`; it runs in a few seconds and needs `numpy` and `matplotlib`. If LaTeX with
the `txfonts` package is available, the figures are rendered exactly as in the manuscript; otherwise the STIX
fonts are used.

## The problem

Each instance is a set of $n$ distinct integers sampled from $[1,3n]$ that admits a **unique** perfect partition
into three subsets of equal sum, up to permutations of the subsets. The ground state of the cost Hamiltonian of
Eq. (33) is therefore sixfold degenerate with energy $E_0=0$, and every energy in the data directly measures
the distance from the optimal solution. Uniqueness was verified by enumerating all $3^n$ assignments.

## Data format

All energies are expectation values of $H_C$ as written in Eq. (33).

**`instances.json`** — one entry per instance: `n`, `instance` (0–19), `numbers` (ascending), `subset_sum`,
`optimal_subsets`, `ground_state_degeneracy`, and the `generation_seed` and `sampling_range` used to draw it.

**`adapt_runs.json`** — one entry per run, for every size, pool, instance and ordering
($8\times2\times20\times5=1600$ runs):

| Field | Meaning |
|---|---|
| `n`, `l`, `instance` | size, pool order, instance |
| `ordering`, `numbers_per_qutrit` | 0 is the ascending order of the numbers, 1–4 are random permutations; `numbers_per_qutrit[j]` is the number encoded by qutrit `j+1` |
| `energy` | `energy[k]` is the energy after `k` operators; `energy[0]` is the initial state $\lvert +_3\rangle^{\otimes n}$ |
| `operators` | the selected operators, each a list of factors `[site, axis]` (sites numbered from 1); the generator is the Hermitian part of their product |
| `parameters` | the optimized angles of the final ansatz |
| `stop_reason` | `gradient_norm_below_epsilon` or `max_parameters` (80) |

Settings: gradient threshold $\varepsilon=10^{-3}$, at most 80 parameters, warm start
$\boldsymbol\theta_k=(\boldsymbol\theta^*_{k-1},0)$, BFGS with analytic gradients. Runs with $n\le9$ were computed
with a cap of 150 parameters and are stored truncated at 80; because ADAPT is deterministic, their first 80
iterations are identical to a run with cap 80.

**`qaoa_interp.json`** — one entry per instance: `energy_per_depth[p-1]` is the energy at depth $p$ ($2p$
parameters), and `angles_p40` the optimized angles at $p=40$. Mixer $\sum_jL_x^{(j)}$ and initial state
$\lvert +_3\rangle^{\otimes n}$, following Deller *et al.* (Phys. Rev. A **107**, 062410). The optimizer
works with $H_C/\sigma$, with $\sigma$ the standard deviation of $H_C$ over the computational basis (stored as
`H_C_std`); this only rescales the angles $\gamma$.

**`qaoa_random_restarts_p40.json`** — `final_energy` of 10 optimizations per instance at $p=40$ from angles
drawn uniformly in $[-\pi,\pi]$. Used in Sec. III B to motivate the INTERP initialization.

## Provenance

The runs were produced on the BitWit workstation (AMD Threadripper 7980X) by H. Díaz-Moraga with
`cluster/benchmark_mwnp.py` (Qudit-ADAPT, engine in `funciones/utilidades_mwnp.py`) and
`cluster/qaoa_benchmark_mwnp.py` (QAOA, engine in `funciones/qaoa_mwnp.py`). The full raw outputs, including
per-iteration diagnostics that the paper does not use, are in `resultados/mwnp_ordenes_5a9/`,
`resultados/mwnp_k80/` and `resultados/qaoa_p40/`; the files in `data/` are a cleaned export of the fields
the paper uses.
