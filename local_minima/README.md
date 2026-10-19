# Local minima and the warm start (Sec. III C)

Data and analysis notebook for the local-minimum analysis of the paper, on the irregular graph $G_1$ and the
complete graph $K_6$ (Max 3-Cut, six qutrits), with the $l=1$ and $l=2$ operator pools.

```
local_minima/
├── local_minima.ipynb     regenerates Figs. 8–9 and the numbers of Sec. III C
├── figures/               the figures written by the notebook (one panel per file)
├── graphs/                the graph drawings used as insets
└── data/
    ├── G1_l1.json   G1_l2.json
    └── K6_l1.json   K6_l2.json
```

Run it with

```bash
jupyter lab local_minima/local_minima.ipynb
```

The figures and tables take a few seconds. The last cell recomputes the gradient variance at the random
initial points with a small self-contained simulator and takes a few minutes. Only `numpy` and
`matplotlib` are needed. If LaTeX with the `txfonts` package is available, the figures are rendered exactly
as in the manuscript; otherwise the STIX fonts are used.

## What was computed

Each file is one Qudit-ADAPT run. At every iteration $k$ the $k$-parameter ansatz chosen by the algorithm
was re-optimized from three starting points, with the operator sequence fixed:

| Field | Starting point |
|---|---|
| `recycled_energy[k]` | warm start $\boldsymbol\theta_k=(\boldsymbol\theta^*_{k-1},0)$, as in Qudit-ADAPT |
| `cold_energy[k]` | cold restart $\boldsymbol\theta_k=\boldsymbol 0_k$ |
| `random_runs[k]` | 100 random initializations $\theta_j\sim\mathcal U(-\pi,\pi)$, each with its initial point `x0` and converged `final_energy` |

Other fields: `edges`, `ground_energy`, `initial_problem_energy`, the pool (`pool_labels`) and the selected
operators (`ansatz_op_labels`, `params`), gradient traces, and the run settings (`epsilon = 1e-2`,
`n_random = 100`, `seed = 0`). Operator labels are products of angular-momentum factors `(site, axis)`; the
generator is their Hermitian part, and the ansatz is
$e^{-i\theta_kA_k}\cdots e^{-i\theta_1A_1}\lvert +_3\rangle^{\otimes n}$.

For $l=2$, `pool_labels` lists the $l=1$ operators twice, once from $O_1$ and once from $O_3$. Duplicates have
identical gradients and never change the selected operator; the number of distinct operators is
`len(set(pool_labels))` (2658 for $K_6$, 1138 for $G_1$).

## Provenance

Produced on the BitWit workstation by H. Díaz-Moraga with `cluster/main_bp.py` (engine in
`funciones/utilidades_bp.py`), with the commands listed in `REPRODUCING.md`. These files are the same runs as
`resultados/json/fig1a_l{1,2}_bp100.json` and `resultados/json/k6_l{1,2}_bp100.json`, with the graph named
explicitly and two internal bookkeeping fields removed.
