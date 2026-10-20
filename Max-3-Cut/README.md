# Qudit-ADAPT vs QAOA on Max 3-Cut (Figs. 1–4, Table I)

Data and analysis notebooks for the main benchmark of the paper: Qudit-ADAPT ($l=1,2$ pools) against
qudit QAOA, on six qutrits ($n=6$), both on five fixed graphs ($G_1$–$G_4$ irregular, $G_5$ and $K_6$
regular) and on an ensemble of 300 random graphs.

```
Max-3-Cut/
├── qaoa_comparison.ipynb     regenerates Figs. 1–3 and Table I, on G1–G5 and K6
├── random_graphs_n6.ipynb    regenerates Fig. 4, on the 300-graph ensemble
├── data/
│   ├── grafos_comparacion.txt, grafos_n5.txt, grafos_n6.txt    the fixed benchmark graphs
│   ├── grafos_regulares.txt, grafos_regulares_grado{4,5}.txt   regular graphs by degree
│   └── imagenes/              drawings of the benchmark graphs, used as insets
└── results/
    ├── csv/, json/            Qudit-ADAPT and QAOA runs, one file per experiment (see DATA.md),
    │                            plus fig1a_l{1,2}_curva.json (G1 row of Table I, from local_minima/)
    └── images/                the figures written by the notebooks
```

Run with

```bash
python cluster/main_comparaciones.py          # Qudit-ADAPT on G1-G5, K6 -> results/{csv,json}/
python cluster/main_QAOA.py                   # QAOA on the same graphs
jupyter nbconvert --execute --inplace Max-3-Cut/qaoa_comparison.ipynb

python cluster/main.py                        # 300-graph ensemble -> results/{csv,json}/
jupyter nbconvert --execute --inplace Max-3-Cut/random_graphs_n6.ipynb
```

See [`../REPRODUCING.md`](../REPRODUCING.md) for the full set of commands and runtimes, and
[`../DATA.md`](../DATA.md) for who produced each file and on which machine.

## What was computed

**`qaoa_comparison.ipynb`** — Qudit-ADAPT ($l=1,2$) and QAOA ($p$ up to 20, 25 random restarts per
depth) on the five benchmark graphs. Figs. 1–3 (energy error, convergence, approximation ratio) and the
parameter counts of Table I.

**`random_graphs_n6.ipynb`** — the same two Qudit-ADAPT pools over 300 random $n=6$ graphs, both run to
free convergence and capped at a common budget of $k_{\max}=50$ ADAPT iterations. Fig. 4 uses the
common-budget runs, so that $l=1$ and $l=2$ are compared under the same ansatz-growth budget rather than
at their own convergence points.

## Provenance

Produced on a laptop by J. Molina. See [`../DATA.md`](../DATA.md) for the exceptions (`fig1a_l{1,2}_curva.json`
and the $K_6$, $l=2$ curve, both from H. Díaz-Moraga's `local_minima/` runs on BitWit).
