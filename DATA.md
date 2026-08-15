# Data provenance and run environment

Every dataset in `resultados/` is listed below with who produced it, on which
machine, and which figure or table of the paper it supports. Runs marked
*thesis material* are not used by the manuscript; they belong to the B.Sc.
thesis this repository also accompanies.

## Machines

**Laptop** — 8 cores, 11 GB RAM, Ubuntu, Python 3.12.
Used for the benchmark and ensemble runs, and for all figure rendering.

**BitWit** — AMD Ryzen Threadripper 7980X (64 cores / 128 threads), 125 GB RAM,
Debian GNU/Linux 13 (trixie), Python 3.12.
Used for the landscape scans of Figs. 5–6, which re-optimize 100 independent
random initializations at every ADAPT iteration and are the only runs in the
paper heavy enough to need it.

Both use the same pinned dependencies (`requirements.txt`); the numbers are
machine-independent up to floating-point noise. The scans were run with
`--n_jobs 1`: the linear algebra is already threaded through BLAS, and adding
process-level parallelism on top of it made things slower, not faster.

## Datasets

### Figures 1–3 and Table I — benchmark instances

Qudit-ADAPT and QAOA on $G_1$–$G_4$ (irregular) and $G_5$, $K_6$ (regular).

| File | Produced by | Where | Script |
|---|---|---|---|
| `resultados/json/resultados_completos_comparacion_n6_l{1,2}.json` | J. Molina | laptop | `cluster/main_comparaciones.py` |
| `resultados/json/resultados_completos_regulares_n6_l{1,2}.json` | J. Molina | laptop | `cluster/main_comparaciones.py` |
| `resultados/json/resultados_qaoa_n6_p20_jx_r25_grafos_1_all.json` | J. Molina | laptop | `cluster/main_QAOA.py` |
| `resultados/json/resultados_qaoa_regulares_n6_p20_jx_r25_grafos_1_4.json` | J. Molina | laptop | `cluster/main_QAOA.py` |

QAOA uses $p=20$ layers and 25 random restarts per depth; the median is
plotted and the shaded band is the interquartile range.

### Figure 4 — 300-graph ensemble

| File | Produced by | Where | Script |
|---|---|---|---|
| `resultados/json/resultados_completos_n6_l{1,2}_1_300.json` | J. Molina | laptop | `cluster/main.py` |
| `resultados/json/resultados_completos_n6_l{1,2}_common_budget.json` | J. Molina | laptop | `cluster/main.py` |

`_1_300` runs to convergence; `_common_budget` caps every instance at
$k_{\max}=50$ ADAPT iterations so the two pools are compared under the same
ansatz-growth budget. Figure 4 uses the common-budget runs.

### Figures 5–6 — local minima and warm start

At each ADAPT iteration the same $k$-parameter ansatz is re-optimized from the
warm start, from a cold restart at $\boldsymbol\theta=0$, and from 100
independent random initializations.

| File | Produced by | Where | Runtime | Script |
|---|---|---|---|---|
| `resultados/json/fig1a_l1_bp100.json` | H. Díaz-Moraga | BitWit | 0.1 min | `cluster/main_bp.py` |
| `resultados/json/fig1a_l2_bp100.json` | H. Díaz-Moraga | BitWit | 139 min | `cluster/main_bp.py` |
| `resultados/json/k6_l1_bp100.json` | H. Díaz-Moraga | BitWit | 189 min | `cluster/main_bp.py` |
| `resultados/json/k6_l2_bp100.json` | H. Díaz-Moraga | BitWit | 24 min | `cluster/main_bp.py` |

All four use `--n_random 100 --seed 0 --epsilon 1e-2`. Console logs are kept in
`resultados/logs/`. Rendered with `cluster/figuras_paper.py`.

Note that $K_6$ with $\ell=2$ is the *fastest* of the three heavy runs despite
having the largest pool (2724 operators): it converges after 18 parameters,
while $K_6$ with $\ell=1$ needs 31 and never reaches machine precision.

### Table I parameter counts

| File | Produced by | Where | Script |
|---|---|---|---|
| `resultados/json/fig1a_l{1,2}_curva.json` | H. Díaz-Moraga | BitWit | `cluster/main_bp.py --n_random 0` |

Convergence curves without the random-restart cloud, used for the $G_1$ rows of
Table I and as the ADAPT side of Table IV.

### Appendix B — Tables II, III, IV

| File | Produced by | Where | Script |
|---|---|---|---|
| `resultados/json/bp_n6_l2_grafo2_r20_k30.json` | H. Díaz-Moraga | BitWit | `cluster/main_bp.py --base angular` |
| `resultados/json/bp_n6_l2_grafo2_r20_k30_gellmann.json` | H. Díaz-Moraga | BitWit | `cluster/main_bp.py --base gellmann` |

Same instance ($G_2$, $\ell=2$), same algorithm, only the basis in which the
nested commutators are decomposed differs. Table IV needs no run at all: the
native gate count of a QAOA circuit is fixed by the graph and the number of
layers, so `cluster/conteo_qaoa.py` computes it directly.

### Thesis material — not used by the paper

Kept because this repository also accompanies J. Molina's B.Sc. thesis
(`Tesis_Joaquín_Molina.pdf`).

| File | Contents |
|---|---|
| `resultados/json/resultados_completos_n5_l{1,2}_*.json` | The same ensemble at $n=5$ |
| `resultados/json/kn_l{1,2}_n4-10.json` | Scaling on complete graphs $K_n$, $n=4\ldots10$ |
| `resultados/json/resultados_l2_controlado_por_iteraciones_l1.json` | $\ell=2$ capped at the iteration count $\ell=1$ needed |
| `resultados/images/{01..11}_*.png`, `cb_*.png` | Ensemble analysis plots |
| `cuadernillos/analisis_kn.ipynb`, `analisis_localidad.ipynb` | Their analysis |

## Exploratory work not in the paper

The `exploratorio` branch preserves the material that was investigated and
left out: a hardware-efficient-ansatz positive control, the Heisenberg–Weyl
operator pool, an ensemble comparison across pools, and a local-minimum audit
of the $K_6$ run. Its commit message lists what is worth resuming — most
importantly the gradient variance as a function of $n$ rather than of circuit
depth, which is the measurement the barren-plateau argument would need to
become conclusive.
