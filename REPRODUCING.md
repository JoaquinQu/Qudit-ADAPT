# Reproducing the figures and tables

One command per figure. Runtimes are for the machines listed in
[`DATA.md`](DATA.md); everything except Figs. 5–6 finishes in minutes.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All scripts are run from the repository root.

---

## Figures 1–3 and Table I — Qudit-ADAPT vs QAOA

```bash
python Max-3-Cut/cluster/main_comparaciones.py
python Max-3-Cut/cluster/main_QAOA.py
jupyter nbconvert --execute --inplace Max-3-Cut/qaoa_comparison.ipynb
```

The two scripts write to `Max-3-Cut/results/{json,csv}/`; the notebook reads them and
draws the panels. `main_QAOA.py` is the slow one — 25 random restarts at every
depth up to $p=20$, for six graphs.

## Figure 4 — 300-graph ensemble

```bash
python Max-3-Cut/cluster/main.py
jupyter nbconvert --execute --inplace Max-3-Cut/random_graphs_n6.ipynb
```

Figure 4 uses the common-budget runs ($k_{\max}=50$), so that the $\ell=1$ and
$\ell=2$ pools are compared under the same ansatz-growth budget rather than at
their own convergence points.

## Figures 5–6 — local minima and warm start

The heavy ones. Each ADAPT iteration re-optimizes the same ansatz from 100
independent random initializations, so cost grows quadratically with the final
ansatz size.

```bash
# G_1  (Fig. 5a and 5b)
python local_minima/cluster/main_bp.py --input_file grafos_comparacion.txt --grafo 1 --l 1 \
    --n_random 100 --max_iteration 40 --output fig1a_l1_bp100.json
python local_minima/cluster/main_bp.py --input_file grafos_comparacion.txt --grafo 1 --l 2 \
    --n_random 100 --max_iteration 40 --output fig1a_l2_bp100.json

# K_6  (Fig. 6a and 6b)
python local_minima/cluster/main_bp.py --input_file grafo_completo_n6.txt --grafo 1 --l 1 \
    --n_random 100 --max_iteration 40 --output k6_l1_bp100.json
python local_minima/cluster/main_bp.py --input_file grafo_completo_n6.txt --grafo 1 --l 2 \
    --n_random 100 --max_iteration 40 --output k6_l2_bp100.json

python local_minima/cluster/figuras_paper.py
```

`--input_file` is resolved against the root `data/` (also not tracked; it keeps local copies
of `grafos_comparacion.txt` and `grafo_completo_n6.txt` for this script's sake), and
`main_bp.py` always writes under the root `results/json/` (not tracked either). The published
copy of the four runs above lives in `local_minima/data/{G1,K6}_l{1,2}.json`, with the graph
named explicitly and two internal bookkeeping fields removed — see
[`local_minima/README.md`](local_minima/README.md).

`main_bp.py` checkpoints after every iteration and resumes from the checkpoint
by default, so a run can be interrupted and restarted without losing work. Pass
`--no_resume` to start over. For unattended execution:

```bash
bash local_minima/cluster/lanzar_bp.sh todas    # launches the four runs in the background
bash local_minima/cluster/lanzar_bp.sh estado   # progress
bash local_minima/cluster/lanzar_bp.sh detener  # stop them
```

`figuras_paper.py` renders with `text.usetex=True` and the `txfonts` package,
matching the manuscript. That needs a LaTeX installation with `cm-super`; set
`USETEX = False` at the top of the script to fall back to STIX, which is close
enough for drafts.

## Table I

The approximation ratios come from the runs above. The $G_1$ rows and the
parameter counts are printed by

```bash
python local_minima/cluster/figuras_paper.py     # prints k, pool size and final errors
```

## Appendix B — Tables II and III

```bash
python local_minima/cluster/main_bp.py --input_file grafos_comparacion.txt --grafo 2 --l 2 \
    --n_random 20 --max_iteration 30 --base angular \
    --output bp_n6_l2_grafo2_r20_k30.json
python local_minima/cluster/main_bp.py --input_file grafos_comparacion.txt --grafo 2 --l 2 \
    --n_random 20 --max_iteration 30 --base gellmann \
    --output bp_n6_l2_grafo2_r20_k30_gellmann.json

python local_minima/cluster/tablas_apendice.py            # text
python local_minima/cluster/tablas_apendice.py --latex    # rows ready to paste
```

Same note as above: the two runs land in the root `results/json/`; the published copy is
`local_minima/data/bp_n6_l2_grafo2_r20_k30[_gellmann].json`.

## Appendix B — Table IV

No run needed. The native gate count of a QAOA circuit depends only on the
graph and the number of layers, so no optimization and no optimal angles are
involved:

```bash
python local_minima/cluster/conteo_qaoa.py
```

It also verifies numerically the identity of Eq. (B4),
$H_C|_{(i,j)} = \lambda_3\lambda_3 + \lambda_8\lambda_8 - \tfrac43 I$,
before using it.

---

## If a number does not match

Two things move results legitimately:

**Degenerate gradients.** On symmetric instances many pool operators tie for
the largest gradient — on $K_6$ at $k=1$ there are only 11 distinct gradient
values among 2724 operators, and 90 operators tie for the maximum. Which one
`argmax` returns can differ between BLAS builds. The operators are equivalent
by symmetry and the energy traces agree, but the selected indices may not.

**The optimizer.** `main_bp.py` uses analytic gradients with BFGS. Finite
differences converge to visibly worse energies on the $\ell=2$ pools — that
difference is what separates machine precision from $\sim10^{-5}$ on $K_6$.
