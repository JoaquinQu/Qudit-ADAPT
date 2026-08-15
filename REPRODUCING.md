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
python cluster/main_comparaciones.py
python cluster/main_QAOA.py
jupyter nbconvert --execute --inplace cuadernillos/comparacion_QAOA.ipynb
```

The two scripts write to `resultados/{json,csv}/`; the notebook reads them and
draws the panels. `main_QAOA.py` is the slow one — 25 random restarts at every
depth up to $p=20$, for six graphs.

## Figure 4 — 300-graph ensemble

```bash
python cluster/main.py
jupyter nbconvert --execute --inplace cuadernillos/grafos_aleatorios_n6.ipynb
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
python cluster/main_bp.py --input_file grafos_comparacion.txt --grafo 1 --l 1 \
    --n_random 100 --max_iteration 40 --output fig1a_l1_bp100.json
python cluster/main_bp.py --input_file grafos_comparacion.txt --grafo 1 --l 2 \
    --n_random 100 --max_iteration 40 --output fig1a_l2_bp100.json

# K_6  (Fig. 6a and 6b)
python cluster/main_bp.py --input_file grafo_completo_n6.txt --grafo 1 --l 1 \
    --n_random 100 --max_iteration 40 --output k6_l1_bp100.json
python cluster/main_bp.py --input_file grafo_completo_n6.txt --grafo 1 --l 2 \
    --n_random 100 --max_iteration 40 --output k6_l2_bp100.json

python cluster/figuras_paper.py
```

`main_bp.py` checkpoints after every iteration and resumes from the checkpoint
by default, so a run can be interrupted and restarted without losing work. Pass
`--no_resume` to start over. For unattended execution:

```bash
bash cluster/lanzar_bp.sh todas    # launches the four runs in the background
bash cluster/lanzar_bp.sh estado   # progress
bash cluster/lanzar_bp.sh detener  # stop them
```

`figuras_paper.py` renders with `text.usetex=True` and the `txfonts` package,
matching the manuscript. That needs a LaTeX installation with `cm-super`; set
`USETEX = False` at the top of the script to fall back to STIX, which is close
enough for drafts.

## Table I

The approximation ratios come from the runs above. The $G_1$ rows and the
parameter counts are printed by

```bash
python cluster/figuras_paper.py     # prints k, pool size and final errors
```

## Appendix B — Tables II and III

```bash
python cluster/main_bp.py --input_file grafos_comparacion.txt --grafo 2 --l 2 \
    --n_random 20 --max_iteration 30 --base angular \
    --output bp_n6_l2_grafo2_r20_k30.json
python cluster/main_bp.py --input_file grafos_comparacion.txt --grafo 2 --l 2 \
    --n_random 20 --max_iteration 30 --base gellmann \
    --output bp_n6_l2_grafo2_r20_k30_gellmann.json

python cluster/tablas_apendice.py            # text
python cluster/tablas_apendice.py --latex    # rows ready to paste
```

## Appendix B — Table IV

No run needed. The native gate count of a QAOA circuit depends only on the
graph and the number of layers, so no optimization and no optimal angles are
involved:

```bash
python cluster/conteo_qaoa.py
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
