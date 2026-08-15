# Qudit-ADAPT

Reference implementation and data for

> **Qudit-ADAPT-VQE: an adaptive variational algorithm with counterdiabatic-inspired improvements for qudits**
> J. Molina, H. Díaz-Moraga, D. Goyeneche, D. Tancara
> Facultad de Física, Pontificia Universidad Católica de Chile

The code solves **Max 3-Cut** on qutrits ($d=3$, spin-1 representation) with two
algorithms and compares them:

- **Qudit-ADAPT** — ADAPT-VQE whose operator pool comes from an approximate
  adiabatic gauge potential, built from nested commutators of the mixer and
  cost Hamiltonians and truncated at order $\ell$.
- **Qudit QAOA** — the fixed-ansatz baseline.

Every figure and table in the paper can be regenerated from this repository.
See [`REPRODUCING.md`](REPRODUCING.md) for the exact command behind each one,
and [`DATA.md`](DATA.md) for where each dataset was produced and by whom.

*(Versión en español: [`README.es.md`](README.es.md).)*

---

## Start here

New to the codebase? The `examples/` notebooks are the intended entry point.
They run in seconds on a laptop, each one is self-contained, and together they
cover the whole pipeline:

| Notebook | What it shows |
|---|---|
| [`01_max3cut_and_hamiltonians`](examples/01_max3cut_and_hamiltonians.ipynb) | The problem: ternary variables, $H_C$, $H_M$, and the exact solution of a six-vertex graph by brute force |
| [`02_operator_pool`](examples/02_operator_pool.ipynb) | Where the pool comes from: nested commutators, why Hermitization is needed, $\ell=1$ vs $\ell=2$ |
| [`03_running_qudit_adapt`](examples/03_running_qudit_adapt.ipynb) | One full ADAPT run on $G_1$: operator selection, warm start, convergence |
| [`04_native_gate_count`](examples/04_native_gate_count.ipynb) | Appendix B: compiling the ansatz to trapped-ion native gates, and the QAOA comparison |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab examples/
```

## Layout

```
funciones/      algorithms — all the physics lives here
cluster/        runnable scripts: one experiment or one figure each
examples/       annotated notebooks, start here
cuadernillos/   the analysis notebooks that produced the paper figures
datos/          input graphs
resultados/     outputs: json (full traces), csv (summaries), images, logs
papers/         background literature
```

Nothing in `cluster/` or `cuadernillos/` implements physics — they import from
`funciones/` and orchestrate. The dependency graph is a tree, no cycles:

```
utilidades.py  ──────────┬──> utilidades_QAOA.py
                         └──> utilidades_bp.py ──┬──> utilidades_gellmann.py
                                                 └──> utilidades_ringbauer.py
```

### `funciones/` — the engine

| Module | Contents |
|---|---|
| `utilidades.py` | Qutrit operators, $H_C$ and $H_M$, symbolic nested commutators, the original CD-ADAPT loop |
| `utilidades_QAOA.py` | Qudit QAOA: layer construction, energy, depth scan with warm start |
| `utilidades_bp.py` | Vectorized ADAPT engine with analytic gradients, the local-minimum landscape scan, and native gate counting |
| `utilidades_gellmann.py` | The same pool rebuilt in the $\mathfrak{su}(3)$ Gell-Mann basis (Appendix B) |
| `utilidades_ringbauer.py` | Decomposition of a single-qudit unitary into two-level rotations, following Ringbauer *et al.* |

### `cluster/` — experiments

| Script | Produces |
|---|---|
| `main_comparaciones.py` | Qudit-ADAPT on the benchmark graphs → Figs. 1–3, Table I |
| `main_QAOA.py` | QAOA baseline on the same graphs → Figs. 1–3, Table I |
| `main.py` | The 300-random-graph ensemble → Fig. 4 |
| `main_bp.py` | Warm/cold/random-restart landscape scan → Figs. 5–6 |
| `figuras_paper.py` | Renders Figs. 5–6 in the manuscript's typography |
| `tablas_apendice.py` | Tables II and III |
| `conteo_qaoa.py` | Table IV |
| `lanzar_bp.sh` | Launcher for long unattended runs |

## Conventions

Three things are easy to get wrong when reading the code:

**Sites are 1-indexed.** Graph edges are `(1,2)`, not `(0,1)`. Levels inside a
qutrit are 0-indexed, so `lambda_1` acts on the pair `(0,1)`.

**Operator labels are canonical strings.** A pool operator is identified by a
string like `"((1, 'y'), (3, 'x'), (4, 'z'))"` for angular momentum or
`"((2, 2), (3, 8), (4, 6))"` for Gell-Mann. The pool is sorted by that string,
so operator indices are reproducible across runs.

**The ansatz applies right to left.** `ansatz_op_labels[0]` is the operator
closest to the reference state $|\phi_g\rangle$, i.e. applied first.

Gradients are analytic (adjoint/backpropagation), not finite differences:
$\partial E/\partial\theta_j = 2\,\mathrm{Im}\langle\sigma_j|A_j|\varphi_j\rangle$.
This is what lets the $\ell=2$ runs reach machine precision.

## Language

Code comments are in Spanish, matching the authors. Documentation, examples and
scripts added for the public release are in English. Both READMEs are kept in
sync; the English one is authoritative.

## Citation

```bibtex
@article{molina2026quditadapt,
  title   = {Qudit-ADAPT-VQE: an adaptive variational algorithm with
             counterdiabatic-inspired improvements for qudits},
  author  = {Molina, Joaqu\'in and D\'iaz-Moraga, Herbert and
             Goyeneche, Dardo and Tancara, Diego},
  year    = {2026}
}
```
