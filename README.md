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
functions/                     the engine — all the physics lives here, one package
examples/                      annotated notebooks, start here
Max-3-Cut/                     Figs. 1-4 and Table I, self-contained (see its own README)
local_minima/                  Figs. 5-6 and Sec. III C, self-contained (see its own README)
multiway_number_partitioning/  Sec. III B, self-contained (see its own README)
```

Each of the three topic folders above has its own `cluster/` with the scripts that produce it
(e.g. `Max-3-Cut/cluster/main_comparaciones.py`) — see that folder's own README for the list, or
[`REPRODUCING.md`](REPRODUCING.md) for the exact command behind every figure.

The root also has `data/`, `results/` and `cluster/` folders, none part of the public repo: they are
where `main_bp.py` and the mwnp scripts read and write by default (most scripts resolve paths
relative to the repository root, not to wherever they happen to live), and `Max-3-Cut/`,
`local_minima/` and `multiway_number_partitioning/` ship the tracked, cleaned copies of whatever they
actually need.

(`multiway number partition/` — an older, exploratory take on multiway, in Spanish, with a
known broken import — is kept on disk but gitignored; `multiway_number_partitioning/` above
is the one that is part of the paper.)

Nothing in any `cluster/` subfolder implements physics — they import from `functions/` and
orchestrate. The dependency graph is a tree, no cycles:

```
utilidades.py  ──────────┬──> utilidades_QAOA.py
                         └──> utilidades_bp.py ──┬──> utilidades_gellmann.py
                                                 └──> utilidades_ringbauer.py
```

### `functions/` — the engine

| Module | Contents |
|---|---|
| `utilidades.py` | Qutrit operators, $H_C$ and $H_M$, symbolic nested commutators, the original CD-ADAPT loop |
| `utilidades_QAOA.py` | Qudit QAOA: layer construction, energy, depth scan with warm start |
| `utilidades_bp.py` | Vectorized ADAPT engine with analytic gradients, the local-minimum landscape scan, and native gate counting |
| `utilidades_gellmann.py` | The same pool rebuilt in the $\mathfrak{su}(3)$ Gell-Mann basis (Appendix B) |
| `utilidades_ringbauer.py` | Decomposition of a single-qudit unitary into two-level rotations, following Ringbauer *et al.* |

`utilidades.py` and `utilidades_QAOA.py` were translated to English, since they are what
`Max-3-Cut/`'s notebooks import and the one piece of the paper-facing code meant to be read
start to end by someone outside the group. The rest of the module keeps its Spanish
comments. There used to be a second folder, `funciones/`, left behind as a compatibility
shim while only `utilidades.py`/`utilidades_QAOA.py` had moved; it has since been folded
back in — every module that depends on the engine now imports from `functions/` directly,
`funciones/` no longer exists.

### `<topic>/cluster/` — experiments

| Script | Produces |
|---|---|
| `Max-3-Cut/cluster/main_comparaciones.py` | Qudit-ADAPT on the benchmark graphs → Figs. 1–3, Table I |
| `Max-3-Cut/cluster/main_QAOA.py` | QAOA baseline on the same graphs → Figs. 1–3, Table I |
| `Max-3-Cut/cluster/main.py` | The 300-random-graph ensemble → Fig. 4 |
| `local_minima/cluster/main_bp.py` | Warm/cold/random-restart landscape scan → Figs. 5–6 (writes to the root `data/`/`results/`; its output is what `local_minima/` ships a cleaned copy of) |
| `local_minima/cluster/figuras_paper.py` | Renders Figs. 5–6 in the manuscript's typography |
| `local_minima/cluster/tablas_apendice.py` | Tables II and III |
| `local_minima/cluster/conteo_qaoa.py` | Table IV |
| `local_minima/cluster/lanzar_bp.sh` | Launcher for long unattended runs |

Scripts resolve paths relative to the repository root (`Path(__file__).resolve().parent.parent.parent`),
not to wherever they happen to live, so moving a script under its topic folder's `cluster/` never
breaks its own path logic. `main_comparaciones.py`, `main_QAOA.py`, `main.py`, `main_n5.py` and
`run_kn.py` read their input graphs from `Max-3-Cut/data/` and write to `Max-3-Cut/results/`.
`main_bp.py` (and everything feeding `local_minima/`, `multiway_number_partitioning/` or
Appendix B) still resolves relative paths against the root `data/`/`results/`. Neither root
folder is part of the public repo, though: every file that is actually published — the fixed
benchmark graphs, the figures, the run data — has a tracked copy inside the topic folder it
belongs to (`Max-3-Cut/data/`, `local_minima/data/`, ...), and the root `data/`/`results/` only
hold the untracked local copies these scripts need to run.

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

Exception: `functions/utilidades.py` and `functions/utilidades_QAOA.py` — the engine
behind `Max-3-Cut/` — were translated to English, since that is the one piece of the
paper-facing code meant to be read start to end by someone outside the group.
`functions/utilidades_bp.py`, `utilidades_gellmann.py` and `utilidades_ringbauer.py` keep
their Spanish comments, unchanged.

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
