"""
Gradient variance of the Qudit-ADAPT ansatz against system size.

This is the measurement the barren-plateau argument of the manuscript is
missing. There we showed that the variance of dE/dtheta is essentially flat
against *ansatz depth* at fixed n = 6. But barren plateaus are a *system-size*
phenomenon: the claim to test is whether

    Var[dE/dtheta] ~ exp(-alpha n)

as the number of qudits grows. Ref. [25] of the paper reports Var ~ d^{-n} for
highly expressive qudit ansaetze, so the relevant question is whether the
counterdiabatic pool inherits that decay or escapes it.

Multiway number partitioning is a good testbed: it is a genuine optimization
problem, its interaction graph is complete (the densest case, hence the least
favourable for us), and instance hardness is tunable through the range of the
numbers.

METHOD
------
For each size n we draw several random instances. For each instance we build an
ansatz of k operators drawn uniformly from the counterdiabatic pool, sample
parameters uniformly in [-pi, pi], and evaluate the exact gradient. Drawing the
operators at random rather than by the ADAPT criterion is deliberate: a barren
plateau is a property of the ansatz *family*, not of one selected circuit, and
this isolates the pool itself.

H_C is normalized by its spectral range. This is not cosmetic: the energy scale
of the partition Hamiltonian grows like (sum a)^2, i.e. ~n^2, so an unnormalized
variance would rise with n from scale alone and hide any exponential decay.

    python cluster/varianza_mwnp.py
    python cluster/varianza_mwnp.py --n_max 12 --instancias 5
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import time

import numpy as np

from funciones.utilidades_mwnp import (
    estado_referencia,
    hamiltoniano_diag,
    energia_y_grad,
    preparar_pool,
)

SALIDA = PROJECT_ROOT / "resultados" / "json" / "varianza_mwnp.json"


def una_medicion(a, pool, k, n_muestras, rng):
    """Var(dE/dtheta) de un ansatz de k operadores al azar, sobre una instancia."""
    n = len(a)
    hdiag = hamiltoniano_diag(a)
    rango = float(hdiag.max() - hdiag.min())
    hdiag = hdiag / rango                      # escala espectral 1
    psi0 = estado_referencia(n)

    ops = [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
    grads = np.empty((n_muestras, k))
    for s in range(n_muestras):
        theta = rng.uniform(-np.pi, np.pi, size=k)
        _, grads[s] = energia_y_grad(theta, ops, psi0, hdiag, n)

    return {
        "var_total": float(np.var(grads)),
        "var_primera": float(np.var(grads[:, 0])),
        "abs_medio": float(np.mean(np.abs(grads))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_min", type=int, default=6)
    p.add_argument("--n_max", type=int, default=14)
    p.add_argument("--instancias", type=int, default=3)
    p.add_argument("--muestras", type=int, default=80)
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--rango", type=int, default=100, help="a_i ~ enteros en [1, rango]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--salida", type=str, default=str(SALIDA))
    args = p.parse_args()

    filas = []
    t_global = time.time()
    print(f"{'n':>3s}{'k':>4s}{'|pool|':>8s}{'Var[dE/dth]':>14s}"
          f"{'Var 1a comp':>14s}{'|dE/dth|':>12s}{'t':>8s}", flush=True)

    for n in range(args.n_min, args.n_max + 1):
        pool = preparar_pool(n, l=1)
        for k in args.ks:
            if k > len(pool):
                continue
            t0 = time.time()
            acum = []
            for r in range(args.instancias):
                # Semilla por (n, k, instancia): reproducible y sin correlación
                rng = np.random.default_rng([args.seed, n, k, r])
                a = rng.integers(1, args.rango + 1, size=n).tolist()
                acum.append(una_medicion(a, pool, k, args.muestras, rng))

            fila = {
                "n": n, "k": k, "pool_size": len(pool),
                "instancias": args.instancias, "muestras": args.muestras,
                "var_total": float(np.mean([x["var_total"] for x in acum])),
                "var_total_std": float(np.std([x["var_total"] for x in acum])),
                "var_primera": float(np.mean([x["var_primera"] for x in acum])),
                "abs_medio": float(np.mean([x["abs_medio"] for x in acum])),
                "runtime_s": time.time() - t0,
            }
            filas.append(fila)
            print(f"{n:3d}{k:4d}{len(pool):8d}{fila['var_total']:14.4e}"
                  f"{fila['var_primera']:14.4e}{fila['abs_medio']:12.4e}"
                  f"{fila['runtime_s']:7.1f}s", flush=True)

            Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
            with open(args.salida, "w", encoding="utf-8") as f:
                json.dump({"config": vars(args), "filas": filas}, f, indent=2)

    # Ajuste exponencial Var ~ C exp(-alpha n), por cada k
    print(f"\nAjuste Var ~ C exp(-alpha n)   [alpha > 0 significa decaimiento]")
    for k in args.ks:
        sub = [f for f in filas if f["k"] == k and f["var_total"] > 0]
        if len(sub) < 3:
            continue
        ns = np.array([f["n"] for f in sub], float)
        lv = np.log(np.array([f["var_total"] for f in sub], float))
        alpha, c = np.polyfit(ns, lv, 1)
        # d^{-n} con d = 3 corresponde a alpha = ln 3
        print(f"  k={k:3d}: alpha = {-alpha:+.4f}   (Var ~ {np.exp(-alpha):.3f}^-n)"
              f"   |  referencia d^-n: alpha = {np.log(3):.4f}")

    print(f"\nlisto en {(time.time()-t_global)/60:.1f} min -> {args.salida}")


if __name__ == "__main__":
    main()
