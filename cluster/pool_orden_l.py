"""
Tamaño del pool contradiabático de K_n para cualquier orden l, contado por soporte.

Construye simbólicamente O_1, O_3, ..., O_{2l-1} para n chico y cuenta los
operadores distintos con soporte en w sitios. Por la simetría de K_n bajo
permutaciones, cada subconjunto de w sitios aporta el mismo número t_w, así que
N_l(n) = sum_w t_w C(n, w), y basta un n >= 2l para conocer todos los t_w.

    python cluster/pool_orden_l.py --l 4 --n 2 3 4 5
"""
import argparse, ast, json, sys, time
from math import comb
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import sympy as sp
from funciones.utilidades import Had, dHad_dlam, nested_commutators, canonical_op

p = argparse.ArgumentParser()
p.add_argument("--l", type=int, required=True)
p.add_argument("--n", type=int, nargs="+", required=True)
args = p.parse_args()
lam = sp.symbols("lam", real=True)
salida = PROJECT_ROOT / "resultados" / "json" / f"pool_l{args.l}.json"
filas = json.load(open(salida)) if salida.exists() else []
for n in args.n:
    edges = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
    t = time.time()
    res = nested_commutators(Had(n, edges, lam), dHad_dlam(n, edges), order=2 * args.l - 1)
    ops = set()
    for k in range(1, 2 * args.l, 2):
        ops |= {str(canonical_op(op)) for op in res[k].keys()}
    por = {}
    for x in ops:
        w = len({a for a, _ in ast.literal_eval(x)})
        por[w] = por.get(w, 0) + 1
    t_w = {w: c / comb(n, w) for w, c in sorted(por.items())}
    filas.append({"l": args.l, "n": n, "total": len(ops), "por_soporte": por, "t_w": t_w,
                  "t_s": time.time() - t})
    print(n, len(ops), "t_w:", t_w, f"{time.time()-t:.0f} s", flush=True)
    json.dump(filas, open(salida, "w"), indent=1)
