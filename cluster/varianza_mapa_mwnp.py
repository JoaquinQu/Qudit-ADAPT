"""
Mapa de la varianza del gradiente contra número de qutrits n y profundidad k.

Reemplaza la medición anterior (3 instancias aleatorias con números en [1, 100]
y k = 5, 10, 20) por una sobre las instancias del benchmark:

  * n = 5..12, k = 1, 2, 4, 8, 16, 32, 64;
  * 20 instancias por n (las del benchmark, orden ascendente);
  * dos familias de circuitos de k operadores:
      "aleatoria": k operadores sorteados del pool (l = 1 o l = 2), la familia
                   que importa para la pregunta de barren plateaus del pool;
      "adapt":     los primeros k operadores que eligió ADAPT en la corrida real
                   de esa instancia (orden 0), el circuito que de verdad se usa;
  * 100 puntos de parámetros uniformes en [-pi, pi]^k por circuito, gradiente
    exacto (método adjunto);
  * H_C normalizado por su rango espectral, para que la escala de energía no
    crezca con n.

Para cada circuito se guarda Var[dE/dtheta_j] promediada sobre j (y la del
primer parámetro, y |dE/dtheta| medio). La unidad estadística es la instancia:
se resumen con mediana y rango intercuartil sobre las 20.

    python cluster/varianza_mapa_mwnp.py --procesos 16 --nucleos 0,1,8,9,...
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import multiprocessing
import time

import numpy as np

from funciones.utilidades_mwnp import (energia_y_grad, estado_referencia, etiquetas_pool,
                                       hamiltoniano_diag, operadores_desde_etiquetas)

KS = (1, 2, 4, 8, 16, 32, 64)
MUESTRAS = 100


def fijar_nucleo(contador, nucleos):
    with contador.get_lock():
        i = contador.value
        contador.value += 1
    os.sched_setaffinity(0, {nucleos[i % len(nucleos)]})


def rango(texto):
    fuera = []
    for t in texto.split(","):
        if "-" in t:
            a, b = t.split("-")
            fuera += list(range(int(a), int(b) + 1))
        else:
            fuera.append(int(t))
    return fuera


def secuencia_adapt(n, l, base, carpetas):
    """Operadores elegidos por ADAPT (orden 0 de la instancia `base`)."""
    for c in carpetas:
        f = Path(c) / f"n{n}_l{l}_i{base * 100:02d}.json"
        if f.exists():
            return json.load(open(f, encoding="utf-8"))["trazas"]["operadores"]
    return None


def tarea(args):
    n, l, familia, inst, carpeta, carpetas_adapt = args
    salida = Path(carpeta) / f"n{n}_l{l}_{familia}_i{inst['id']:02d}.json"
    if salida.exists():
        return (n, l, familia, inst["id"], "ya existía")
    t0 = time.time()
    h = hamiltoniano_diag(inst["a"])
    h = h / (h.max() - h.min())
    psi0 = estado_referencia(n)
    rng = np.random.default_rng([n, l, inst["id"], 0 if familia == "aleatoria" else 1])
    if familia == "aleatoria":
        etiquetas = etiquetas_pool(n, l)
        sec = [etiquetas[i] for i in rng.choice(len(etiquetas), max(KS), replace=True)]
    else:
        sec = secuencia_adapt(n, l, inst["id"], carpetas_adapt)
        if sec is None:
            return (n, l, familia, inst["id"], "sin corrida ADAPT")
    filas = []
    for k in KS:
        if k > len(sec):
            break
        ops = operadores_desde_etiquetas(sec[:k])
        G = np.empty((MUESTRAS, k))
        for s in range(MUESTRAS):
            _, G[s] = energia_y_grad(rng.uniform(-np.pi, np.pi, k), ops, psi0, h, n)
        filas.append({"k": k, "var": float(np.mean(G.var(axis=0))), "var_primero": float(G[:, 0].var()),
                      "abs_medio": float(np.mean(np.abs(G)))})
    json.dump({"n": n, "l": l, "familia": familia, "instancia": inst["id"], "a": inst["a"],
               "muestras": MUESTRAS, "normalizacion": "H_C / (max - min)", "filas": filas,
               "t_s": time.time() - t0},
              open(salida, "w", encoding="utf-8"), indent=1)
    return (n, l, familia, inst["id"], f"ok {time.time() - t0:.0f} s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=list(range(5, 13)))
    p.add_argument("--procesos", type=int, default=8)
    p.add_argument("--nucleos", type=str, default=None)
    p.add_argument("--carpeta", type=str, default=str(PROJECT_ROOT / "resultados" / "varianza_mapa"))
    p.add_argument("--adapt", type=str, nargs="+", required=True,
                   help="carpetas con las corridas ADAPT (n <= 9 y n >= 10)")
    args = p.parse_args()
    Path(args.carpeta).mkdir(parents=True, exist_ok=True)
    inst = [i for i in json.load(open(PROJECT_ROOT / "datos" / "mwnp_instancias_5a12.json"))["instancias"]
            if i["n"] in args.n]
    for n in args.n:                        # las etiquetas del pool, una vez, antes de repartir
        for l in (1, 2):
            etiquetas_pool(n, l)
    tareas = [(i["n"], l, fam, i, args.carpeta, args.adapt) for i in inst for l in (1, 2)
              for fam in ("aleatoria", "adapt")]
    tareas.sort(key=lambda t: -t[0])
    ctx = multiprocessing.get_context("fork")
    extra = {}
    if args.nucleos:
        extra = {"initializer": fijar_nucleo, "initargs": (ctx.Value("i", 0), rango(args.nucleos))}
    print(f"{len(tareas)} tareas", flush=True)
    with ctx.Pool(args.procesos, **extra) as pool:
        for k, r in enumerate(pool.imap_unordered(tarea, tareas), 1):
            print(f"[{k}/{len(tareas)}] n={r[0]} l={r[1]} {r[2]} #{r[3]} {r[4]}", flush=True)


if __name__ == "__main__":
    main()
