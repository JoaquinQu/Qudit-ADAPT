"""
Benchmark de Qudit-ADAPT sobre multiway number partitioning, por lotes y en paralelo.

Corre ADAPT sobre cada combinación (n, l, instancia) del conjunto fijado en
`datos/mwnp_instancias.json` y guarda UN archivo por corrida en
`resultados/mwnp/n{n}_l{l}_i{id}.json`. Con un archivo por corrida:

  * los procesos en paralelo nunca escriben el mismo archivo;
  * el lote se puede interrumpir y relanzar: lo que ya terminó se salta;
  * agregar la segunda tanda de instancias es sólo cambiar --ids.

PARALELISMO. A n <= 10 el estado pesa menos de 1 MB, así que dar muchos hilos
de BLAS a una sola corrida no rinde: el trabajo por operación es demasiado
chico. Rinde mucho más correr varias instancias a la vez con pocos hilos cada
una. `--procesos` fija cuántas corridas simultáneas y `--hilos` los hilos de
BLAS de cada una; en BitWit, 16 x 4 usa 64 de los 128 hilos.

ESCALA DE ENERGÍA. Se reporta en la de Joaquín,
    E_J = 2 E_nuestra + (2/3) (sum a)^2  =  < sum_s (S_s - mu)^2 >,
porque es la honesta para este problema: E_J es la desviación cuadrática media
de las sumas de las cajas, y vale exactamente 0 en una partición perfecta.
El umbral --epsilon se interpreta también en esa escala, igual que en su código.

    python cluster/benchmark_mwnp.py --n 5 6 7 --l 1 2 --ids 0-9
    python cluster/benchmark_mwnp.py --n 10 --l 1 --ids 0-9 --procesos 16 --hilos 4
"""

import os
import sys

# Los hilos de BLAS se fijan ANTES de importar numpy: después ya no cambian.
def _hilos_desde_argv(defecto=1):
    for i, arg in enumerate(sys.argv):
        if arg == "--hilos" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return str(defecto)

_H = _hilos_desde_argv()
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = _H

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import platform
import socket
import time
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np

INSTANCIAS = PROJECT_ROOT / "datos" / "mwnp_instancias.json"
CARPETA = PROJECT_ROOT / "resultados" / "mwnp"


def commit_del_codigo():
    """
    Hash del commit que se está corriendo, para que cada resultado se pueda
    rastrear al código exacto que lo produjo. En BitWit el árbol se exporta con
    `git archive`, sin .git, así que el hash viaja en un archivo aparte.
    """
    marca = PROJECT_ROOT / "VERSION_COMMIT"
    if marca.exists():
        return marca.read_text().strip()
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True).stdout.strip() or "desconocido"
    except Exception:
        return "desconocido"


def a_escala_j(E, a):
    """Energía de nuestra escala a la de Joaquín."""
    return 2.0 * float(E) + (2.0 / 3.0) * float(np.sum(a)) ** 2


def ruta_corrida(n, l, ident, carpeta=CARPETA):
    return Path(carpeta) / f"n{n}_l{l}_i{ident:02d}.json"


def conteo_nativo(etiquetas):
    """Compuertas nativas acumuladas (Algoritmo 1 de Ringbauer), paso a paso."""
    from funciones.utilidades_bp import conteo_compuertas
    R, MS, no_prod = [0], [0], 0
    for lab in etiquetas:
        c = conteo_compuertas(lab, base="angular")
        R.append(R[-1] + c["r_dos_niveles"])
        MS.append(MS[-1] + c["ms"])
        no_prod += 0 if c["es_producto"] else 1
    return {"r_dos_niveles": R, "ms": MS,
            "total": [r + m for r, m in zip(R, MS)],
            "generadores_que_no_factorizan": no_prod}


def correr(tarea):
    """Una corrida. Devuelve un resumen corto; el detalle queda en disco."""
    from funciones.utilidades_mwnp import adapt_mwnp, preparar_pool

    inst, l, eps_j, max_it, carpeta = tarea
    n, ident, a = inst["n"], inst["id"], inst["a"]
    ruta = ruta_corrida(n, l, ident, carpeta)
    if ruta.exists():
        return {"n": n, "l": l, "id": ident, "estado": "ya existía"}

    t0 = time.time()
    try:
        pool = preparar_pool(n, l)
        # El umbral viene en la escala de Joaquín, donde los gradientes valen
        # el doble que en la nuestra.
        r = adapt_mwnp(a, l=l, epsilon=eps_j / 2.0, max_iteration=max_it,
                       pool=pool, mostrar=False)
    except Exception as e:                       # que una falla no tumbe el lote
        return {"n": n, "l": l, "id": ident, "estado": f"ERROR: {e!r}"}

    E_j = [a_escala_j(E, a) for E in r["energy_trace"]]
    E0_j = a_escala_j(r["ground_energy"], a)
    nativo = conteo_nativo(r["ansatz_op_labels"])

    salida = {
        "instancia": inst,
        "config": {"l": l, "epsilon_escala_j": eps_j, "max_iteration": max_it,
                   "optimizador": "BFGS, gtol=1e-10, jac analítico (adjunto)",
                   "inicializacion": "warm start: (theta*_{k-1}, 0)"},
        "resultado": {
            "num_parametros": r["num_ansatz_ops"],
            "stop_reason": r["stop_reason"],
            "p_exito": r["prob_subespacio_optimo"],
            "p_azar": inst["p_azar"],
            "mejora_sobre_azar": r["prob_subespacio_optimo"] / inst["p_azar"],
            "E0_j": E0_j,
            "E_final_j": E_j[-1],
            "error_abs_j": E_j[-1] - E0_j,
            "encontro_la_particion": (r["top_particiones"][0]["desbalance"] == 0),
            "cadena_mas_probable": r["top_particiones"][0]["trits"],
            "particion_correcta": inst["particion_optima"],
            "top_particiones": r["top_particiones"],
            "pool_size": r["pool_size"],
            "mediciones_de_gradiente": r["pool_size"] * len(r["grad_norm_trace"]),
            "compuertas_nativas": nativo,
        },
        "trazas": {
            "energia_j": E_j,
            "norma_grad_j": [2.0 * g for g in r["grad_norm_trace"]],
            "p_exito": r["p_optimo_trace"],
            "desbalance_top": r["desbalance_trace"],
            "operadores": r["ansatz_op_labels"],
            "indices_pool": r["ansatz_op_indices"],
            "seleccion": r["seleccion_trace"],
            "parametros": r["params_trace"],
            "bfgs": r["bfgs_trace"],
            "tiempo_iter_s": r["tiempo_iter_trace"],
        },
        "ejecucion": {
            "runtime_s": time.time() - t0,
            "commit": commit_del_codigo(),
            "host": socket.gethostname(),
            "hilos_blas": int(os.environ.get("OMP_NUM_THREADS", "1")),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }

    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=1)
    os.replace(tmp, ruta)

    res = salida["resultado"]
    return {"n": n, "l": l, "id": ident, "estado": "ok",
            "k": res["num_parametros"], "p": res["p_exito"],
            "acierto": res["encontro_la_particion"],
            "t": salida["ejecucion"]["runtime_s"]}


def rango_ids(texto):
    """'0-9' -> [0..9];  '0,3,5' -> [0,3,5]."""
    ids = []
    for trozo in texto.split(","):
        if "-" in trozo:
            a, b = trozo.split("-")
            ids += list(range(int(a), int(b) + 1))
        else:
            ids.append(int(trozo))
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10])
    p.add_argument("--l", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    p.add_argument("--ids", type=str, default="0-9")
    p.add_argument("--epsilon", type=float, default=1e-3,
                   help="umbral de la norma del gradiente, en la escala de Joaquín")
    p.add_argument("--max_iteration", type=int, default=150)
    p.add_argument("--procesos", type=int, default=1)
    p.add_argument("--hilos", type=int, default=1)
    p.add_argument("--carpeta", type=str, default=str(CARPETA))
    args = p.parse_args()

    from funciones.utilidades_mwnp import etiquetas_pool

    todas = json.load(open(INSTANCIAS, encoding="utf-8"))["instancias"]
    ids = set(rango_ids(args.ids))
    elegidas = [i for i in todas if i["n"] in args.n and i["id"] in ids]

    # Los pools se construyen UNA vez acá, antes de repartir: si no, varios
    # procesos harían la misma expansión simbólica a la vez.
    for n in sorted(set(args.n)):
        for l in args.l:
            t = time.time()
            etiquetas_pool(n, l)
            print(f"pool n={n:2d} l={l}: listo en {time.time()-t:6.1f} s", flush=True)

    # Lo más caro primero, para que ningún proceso quede con la cola larga.
    tareas = [(inst, l, args.epsilon, args.max_iteration, args.carpeta)
              for inst in elegidas for l in args.l]
    tareas.sort(key=lambda t: (t[0]["n"], t[1]), reverse=True)

    print(f"\n{len(tareas)} corridas  |  {args.procesos} procesos x {args.hilos} hilos"
          f"  |  epsilon = {args.epsilon:g} (escala J)\n", flush=True)

    t0 = time.time()
    hechas = 0
    with Pool(args.procesos) as pool:
        for r in pool.imap_unordered(correr, tareas):
            hechas += 1
            if r["estado"] == "ok":
                print(f"[{hechas:3d}/{len(tareas)}] n={r['n']:2d} l={r['l']} #{r['id']:02d}"
                      f"  k={r['k']:3d}  p_exito={r['p']:.4f}"
                      f"  {'ACIERTA' if r['acierto'] else 'falla  '}  {r['t']:7.1f} s",
                      flush=True)
            else:
                print(f"[{hechas:3d}/{len(tareas)}] n={r['n']:2d} l={r['l']} "
                      f"#{r['id']:02d}  {r['estado']}", flush=True)

    print(f"\nlote terminado en {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
