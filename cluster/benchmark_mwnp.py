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
import multiprocessing

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


SUFIJOS = {"warm": "", "cold": "_cold", "fija0": "_fija0"}


def ruta_corrida(n, l, ident, carpeta=CARPETA, estrategia="warm"):
    # warm no lleva sufijo: son los archivos de la fase 1, que ya existen.
    return Path(carpeta) / f"n{n}_l{l}_i{ident:02d}{SUFIJOS[estrategia]}.json"


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


# Pools ya preparados en el proceso padre, antes de crear los procesos hijos.
# Con fork, los hijos los heredan por copia en escritura: los bloques de cada
# operador (hasta 81x81) no se escriben nunca, así que todos los procesos leen
# la misma memoria. A n = 12 con l = 2 el pool pesa ~2 GB; sin compartirlo,
# 50 procesos necesitarían 100 GB sólo para eso.
POOLS = {}


def correr(tarea):
    """Una corrida. Devuelve un resumen corto; el detalle queda en disco."""
    from funciones.utilidades_mwnp import adapt_mwnp, preparar_pool

    inst, l, eps_j, max_it, carpeta, estrategia = tarea
    n, ident, a = inst["n"], inst["id"], inst["a"]
    ruta = ruta_corrida(n, l, ident, carpeta, estrategia)
    if ruta.exists():
        return {"n": n, "l": l, "id": ident, "estado": "ya existía"}
    if estrategia == "fija0":
        return correr_fija0(inst, l, carpeta)

    t0 = time.time()
    try:
        pool = POOLS[(n, l)] if (n, l) in POOLS else preparar_pool(n, l)
        # El umbral viene en la escala de Joaquín, donde los gradientes valen
        # el doble que en la nuestra.
        r = adapt_mwnp(a, l=l, epsilon=eps_j / 2.0, max_iteration=max_it,
                       pool=pool, mostrar=False, inicializacion=estrategia)
    except Exception as e:                       # que una falla no tumbe el lote
        return {"n": n, "l": l, "id": ident, "estado": f"ERROR: {e!r}"}

    E_j = [a_escala_j(E, a) for E in r["energy_trace"]]
    E0_j = a_escala_j(r["ground_energy"], a)
    nativo = conteo_nativo(r["ansatz_op_labels"])

    salida = {
        "instancia": inst,
        "config": {"l": l, "epsilon_escala_j": eps_j, "max_iteration": max_it,
                   "optimizador": "BFGS, gtol=1e-10, jac analítico (adjunto)",
                   "estrategia": estrategia,
                   "inicializacion": ("warm start: (theta*_{k-1}, 0)" if estrategia == "warm"
                                      else "cold: theta = 0 en cada paso de ADAPT")},
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


def correr_fija0(inst, l, carpeta):
    """
    Variante (i): la MISMA secuencia de operadores que eligió el warm start,
    reoptimizada desde theta = 0 para cada tamaño k del circuito.

    Necesita la corrida warm de la misma instancia, de la que lee los
    operadores. Aísla el efecto de la inicialización: si con ese mismo circuito
    tampoco se llega a la solución, el problema no era el punto de partida de
    BFGS sino qué operadores se eligieron.
    """
    from funciones.utilidades_mwnp import reoptimizar_secuencia

    n, ident, a = inst["n"], inst["id"], inst["a"]
    origen = ruta_corrida(n, l, ident, carpeta, "warm")
    if not origen.exists():
        return {"n": n, "l": l, "id": ident, "estado": "falta la corrida warm"}
    warm = json.load(open(origen, encoding="utf-8"))
    etiquetas = warm["trazas"]["operadores"]

    t0 = time.time()
    try:
        r = reoptimizar_secuencia(a, etiquetas, cada_k=True)
    except Exception as e:
        return {"n": n, "l": l, "id": ident, "estado": f"ERROR: {e!r}"}

    tr = r["traza"]
    # Se antepone k = 0, la superposición uniforme, para que el índice de la
    # traza sea el tamaño del circuito igual que en las corridas warm y cold.
    E_j = [a_escala_j(E, a) for E in tr["energia"]]
    E0_j = a_escala_j(r["ground_energy"], a)
    top = r["top_particiones"]
    salida = {
        "instancia": inst,
        "config": {"l": l, "estrategia": "fija0",
                   "optimizador": "BFGS, gtol=1e-10, jac analítico (adjunto)",
                   "inicializacion": "theta = 0, secuencia fija de la corrida warm",
                   "origen": origen.name},
        "resultado": {
            "num_parametros": len(etiquetas),
            "p_exito": tr["p_exito"][-1],
            "p_azar": inst["p_azar"],
            "mejora_sobre_azar": tr["p_exito"][-1] / inst["p_azar"],
            "E0_j": E0_j, "E_final_j": E_j[-1], "error_abs_j": E_j[-1] - E0_j,
            "encontro_la_particion": top[0]["desbalance"] == 0,
            "cadena_mas_probable": top[0]["trits"],
            "particion_correcta": inst["particion_optima"],
            "top_particiones": top,
            "pool_size": warm["resultado"]["pool_size"],
            "compuertas_nativas": warm["resultado"]["compuertas_nativas"],
        },
        "trazas": {
            "energia_j": [warm["trazas"]["energia_j"][0]] + E_j,
            "p_exito": [warm["trazas"]["p_exito"][0]] + tr["p_exito"],
            "desbalance_top": [warm["trazas"]["desbalance_top"][0]] + tr["desbalance_top"],
            "operadores": etiquetas,
            "parametros": tr["parametros"],
            "bfgs": tr["bfgs"],
        },
        "ejecucion": {
            "runtime_s": time.time() - t0,
            "commit": commit_del_codigo(),
            "host": socket.gethostname(),
            "hilos_blas": int(os.environ.get("OMP_NUM_THREADS", "1")),
            "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    ruta = ruta_corrida(n, l, ident, carpeta, "fija0")
    tmp = ruta.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=1)
    os.replace(tmp, ruta)
    return {"n": n, "l": l, "id": ident, "estado": "ok",
            "k": len(etiquetas), "p": tr["p_exito"][-1],
            "acierto": salida["resultado"]["encontro_la_particion"],
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
    p.add_argument("--instancias", type=str, default=str(INSTANCIAS),
                   help="archivo de instancias; por defecto el conjunto fijo del benchmark")
    p.add_argument("--estrategia", type=str, default="warm",
                   choices=["warm", "cold", "fija0"],
                   help="warm: ADAPT estándar | cold (ii): ADAPT desde theta=0 en cada paso"
                        " | fija0 (i): la secuencia warm reoptimizada desde theta=0")
    args = p.parse_args()

    from funciones.utilidades_mwnp import etiquetas_pool, preparar_pool

    todas = json.load(open(args.instancias, encoding="utf-8"))["instancias"]
    ids = set(rango_ids(args.ids))
    elegidas = [i for i in todas if i["n"] in args.n and i["id"] in ids]

    # Los pools se construyen UNA vez acá, antes de repartir: si no, varios
    # procesos harían la misma expansión simbólica a la vez.
    for n in sorted(set(args.n)):
        for l in args.l:
            t = time.time()
            etiquetas_pool(n, l)
            if args.estrategia != "fija0":
                POOLS[(n, l)] = preparar_pool(n, l)
            print(f"pool n={n:2d} l={l}: listo en {time.time()-t:6.1f} s", flush=True)

    # Lo más caro primero, para que ningún proceso quede con la cola larga.
    tareas = [(inst, l, args.epsilon, args.max_iteration, args.carpeta, args.estrategia)
              for inst in elegidas for l in args.l]
    # El orden 0 (ascendente) va primero: así la tanda con un orden por
    # instancia se completa antes que las repeticiones con otros órdenes.
    tareas.sort(key=lambda t: (t[0].get("orden", 0) != 0, -t[0]["n"], -t[1]))

    print(f"\nestrategia: {args.estrategia}")
    print(f"{len(tareas)} corridas  |  {args.procesos} procesos x {args.hilos} hilos"
          f"  |  epsilon = {args.epsilon:g} (escala J)\n", flush=True)

    t0 = time.time()
    hechas = 0
    # fork explícito: los hijos heredan POOLS sin copiarlo (ver arriba).
    with multiprocessing.get_context("fork").Pool(args.procesos) as pool:
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
