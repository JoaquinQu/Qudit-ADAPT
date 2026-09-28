"""
Benchmark de QAOA para qutrits sobre multiway number partitioning, en paralelo.

Protocolo (el del QAOA de Max 3-Cut del paper, adaptado a la escala de MWNP):
  * p capas (por defecto 40, es decir 80 parámetros);
  * `--reinicios` puntos iniciales aleatorios por instancia, con todos los
    parámetros uniformes en [-pi, pi] y semilla determinista por
    (semilla, n, instancia, reinicio);
  * L-BFGS-B con cotas [-pi, pi], gradiente analítico, hasta `--maxiter`
    iteraciones;
  * H_C normalizado por su desviación estándar sobre la base computacional
    (ver `funciones/qaoa_mwnp.py`); las energías se guardan en la escala de
    J. Molina.

QAOA no depende del orden de los números: permutar qué número va en qué qutrit
permuta el estado final, y p_éxito no cambia. Basta una corrida por instancia.

Con `--interp` se agrega, por instancia, la cadena INTERP (Zhou et al. 2020):
barrido en grilla para p = 1 y, de p a p + 1, interpolación de los óptimos
como punto inicial. Da la curva completa de p_éxito contra p con un solo
optimizador por profundidad.

Un archivo por tarea en `--carpeta`:
    n{n}_i{id}_r{reinicio}.json   y   n{n}_i{id}_interp.json
Lo que ya existe se salta, así que el lote se puede interrumpir y relanzar.

    python cluster/qaoa_benchmark_mwnp.py --n 5 6 --ids 0-19 --procesos 8
    python cluster/qaoa_benchmark_mwnp.py --n 5 6 7 8 9 10 11 12 --interp --procesos 16
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import multiprocessing
import socket
import time
from datetime import datetime, timezone

import numpy as np

from funciones.qaoa_mwnp import evolucionar, interp, optimizar, p_exito
from funciones.utilidades_mwnp import hamiltoniano_joaquin, leer_estado

INSTANCIAS = PROJECT_ROOT / "datos" / "mwnp_instancias_5a12.json"
CARPETA = PROJECT_ROOT / "resultados" / "qaoa_p40"
COTA = np.pi


def commit_del_codigo():
    marca = PROJECT_ROOT / "VERSION_COMMIT"
    if marca.exists():
        return marca.read_text().strip()
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True).stdout.strip() or "desconocido"
    except Exception:
        return "desconocido"


def rango_ids(texto):
    ids = []
    for trozo in texto.split(","):
        if "-" in trozo:
            a, b = trozo.split("-")
            ids += list(range(int(a), int(b) + 1))
        else:
            ids.append(int(trozo))
    return ids


def ruta_tarea(carpeta, n, ident, reinicio):
    sufijo = "interp" if reinicio == "interp" else f"r{reinicio:02d}"
    return Path(carpeta) / f"n{n}_i{ident:02d}_{sufijo}.json"


def escribir(ruta, datos):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=1)
    os.replace(tmp, ruta)


def resumen_estado(x, hn, h, p, mezclador, a):
    psi = evolucionar(x, hn, p, mezclador)
    top = leer_estado(psi, a, cuantos=5)
    return {"p_exito": p_exito(psi, h), "E_j": float(np.real(np.vdot(psi, h * psi))),
            "cadena_mas_probable": top[0]["trits"],
            "encontro_la_particion": top[0]["desbalance"] == 0,
            "top_particiones": top}


def ejecucion(t0):
    return {"runtime_s": time.time() - t0, "commit": commit_del_codigo(),
            "host": socket.gethostname(), "hilos_blas": 1, "numpy": np.__version__,
            "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def correr(tarea):
    inst, reinicio, cfg = tarea
    n, ident, a = inst["n"], inst["id"], inst["a"]
    ruta = ruta_tarea(cfg["carpeta"], n, ident, reinicio)
    if ruta.exists():
        return {"n": n, "id": ident, "r": reinicio, "estado": "ya existía"}
    t0 = time.time()
    p, mezclador = cfg["p"], cfg["mezclador"]
    h = hamiltoniano_joaquin(a)
    sigma = float(h.std())
    hn = h / sigma
    config = {"p": p, "mezclador": mezclador, "normalizacion": "H_C / std(H_C)",
              "sigma": sigma, "cota": COTA, "maxiter": cfg["maxiter"], "gtol": cfg["gtol"],
              "ftol": 1e-12, "optimizador": "L-BFGS-B, jac analítico (adjunto)"}
    try:
        if reinicio == "interp":
            salida = correr_interp(inst, hn, h, sigma, cfg, config)
        else:
            rng = np.random.default_rng([cfg["semilla"], n, ident, reinicio])
            x0 = rng.uniform(-COTA, COTA, 2 * p)
            r = optimizar(hn, p, x0, mezclador, cota=COTA, maxiter=cfg["maxiter"], gtol=cfg["gtol"])
            salida = {
                "instancia": inst, "reinicio": reinicio,
                "config": {**config, "inicializacion": "uniforme en [-pi, pi]",
                           "semilla": [cfg["semilla"], n, ident, reinicio]},
                "resultado": {**resumen_estado(r["x"], hn, h, p, mezclador, a),
                              "E_norm": r["E"], "E0_j": float(h.min()),
                              "p_azar": inst["p_azar"], "particion_correcta": inst["particion_optima"]},
                "optimizador": {k: r[k] for k in ("nit", "nfev", "exito", "mensaje", "t_s")},
                "parametros": {"x0": x0.tolist(), "x": r["x"].tolist()},
            }
    except Exception as e:
        return {"n": n, "id": ident, "r": reinicio, "estado": f"ERROR: {e!r}"}
    salida["ejecucion"] = ejecucion(t0)
    escribir(ruta, salida)
    res = salida["resultado"] if reinicio != "interp" else salida["curva"][-1]
    return {"n": n, "id": ident, "r": reinicio, "estado": "ok", "p": res["p_exito"],
            "t": salida["ejecucion"]["runtime_s"]}


def correr_interp(inst, hn, h, sigma, cfg, config):
    """Cadena INTERP de p = 1 a p = P, con barrido en grilla para p = 1."""
    p_max, mezclador, a = cfg["p"], cfg["mezclador"], inst["a"]
    from funciones.qaoa_mwnp import energia_y_grad
    G, B = np.meshgrid(np.linspace(0.02, 3.0, 40), np.linspace(-COTA, COTA, 41))
    E = [energia_y_grad([g, b], hn, 1, mezclador)[0] for g, b in zip(G.ravel(), B.ravel())]
    x = np.array([G.ravel()[int(np.argmin(E))], B.ravel()[int(np.argmin(E))]])
    curva = []
    for p in range(1, p_max + 1):
        if p > 1:
            x = interp(x, p - 1)
        r = optimizar(hn, p, x, mezclador, cota=COTA, maxiter=cfg["maxiter"], gtol=cfg["gtol"])
        x = r["x"]
        est = resumen_estado(x, hn, h, p, mezclador, a)
        curva.append({"p": p, "parametros": 2 * p, "p_exito": est["p_exito"], "E_j": est["E_j"],
                      "E_norm": r["E"], "encontro_la_particion": est["encontro_la_particion"],
                      "nit": r["nit"], "nfev": r["nfev"], "exito": r["exito"], "t_s": r["t_s"],
                      "x": x.tolist()})
    return {"instancia": inst, "reinicio": "interp",
            "config": {**config, "p": p_max, "inicializacion": "grilla en p = 1 + INTERP"},
            "curva": curva}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=list(range(5, 13)))
    p.add_argument("--ids", type=str, default="0-19")
    p.add_argument("--p", type=int, default=40)
    p.add_argument("--reinicios", type=int, default=10)
    p.add_argument("--interp", action="store_true", help="agrega la cadena INTERP por instancia")
    p.add_argument("--solo_interp", action="store_true", help="sólo la cadena INTERP")
    p.add_argument("--mezclador", type=str, default="jx", choices=["jx", "x"])
    p.add_argument("--maxiter", type=int, default=1000)
    p.add_argument("--gtol", type=float, default=1e-8)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--procesos", type=int, default=1)
    p.add_argument("--carpeta", type=str, default=str(CARPETA))
    p.add_argument("--instancias", type=str, default=str(INSTANCIAS))
    args = p.parse_args()

    todas = json.load(open(args.instancias, encoding="utf-8"))["instancias"]
    ids = set(rango_ids(args.ids))
    elegidas = [i for i in todas if i["n"] in args.n and i["id"] in ids]
    cfg = {"p": args.p, "mezclador": args.mezclador, "maxiter": args.maxiter, "gtol": args.gtol,
           "semilla": args.semilla, "carpeta": args.carpeta}

    reinicios = [] if args.solo_interp else list(range(args.reinicios))
    if args.interp or args.solo_interp:
        reinicios.append("interp")
    tareas = [(inst, r, cfg) for inst in elegidas for r in reinicios]
    tareas.sort(key=lambda t: -t[0]["n"])           # lo más caro primero

    print(f"{len(tareas)} tareas | p = {args.p} | mezclador {args.mezclador} | "
          f"{args.procesos} procesos | carpeta {args.carpeta}", flush=True)
    t0 = time.time()
    with multiprocessing.get_context("fork").Pool(args.procesos) as pool:
        for k, r in enumerate(pool.imap_unordered(correr, tareas), 1):
            extra = f"p_exito={r['p']:.4f}  {r['t']:8.1f} s" if r["estado"] == "ok" else r["estado"]
            print(f"[{k:4d}/{len(tareas)}] n={r['n']:2d} #{r['id']:02d} r={r['r']}  {extra}", flush=True)
    print(f"\nlote terminado en {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
