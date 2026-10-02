"""
Qudit-ADAPT (l = 1, 2; hasta k = 80) contra QAOA (p = 40, 10 reinicios), n = 5..12.

Fuentes:
  * ADAPT n = 5..9: `resultados/mwnp_ordenes_5a9`, corridas con tope 150. ADAPT
    es determinista y el tope sólo corta la corrida, así que sus primeras 80
    iteraciones son exactamente la corrida con tope 80: se lee p_éxito en
    k = min(80, k_final) de la traza.
  * ADAPT n = 10..12: `resultados/mwnp_k80`, tope 80.
  * QAOA: `resultados/qaoa_p40`, un archivo por (instancia, reinicio) y la
    cadena INTERP por instancia.
Las dos tandas de ADAPT son del mismo motor y de BitWit con 1 hilo.

Para ADAPT, cada instancia se corrió bajo 5 órdenes de sus números; la unidad
estadística es la instancia (tasa = fracción de órdenes con p_éxito >= 0.1).
También se reporta sólo el orden 0 (ascendente): un conjunto de números, una
corrida. QAOA no depende del orden. De sus 10 reinicios se reporta el de menor
energía (el que elegiría un experimento, que no conoce la solución) y la
mediana sobre reinicios (el estadístico del paper).

Funciona con datos parciales: resume lo que haya.

    python cluster/analizar_k80_qaoa.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from collections import defaultdict

import numpy as np

RES = PROJECT_ROOT / "resultados"
UMBRAL = 0.1
K = 80


def cargar_adapt():
    """(n, l, base, orden) -> {p, k, E_j}, con p_éxito en k = min(80, k_final)."""
    filas = {}
    for carpeta, de_prefijo in ((RES / "mwnp_ordenes_5a9", True), (RES / "mwnp_k80", False)):
        for f in carpeta.glob("n*_l*_i*.json"):
            if "_cold" in f.name or "_fija0" in f.name:
                continue
            d = json.load(open(f, encoding="utf-8"))
            i, tr = d["instancia"], d["trazas"]
            if d["config"].get("estrategia", "warm") != "warm":
                continue
            k = min(K, len(tr["p_exito"]) - 1)
            cn = d["resultado"]["compuertas_nativas"]       # acumuladas: índice k = primeros k operadores
            filas[(i["n"], d["config"]["l"], i["instancia_base"], i["orden"])] = {
                "p": tr["p_exito"][k], "k": k, "E_j": tr["energia_j"][k], "E_ini_j": tr["energia_j"][0],
                "r_local": cn["r_dos_niveles"][k], "ms": cn["ms"][k], "total": cn["total"][k],
                "p_azar": i["p_azar"], "fuente": carpeta.name}
    return filas


def cargar_qaoa():
    reinicios, interp = defaultdict(dict), {}
    for f in (RES / "qaoa_p40").glob("n*_i*_*.json"):
        d = json.load(open(f, encoding="utf-8"))
        i = d["instancia"]
        clave = (i["n"], i["id"])
        if d["reinicio"] == "interp":
            interp[clave] = d["curva"]
        else:
            r = d["resultado"]
            reinicios[clave][d["reinicio"]] = {"p": r["p_exito"], "E_j": r["E_j"],
                                               "nit": d["optimizador"]["nit"],
                                               "exito": d["optimizador"]["exito"]}
    return reinicios, interp


def resumen(adapt, reinicios, interp):
    filas = []
    for n in sorted({k[0] for k in adapt} | {k[0] for k in reinicios}):
        fila = {"n": n, "p_azar": 6 / 3 ** n}
        for l in (1, 2):
            rs = {k: v for k, v in adapt.items() if k[0] == n and k[1] == l}
            if not rs:
                continue
            bases = sorted({k[2] for k in rs})
            por_inst = [[v["p"] for k, v in rs.items() if k[2] == b] for b in bases]
            orden0 = [rs[(n, l, b, 0)]["p"] for b in bases if (n, l, b, 0) in rs]
            fila[f"adapt_l{l}"] = {
                "instancias": len(bases), "corridas": len(rs),
                "tasa": float(np.mean([np.mean(np.array(ps) >= UMBRAL) for ps in por_inst])),
                "p_mediana": float(np.median([v["p"] for v in rs.values()])),
                "orden0_corridas": len(orden0),
                "orden0_resueltas": int(np.sum(np.array(orden0) >= UMBRAL)),
                "orden0_p_mediana": float(np.median(orden0)) if orden0 else None,
            }
        qs = {k: v for k, v in reinicios.items() if k[0] == n}
        if qs:
            mejor = [min(v.values(), key=lambda x: x["E_j"])["p"] for v in qs.values()]
            todos = [x["p"] for v in qs.values() for x in v.values()]
            fila["qaoa"] = {
                "instancias": len(qs), "reinicios": len(todos),
                "mejor_resueltas": int(np.sum(np.array(mejor) >= UMBRAL)),
                "mejor_p_mediana": float(np.median(mejor)),
                "reinicios_frac_resuelve": float(np.mean(np.array(todos) >= UMBRAL)),
                "reinicios_p_mediana": float(np.median(todos)),
                "convergidos": float(np.mean([x["exito"] for v in qs.values() for x in v.values()])),
            }
        it = {k: v for k, v in interp.items() if k[0] == n}
        if it:
            p40 = [c[-1]["p_exito"] for c in it.values()]
            fila["interp"] = {"instancias": len(it), "resueltas": int(np.sum(np.array(p40) >= UMBRAL)),
                              "p_mediana": float(np.median(p40))}
        filas.append(fila)
    return filas


def mostrar(filas):
    print(f"resuelve = p_éxito >= {UMBRAL}. ADAPT: tasa promediada sobre 5 órdenes, y resueltas con el orden 0.")
    print(f"QAOA p=40: 'mejor' = reinicio de menor energía de 10.\n")
    print(f"{'n':>2s} {'azar':>8s} | {'ADAPT l=1 tasa':>14s} {'orden0':>7s} {'p med':>8s} | "
          f"{'ADAPT l=2 tasa':>14s} {'orden0':>7s} {'p med':>8s} | {'QAOA mejor':>10s} {'p med':>8s} "
          f"{'reinic.':>8s} | {'INTERP':>7s}")
    for f in filas:
        def a(l):
            x = f.get(f"adapt_l{l}")
            if not x:
                return f"{'--':>14s} {'--':>7s} {'--':>8s}"
            return (f"{x['tasa']:7.2f} ({x['corridas']:3d}) {x['orden0_resueltas']:3d}/{x['orden0_corridas']:<3d}"
                    f" {x['p_mediana']:8.1e}")
        q = f.get("qaoa")
        qs = (f"{q['mejor_resueltas']:4d}/{q['instancias']:<5d} {q['mejor_p_mediana']:8.1e} {q['reinicios']:8d}"
              if q else f"{'--':>10s} {'--':>8s} {'--':>8s}")
        it = f.get("interp")
        its = f"{it['resueltas']:3d}/{it['instancias']:<3d}" if it else f"{'--':>7s}"
        print(f"{f['n']:2d} {f['p_azar']:8.1e} | {a(1)} | {a(2)} | {qs} | {its}")


def main():
    adapt = cargar_adapt()
    reinicios, interp = cargar_qaoa()
    filas = resumen(adapt, reinicios, interp)
    mostrar(filas)
    salida = RES / "json" / "k80_qaoa.json"
    json.dump({"umbral": UMBRAL, "k_max": K, "resumen": filas,
               "adapt": [{"n": k[0], "l": k[1], "base": k[2], "orden": k[3], **v} for k, v in adapt.items()],
               "qaoa": [{"n": k[0], "id": k[1], "reinicio": r, **x}
                        for k, v in reinicios.items() for r, x in v.items()],
               "interp": [{"n": k[0], "id": k[1], "curva": [{c: e[c] for c in ("p", "p_exito", "E_j")} for e in v]}
                          for k, v in interp.items()]},
              open(salida, "w", encoding="utf-8"), indent=1)
    print(f"\n-> {salida.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
