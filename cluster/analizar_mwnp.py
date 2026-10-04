"""
Análisis del benchmark de multiway partitioning: l = 1 contra l = 2 a lo largo de n.

Lee todos los `resultados/mwnp/n*_l*_i*.json` y resume, por (n, l):

  * tasa de acierto: fracción de instancias cuya cadena más probable ES la
    partición correcta, que es lo que entregaría el experimento;
  * p_éxito final, y su cociente contra el azar (6/3^n);
  * p_éxito MÁXIMO a lo largo de la corrida, y cuántas corridas que terminan
    fallando lo tuvieron alto a mitad de camino. Es la firma de que el warm
    start arrastró el estado fuera de la solución: la respuesta estuvo al
    alcance y se perdió;
  * parámetros, compuertas nativas y mediciones de gradiente;
  * tiempo, y qué fracción se va en el barrido del pool y cuál en BFGS, que
    es lo que dice qué subrutina conviene optimizar.

    python cluster/analizar_mwnp.py
    python cluster/analizar_mwnp.py --umbral_perdida 0.05
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from collections import defaultdict

import numpy as np

CARPETA = PROJECT_ROOT / "resultados" / "mwnp"


def estrategia(d):
    """warm si la corrida no la registra: son las de la fase 1."""
    return d["config"].get("estrategia", "warm")


def cargar(carpeta):
    corridas = []
    for ruta in sorted(Path(carpeta).glob("n*_l*_i*.json")):
        d = json.load(open(ruta, encoding="utf-8"))
        corridas.append(d)
    return corridas


def resumen(corridas, umbral_perdida):
    grupos = defaultdict(list)
    for d in corridas:
        grupos[(d["instancia"]["n"], d["config"]["l"], estrategia(d))].append(d)

    filas = []
    for (n, l, est), ds in sorted(grupos.items()):
        r = [d["resultado"] for d in ds]
        t = [d["trazas"] for d in ds]
        acierta = np.array([x["encontro_la_particion"] for x in r])
        p_fin = np.array([x["p_exito"] for x in r])
        p_max = np.array([max(tt["p_exito"]) for tt in t])
        k_pmax = np.array([int(np.argmax(tt["p_exito"])) for tt in t])
        k = np.array([x["num_parametros"] for x in r])
        conv = np.array([x.get("stop_reason", "gradient_norm_below_epsilon")
                         == "gradient_norm_below_epsilon" for x in r])

        # Corridas que FALLAN pero tuvieron la respuesta al alcance.
        perdidas = (~acierta) & (p_max > umbral_perdida)

        bfgs = [b for tt in t for b in tt["bfgs"]]
        t_barr = sum(b.get("t_barrido", 0.0) for b in bfgs)
        t_bfgs = sum(b.get("t_bfgs", 0.0) for b in bfgs)

        filas.append({
            "n": n, "l": l, "estrategia": est, "instancias": len(ds),
            "tasa_acierto": float(acierta.mean()),
            "p_exito_mediana": float(np.median(p_fin)),
            "p_exito_media": float(p_fin.mean()),
            "p_azar": ds[0]["instancia"]["p_azar"],
            "mejora_mediana": float(np.median(p_fin) / ds[0]["instancia"]["p_azar"]),
            "p_max_mediana": float(np.median(p_max)),
            "k_de_p_max_mediana": float(np.median(k_pmax)),
            "fallas": int((~acierta).sum()),
            "fallas_con_respuesta_perdida": int(perdidas.sum()),
            "k_mediana": float(np.median(k)),
            "k_max": int(k.max()),
            "convergidas": int(conv.sum()),
            "nativas_mediana": float(np.median([x["compuertas_nativas"]["total"][-1] for x in r])),
            "ms_mediana": float(np.median([x["compuertas_nativas"]["ms"][-1] for x in r])),
            "mediciones_mediana": float(np.median([x.get("mediciones_de_gradiente", 0) for x in r])),
            "pool": r[0]["pool_size"],
            "t_mediana_s": float(np.median([d["ejecucion"]["runtime_s"] for d in ds])),
            "t_total_s": float(sum(d["ejecucion"]["runtime_s"] for d in ds)),
            "frac_barrido": t_barr / (t_barr + t_bfgs) if (t_barr + t_bfgs) else 0.0,
        })
    return filas


def pareado(corridas):
    """Misma instancia con l = 1 y l = 2 (sólo warm): ¿cuál acierta?"""
    por = defaultdict(dict)
    for d in corridas:
        if estrategia(d) != "warm":
            continue
        por[(d["instancia"]["n"], d["instancia"]["id"])][d["config"]["l"]] = d["resultado"]
    tabla = defaultdict(lambda: {"ambos": 0, "solo_l1": 0, "solo_l2": 0, "ninguno": 0})
    for (n, _), par in por.items():
        if 1 not in par or 2 not in par:
            continue
        a1, a2 = par[1]["encontro_la_particion"], par[2]["encontro_la_particion"]
        clave = "ambos" if a1 and a2 else "solo_l1" if a1 else "solo_l2" if a2 else "ninguno"
        tabla[n][clave] += 1
    return dict(tabla)


def contra_warm(corridas, l):
    """
    Para cada instancia con corrida warm y otra estrategia, ¿quién acierta?
    Es la comparación que pone a prueba la conjetura: la misma instancia,
    resuelta partiendo del óptimo anterior o desde la superposición uniforme.
    """
    por = defaultdict(dict)
    for d in corridas:
        if d["config"]["l"] != l:
            continue
        por[(d["instancia"]["n"], d["instancia"]["id"])][estrategia(d)] = d["resultado"]
    fuera = {}
    for otra in ("cold", "fija0"):
        tabla = defaultdict(lambda: {"ambas": 0, "solo_warm": 0, "solo_otra": 0, "ninguna": 0})
        for (n, _), e in por.items():
            if "warm" not in e or otra not in e:
                continue
            w, o = e["warm"]["encontro_la_particion"], e[otra]["encontro_la_particion"]
            clave = "ambas" if w and o else "solo_warm" if w else "solo_otra" if o else "ninguna"
            tabla[n][clave] += 1
        if tabla:
            fuera[otra] = dict(tabla)
    return fuera


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--carpeta", type=str, default=str(CARPETA))
    p.add_argument("--umbral_perdida", type=float, default=0.05,
                   help="p_éxito a mitad de camino por encima del cual una falla cuenta como respuesta perdida")
    args = p.parse_args()

    corridas = cargar(args.carpeta)
    if not corridas:
        raise SystemExit(f"no hay corridas en {args.carpeta}")
    filas = resumen(corridas, args.umbral_perdida)

    print(f"{len(corridas)} corridas\n")
    print(f"{'n':>3s}{'l':>3s}{'estrat':>7s}{'acierto':>9s}{'p_éx med':>10s}{'x azar':>9s}"
          f"{'p_máx med':>11s}{'@k':>5s}{'fallas':>8s}{'perdidas':>10s}"
          f"{'k med':>7s}{'nativas':>9s}{'t med':>9s}{'%barrido':>10s}")
    for f in filas:
        print(f"{f['n']:3d}{f['l']:3d}{f['estrategia']:>7s}{f['tasa_acierto']:8.0%} {f['p_exito_mediana']:10.3f}"
              f"{f['mejora_mediana']:9.0f}{f['p_max_mediana']:11.3f}{f['k_de_p_max_mediana']:5.0f}"
              f"{f['fallas']:8d}{f['fallas_con_respuesta_perdida']:10d}{f['k_mediana']:7.0f}"
              f"{f['nativas_mediana']:9.0f}{f['t_mediana_s']:8.1f}s{f['frac_barrido']:9.0%}")

    print(f"\n'perdidas' = fallas cuya p_éxito superó {args.umbral_perdida:.0%} a mitad de camino")

    print("\nMisma instancia, l = 1 contra l = 2:")
    for n, t in sorted(pareado(corridas).items()):
        print(f"  n={n:2d}  ambos {t['ambos']:2d}   solo l=1 {t['solo_l1']:2d}"
              f"   solo l=2 {t['solo_l2']:2d}   ninguno {t['ninguno']:2d}")

    for l in (1, 2):
        for otra, tabla in contra_warm(corridas, l).items():
            nombre = {"cold": "(ii) ADAPT desde theta=0", "fija0": "(i) secuencia warm desde theta=0"}[otra]
            print(f"\nl = {l}: warm contra {nombre}")
            for n, x in sorted(tabla.items()):
                print(f"  n={n:2d}  ambas {x['ambas']:2d}   solo warm {x['solo_warm']:2d}"
                      f"   solo {otra} {x['solo_otra']:2d}   ninguna {x['ninguna']:2d}")

    salida = PROJECT_ROOT / "resultados" / "json" / "resumen_mwnp.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"filas": filas, "pareado": pareado(corridas),
               "umbral_perdida": args.umbral_perdida},
              open(salida, "w", encoding="utf-8"), indent=1)
    print(f"\nguardado en {salida.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
