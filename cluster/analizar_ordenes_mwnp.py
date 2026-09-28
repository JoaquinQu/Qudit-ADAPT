"""
l = 1 contra l = 2 en n = 5..9, promediando sobre órdenes de los números.

Cada instancia se corrió bajo 5 órdenes (qué número va en qué qutrit). El
orden no cambia el problema pero sí la trayectoria de ADAPT, porque los
empates del criterio de gradiente se resuelven por índice. Por eso la unidad
estadística es la INSTANCIA, y su tasa de éxito es la fracción de órdenes en
que se resuelve; los intervalos se obtienen remuestreando instancias.

Una corrida "resuelve" si p_éxito >= 0.1 (umbral del informe).

    python cluster/analizar_ordenes_mwnp.py
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
from scipy.stats import binomtest

CARPETA = PROJECT_ROOT / "resultados" / "mwnp_ordenes_5a9"
FASE1 = PROJECT_ROOT / "resultados" / "mwnp"
UMBRAL = 0.1


def cargar(carpeta):
    """estrategia -> {(n, l, instancia_base, orden) -> resumen de la corrida}."""
    runs = defaultdict(dict)
    for ruta in sorted(Path(carpeta).glob("n*_l*_i*.json")):
        d = json.load(open(ruta, encoding="utf-8"))
        i, r, t = d["instancia"], d["resultado"], d["trazas"]
        est = d["config"].get("estrategia", "warm")
        runs[est][(i["n"], d["config"]["l"], i["instancia_base"], i["orden"])] = {
            "p": r["p_exito"],
            "p_traza": t["p_exito"],
            "k": r["num_parametros"],
            # (i) reoptimiza una secuencia fija: no tiene criterio de parada propio
            "conv": r.get("stop_reason") == "gradient_norm_below_epsilon",
            "nativas": r["compuertas_nativas"]["total"][-1],
            "ms": r["compuertas_nativas"]["ms"][-1],
            "t": d["ejecucion"]["runtime_s"],
            "p_azar": i["p_azar"],
            "E": r["E_final_j"],
        }
    return runs


def ic_bootstrap(tasas, reps=10000, seed=0):
    """IC del 95 % de la media, remuestreando instancias."""
    rng = np.random.default_rng(seed)
    tasas = np.asarray(tasas)
    medias = rng.choice(tasas, size=(reps, len(tasas))).mean(axis=1)
    return np.percentile(medias, [2.5, 97.5])


def resumen(runs):
    ns = sorted({k[0] for k in runs})
    salida = {"por_n_l": [], "pareado": [], "orden_ascendente": [], "reproducibilidad": None}

    for n in ns:
        for l in (1, 2):
            rs = {k: v for k, v in runs.items() if k[0] == n and k[1] == l}
            bases = sorted({k[2] for k in rs})
            ords = sorted({k[3] for k in rs})
            tasa = np.array([np.mean([rs[(n, l, b, o)]["p"] >= UMBRAL for o in ords]) for b in bases])
            lo, hi = ic_bootstrap(tasa)
            todos = list(rs.values())
            fallas = [v for v in todos if v["p"] < UMBRAL]
            salida["por_n_l"].append({
                "n": n, "l": l, "instancias": len(bases), "ordenes": len(ords), "corridas": len(todos),
                "tasa": float(tasa.mean()), "ic95": [float(lo), float(hi)],
                "siempre": int((tasa == 1).sum()), "nunca": int((tasa == 0).sum()),
                "depende": int(((tasa > 0) & (tasa < 1)).sum()),
                "p_mediana": float(np.median([v["p"] for v in todos])),
                "p_azar": todos[0]["p_azar"],
                "convergidas": int(sum(v["conv"] for v in todos)),
                "k_mediana": float(np.median([v["k"] for v in todos])),
                "nativas_mediana": float(np.median([v["nativas"] for v in todos])),
                "ms_mediana": float(np.median([v["ms"] for v in todos])),
                "t_mediana_s": float(np.median([v["t"] for v in todos])),
                "fallas": len(fallas),
                "perdidas": int(sum(max(v["p_traza"]) >= UMBRAL for v in fallas)),
                "tasa_por_instancia": {int(b): float(x) for b, x in zip(bases, tasa)},
            })

        # l = 1 contra l = 2 sobre la misma (instancia, orden) y por instancia.
        c = defaultdict(int)
        mejor = peor = igual = 0
        bases = sorted({k[2] for k in runs if k[0] == n})
        for b in bases:
            t1 = t2 = 0
            for o in sorted({k[3] for k in runs if k[0] == n and k[2] == b}):
                a1 = runs[(n, 1, b, o)]["p"] >= UMBRAL
                a2 = runs[(n, 2, b, o)]["p"] >= UMBRAL
                c["ambos" if a1 and a2 else "solo_l1" if a1 else "solo_l2" if a2 else "ninguno"] += 1
                t1 += a1
                t2 += a2
            mejor += t2 > t1
            peor += t2 < t1
            igual += t2 == t1
        p_signo = binomtest(mejor, mejor + peor).pvalue if mejor + peor else 1.0
        salida["pareado"].append({"n": n, **c, "instancias_l2_mejor": mejor,
                                  "instancias_l1_mejor": peor, "instancias_igual": igual,
                                  "p_signo": float(p_signo)})

        # ¿El orden ascendente (orden 0, el del benchmark original) es desfavorable?
        for l in (1, 2):
            asc = [runs[(n, l, b, 0)]["p"] >= UMBRAL for b in bases]
            otros = [np.mean([runs[(n, l, b, o)]["p"] >= UMBRAL for o in range(1, 5)]) for b in bases]
            salida["orden_ascendente"].append({"n": n, "l": l, "tasa_asc": float(np.mean(asc)),
                                               "tasa_otros": float(np.mean(otros))})

    # Reproducibilidad: el orden 0 de las instancias 0-9 es la corrida de la fase 1.
    iguales = total = 0
    for (n, l, b, o), v in runs.items():
        ruta = FASE1 / f"n{n}_l{l}_i{b:02d}.json"
        if o == 0 and ruta.exists():
            viejo = json.load(open(ruta, encoding="utf-8"))["resultado"]["p_exito"]
            total += 1
            iguales += abs(viejo - v["p"]) < 1e-6
    salida["reproducibilidad"] = {"iguales": iguales, "total": total}
    return salida


def contra_warm(runs):
    """
    Warm contra (ii) ADAPT desde theta = 0 ("cold") y contra (i) la secuencia
    warm reoptimizada desde theta = 0 ("fija0"), sobre las mismas
    (instancia, orden). Unidad estadística: la instancia; su tasa es la
    fracción de órdenes resueltos.
    """
    fuera = []
    for otra in ("cold", "fija0"):
        if otra not in runs:
            continue
        for l in (1, 2):
            tot_w = tot_o = 0
            for n in sorted({k[0] for k in runs[otra]}):
                claves = [k for k in runs[otra] if k[0] == n and k[1] == l and k in runs["warm"]]
                if not claves:
                    continue
                bases = sorted({k[2] for k in claves})
                tw, to = [], []
                for b in bases:
                    ks = [k for k in claves if k[2] == b]
                    tw.append(np.mean([runs["warm"][k]["p"] >= UMBRAL for k in ks]))
                    to.append(np.mean([runs[otra][k]["p"] >= UMBRAL for k in ks]))
                tw, to = np.array(tw), np.array(to)
                gw, go = int((tw > to).sum()), int((to > tw).sum())
                tot_w += gw
                tot_o += go
                # energía final: fracción de corridas en que la otra termina más abajo
                baja = np.mean([runs[otra][k]["E"] < runs["warm"][k]["E"] - 1e-6 for k in claves])
                fuera.append({"otra": otra, "l": l, "n": n, "corridas": len(claves),
                              "tasa_warm": float(tw.mean()), "tasa_otra": float(to.mean()),
                              "ic_warm": [float(x) for x in ic_bootstrap(tw)],
                              "ic_otra": [float(x) for x in ic_bootstrap(to)],
                              "gana_warm": gw, "gana_otra": go, "iguales": int((tw == to).sum()),
                              "p_signo": float(binomtest(gw, gw + go).pvalue) if gw + go else 1.0,
                              "frac_E_otra_menor": float(baja)})
            fuera.append({"otra": otra, "l": l, "n": "todos", "gana_warm": tot_w, "gana_otra": tot_o,
                          "p_signo": float(binomtest(tot_w, tot_w + tot_o).pvalue) if tot_w + tot_o else 1.0})
    return fuera


def mostrar_contra_warm(filas):
    nombres = {"cold": "(ii) ADAPT desde theta=0", "fija0": "(i) secuencia warm desde theta=0"}
    for otra in ("cold", "fija0"):
        for l in (1, 2):
            fs = [f for f in filas if f["otra"] == otra and f["l"] == l]
            if not fs:
                continue
            print(f"\nwarm contra {nombres[otra]}, l = {l}:")
            for f in fs:
                if f["n"] == "todos":
                    print(f"  todos: gana warm {f['gana_warm']}, gana {otra} {f['gana_otra']}"
                          f"  (signo p = {f['p_signo']:.1e})")
                else:
                    print(f"  n={f['n']}  tasa warm {f['tasa_warm']:.2f}  {otra} {f['tasa_otra']:.2f}"
                          f"   instancias: warm {f['gana_warm']:2d} / {otra} {f['gana_otra']:2d} / igual {f['iguales']:2d}"
                          f"   E_{otra} < E_warm en {f['frac_E_otra_menor']:.0%} de las corridas  [{f['corridas']}]")


def mostrar(s):
    print(f"resuelve = p_éxito >= {UMBRAL}; tasa = media sobre instancias de la fracción de órdenes resueltos\n")
    print(f"{'n':>2s} {'l':>2s} {'tasa':>6s} {'IC 95 %':>13s} {'siempre/dep/nunca':>18s}"
          f" {'p med':>8s} {'conv':>8s} {'k med':>6s} {'nativas':>8s} {'MS':>5s} {'t med':>8s} {'perdidas':>9s}")
    for f in s["por_n_l"]:
        print(f"{f['n']:2d} {f['l']:2d} {f['tasa']:6.2f} [{f['ic95'][0]:.2f}, {f['ic95'][1]:.2f}]"
              f" {f['siempre']:6d}/{f['depende']:3d}/{f['nunca']:3d}      {f['p_mediana']:8.1e}"
              f" {f['convergidas']:4d}/{f['corridas']:3d} {f['k_mediana']:6.0f} {f['nativas_mediana']:8.0f}"
              f" {f['ms_mediana']:5.0f} {f['t_mediana_s']:7.0f}s {f['perdidas']:4d}/{f['fallas']:3d}")

    print("\nl = 1 contra l = 2, misma (instancia, orden):")
    for f in s["pareado"]:
        print(f"  n={f['n']}  ambos {f['ambos']:3d}  solo l=2 {f['solo_l2']:3d}  solo l=1 {f['solo_l1']:3d}"
              f"  ninguno {f['ninguno']:3d}   |  por instancia: l=2 mejor {f['instancias_l2_mejor']:2d},"
              f" l=1 mejor {f['instancias_l1_mejor']:2d}, igual {f['instancias_igual']:2d}"
              f"  (signo p = {f['p_signo']:.1e})")

    print("\norden ascendente contra los otros 4:")
    for f in s["orden_ascendente"]:
        print(f"  n={f['n']} l={f['l']}  ascendente {f['tasa_asc']:.2f}   otros {f['tasa_otros']:.2f}")

    m = sum(f["instancias_l2_mejor"] for f in s["pareado"])
    q = sum(f["instancias_l1_mejor"] for f in s["pareado"])
    s["signo_total"] = {"l2_mejor": m, "l1_mejor": q, "p": float(binomtest(m, m + q).pvalue)}
    print(f"  todos los n: l=2 mejor en {m} instancias, l=1 mejor en {q}  (signo p = {s['signo_total']['p']:.1e})")

    r = s["reproducibilidad"]
    print(f"\nreproducibilidad contra la fase 1 (orden 0, instancias 0-9): {r['iguales']}/{r['total']} idénticas")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--carpeta", type=str, default=str(CARPETA))
    args = p.parse_args()
    runs = cargar(args.carpeta)
    s = resumen(runs["warm"])
    mostrar(s)
    s["contra_warm"] = contra_warm(runs)
    mostrar_contra_warm(s["contra_warm"])
    salida = PROJECT_ROOT / "resultados" / "json" / "ordenes_5a9.json"
    json.dump(s, open(salida, "w", encoding="utf-8"), indent=1)
    print(f"\nguardado en {salida.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
