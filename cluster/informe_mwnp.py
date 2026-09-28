"""
Figuras, tablas y cifras del informe de multiway number partitioning.

Todo lo que aparece en `informe/informe_mwnp.tex` sale de acá, leído de los
resultados guardados: ninguna cifra del informe se transcribe a mano. Las
cifras sueltas del texto se escriben como macros en `informe/numeros_mwnp.tex`.

    python cluster/informe_mwnp.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import glob
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

RES = PROJECT_ROOT / "resultados"
INF = PROJECT_ROOT / "informe"
FIG = INF / "figuras"
UMBRAL = 0.1          # p_éxito a partir del cual una instancia se considera resuelta
PISO = 1e-12          # para graficar en escala log las probabilidades nulas

C_L1, C_L2 = "#1f5fa8", "#c2410c"
C_WARM, C_COLD, C_FIJA = "#1f5fa8", "#15803d", "#7c3aed"


def estilo():
    mpl.rcdefaults()
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{txfonts}",
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "legend.fontsize": 8.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.dpi": 300,
    })


def guardar(fig, nombre):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{nombre}.pdf", metadata={"CreationDate": None})
    plt.close(fig)


# --------------------------------------------------------------------------
# datos
# --------------------------------------------------------------------------

def cargar_corridas():
    por = defaultdict(dict)          # (n, id) -> {(l, estrategia): dict}
    todas = []
    for f in sorted(glob.glob(str(RES / "mwnp" / "n*_l*_i*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        e = d["config"].get("estrategia", "warm")
        por[(d["instancia"]["n"], d["instancia"]["id"])][(d["config"]["l"], e)] = d
        todas.append(d)
    return por, todas


def serie(por, l, e, n):
    return [por[(n, i)][(l, e)] for i in range(10) if (l, e) in por[(n, i)]]


# --------------------------------------------------------------------------
# figuras
# --------------------------------------------------------------------------

def fig_exito_vs_n(por):
    ns = list(range(5, 11))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    for l, c, m in ((1, C_L1, "o"), (2, C_L2, "s")):
        frac = [np.mean([d["resultado"]["p_exito"] >= UMBRAL for d in serie(por, l, "warm", n)])
                for n in ns]
        a.plot(ns, frac, marker=m, color=c, lw=1.6, ms=5, label=rf"$\ell={l}$")
    a.set_xlabel(r"n\'umero de qutrits $n$")
    a.set_ylabel(r"fracci\'on resuelta ($p_{\rm \acute{e}xito}\geq 0.1$)")
    a.set_ylim(-0.03, 1.03)
    a.set_xticks(ns)
    a.legend(frameon=False, loc="upper right")
    a.set_title(r"(a)", loc="left", fontsize=10)

    rng = np.random.default_rng(0)
    for l, c, m, dx in ((1, C_L1, "o", -0.12), (2, C_L2, "s", 0.12)):
        for n in ns:
            ps = np.array([max(d["resultado"]["p_exito"], PISO) for d in serie(por, l, "warm", n)])
            b.scatter(n + dx + rng.uniform(-0.05, 0.05, len(ps)), ps, s=12, color=c,
                      marker=m, alpha=0.65, lw=0,
                      label=rf"$\ell={l}$" if n == 5 else None)
    b.plot(ns, [6 / 3**n for n in ns], "k--", lw=1, label=r"azar, $6/3^n$")
    b.set_yscale("log")
    b.set_ylim(PISO / 3, 3)
    b.set_xticks(ns)
    b.set_xlabel(r"n\'umero de qutrits $n$")
    b.set_ylabel(r"$p_{\rm \acute{e}xito}$ final")
    b.legend(frameon=False, loc="lower left", ncol=3)
    b.set_title(r"(b)", loc="left", fontsize=10)
    fig.tight_layout(w_pad=2.5)
    guardar(fig, "exito_vs_n")


def fig_pareado(por):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    cmap = plt.get_cmap("viridis")
    for ax, (e, nombre, tit) in zip(axes, (("cold", r"(ii) ADAPT desde $\boldsymbol\theta=0$", "(a)"),
                                          ("fija0", r"(i) secuencia fija desde $\boldsymbol\theta=0$", "(b)"))):
        for n in range(5, 11):
            xs, ys = [], []
            for i in range(10):
                p = por[(n, i)]
                if (1, "warm") in p and (1, e) in p:
                    xs.append(max(p[(1, "warm")]["resultado"]["p_exito"], PISO))
                    ys.append(max(p[(1, e)]["resultado"]["p_exito"], PISO))
            ax.scatter(xs, ys, s=16, color=cmap((n - 5) / 5), alpha=0.8, lw=0, label=rf"$n={n}$")
        ax.plot([PISO / 3, 3], [PISO / 3, 3], color="0.6", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(PISO / 3, 3); ax.set_ylim(PISO / 3, 3)
        ax.set_xlabel(r"$p_{\rm \acute{e}xito}$, warm start")
        ax.set_ylabel(r"$p_{\rm \acute{e}xito}$, " + nombre, fontsize=9.5)
        ax.set_title(tit, loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    fig.tight_layout(w_pad=2.5)
    guardar(fig, "pareado_warm_cold")


def fig_trazas(por, n, ident, nombre):
    p = por[(n, ident)]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for (l, e), c, et in (((1, "warm"), C_WARM, "warm start"),
                          ((1, "fija0"), C_FIJA, r"(i) secuencia fija, $\boldsymbol\theta=0$"),
                          ((1, "cold"), C_COLD, r"(ii) ADAPT desde $\boldsymbol\theta=0$")):
        if (l, e) not in p:
            continue
        tr = np.maximum(np.array(p[(l, e)]["trazas"]["p_exito"]), PISO)
        ax.plot(range(len(tr)), tr, color=c, lw=1.5, marker=".", ms=4, label=et)
    ax.axhline(6 / 3**n, color="k", ls="--", lw=0.9, label=r"azar")
    ax.set_yscale("log")
    ax.set_ylim(PISO / 3, 3)
    ax.set_xlabel(r"n\'umero de par\'ametros $k$")
    ax.set_ylabel(r"$p_{\rm \acute{e}xito}$")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    guardar(fig, nombre)


def fig_varianza():
    d = json.load(open(RES / "json" / "varianza_mwnp.json", encoding="utf-8"))
    filas = d["filas"]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for k, c, m in ((5, "#94a3b8", "^"), (10, C_L1, "o"), (20, "#0f172a", "s")):
        sub = sorted([f for f in filas if f["k"] == k], key=lambda f: f["n"])
        ax.plot([f["n"] for f in sub], [f["var_total"] for f in sub], marker=m,
                color=c, lw=1.3, ms=4, label=rf"medida, $k={k}$")
    sub = sorted([f for f in filas if f["k"] == 10], key=lambda f: f["n"])
    v0 = sub[0]["var_total"]
    ns = np.array([f["n"] for f in sub])
    ax.plot(ns, v0 * 3.0 ** (-(ns - 6)), "--", color=C_L2, lw=1.3,
            label=r"2-dise\~no, $\propto 3^{-n}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"n\'umero de qutrits $n$")
    ax.set_ylabel(r"${\rm Var}\,[\partial_\theta E]$")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    guardar(fig, "varianza")


def fig_tiempos(por):
    ns = list(range(5, 11))
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for l, c, m in ((1, C_L1, "o"), (2, C_L2, "s")):
        t = [np.median([d["ejecucion"]["runtime_s"] for d in serie(por, l, "warm", n)]) for n in ns]
        ax.plot(ns, t, marker=m, color=c, lw=1.5, ms=5, label=rf"$\ell={l}$")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xlabel(r"n\'umero de qutrits $n$")
    ax.set_ylabel(r"tiempo por corrida [s]")
    ax.legend(frameon=False)
    fig.tight_layout()
    guardar(fig, "tiempos")


# --------------------------------------------------------------------------
# tablas y cifras
# --------------------------------------------------------------------------

def tabla_instancias():
    inst = json.load(open(PROJECT_ROOT / "datos" / "mwnp_instancias.json", encoding="utf-8"))["instancias"]
    lineas = []
    for n in range(5, 11):
        ej = [i for i in inst if i["n"] == n][0]
        cajas = r"\;|\;".join(", ".join(map(str, c)) for c in ej["cajas_optimas"])
        lineas.append(rf"{n} & $[1,{3*n}]$ & ${6/3**n:.1e}$ & {ej['a']} & {cajas} \\".replace("e-0", r"\times10^{-").replace("$ &", "}$ &", 1)
                      if False else
                      rf"{n} & $[1,{3*n}]$ & {_sci(6/3**n)} & \texttt{{{ej['a']}}} & {cajas} \\")
    return "\n".join(lineas)


def _sci(x, dig=1):
    if x == 0:
        return "$0$"
    m, e = f"{x:.{dig}e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def tabla_fase1(por):
    lineas = []
    for n in range(5, 11):
        for l in (1, 2):
            ds = serie(por, l, "warm", n)
            r = [d["resultado"] for d in ds]
            p = np.array([x["p_exito"] for x in r])
            res = int(np.sum(p >= UMBRAL))
            k = np.median([x["num_parametros"] for x in r])
            conv = sum(x["stop_reason"] == "gradient_norm_below_epsilon" for x in r)
            nat = np.median([x["compuertas_nativas"]["total"][-1] for x in r])
            ms = np.median([x["compuertas_nativas"]["ms"][-1] for x in r])
            t = np.median([d["ejecucion"]["runtime_s"] for d in ds])
            pref = rf"\multirow{{2}}{{*}}{{{n}}}" if l == 1 else ""
            lineas.append(rf"{pref} & {l} & {res}/10 & {_sci(float(np.median(p)))} & "
                          rf"{k:.0f} & {conv}/10 & {nat:.0f} & {ms:.0f} & {_tiempo(t)} \\")
        if n < 10:
            lineas.append(r"\midrule")
    return "\n".join(lineas)


def _tiempo(s):
    return f"{s:.1f} s" if s < 60 else (f"{s/60:.1f} min" if s < 3600 else f"{s/3600:.1f} h")


def conteo_pareado(por, a, b, n):
    ga = gb = em = 0
    for i in range(10):
        p = por[(n, i)]
        if a not in p or b not in p:
            continue
        x, y = p[a]["resultado"]["p_exito"], p[b]["resultado"]["p_exito"]
        if x > 2 * y and x > 1e-6:
            ga += 1
        elif y > 2 * x and y > 1e-6:
            gb += 1
        else:
            em += 1
    return ga, gb, em


def tabla_fase2(por):
    lineas = []
    for n in range(5, 11):
        res = {e: sum(d["resultado"]["p_exito"] >= UMBRAL for d in serie(por, 1, e, n))
               for e in ("warm", "cold", "fija0")}
        wc = conteo_pareado(por, (1, "warm"), (1, "cold"), n)
        wf = conteo_pareado(por, (1, "warm"), (1, "fija0"), n)
        lineas.append(rf"{n} & {res['warm']} & {res['cold']} & {res['fija0']} & "
                      rf"{wc[0]} / {wc[1]} / {wc[2]} & {wf[0]} / {wf[1]} / {wf[2]} \\")
    return "\n".join(lineas)


def cifras(por, todas):
    m = {}
    tot = lambda f: sum(f(n) for n in range(5, 11))
    # l = 1 contra l = 2
    solo2 = solo1 = 0
    for n in range(5, 11):
        for i in range(10):
            p = por[(n, i)]
            if (1, "warm") in p and (2, "warm") in p:
                a1 = p[(1, "warm")]["resultado"]["p_exito"] >= UMBRAL
                a2 = p[(2, "warm")]["resultado"]["p_exito"] >= UMBRAL
                solo2 += a2 and not a1
                solo1 += a1 and not a2
    m["SoloLDos"], m["SoloLUno"] = solo2, solo1
    wc = [conteo_pareado(por, (1, "warm"), (1, "cold"), n) for n in range(5, 11)]
    wf = [conteo_pareado(por, (1, "warm"), (1, "fija0"), n) for n in range(5, 11)]
    m["GanaWarmCold"], m["GanaCold"] = sum(x[0] for x in wc), sum(x[1] for x in wc)
    m["GanaWarmFija"], m["GanaFija"] = sum(x[0] for x in wf), sum(x[1] for x in wf)
    m["TotalCorridas"] = len(todas)

    # perdidas: fallas con p_max >= 10x azar a mitad de camino (l = 1, warm)
    perd = fall = 0
    for n in range(5, 11):
        for d in serie(por, 1, "warm", n):
            if d["resultado"]["p_exito"] < UMBRAL:
                fall += 1
                perd += max(d["trazas"]["p_exito"]) >= 10 * d["instancia"]["p_azar"]
    m["FallasLUno"], m["PerdidasLUno"] = fall, perd
    conv1 = sum(d["resultado"]["stop_reason"] == "gradient_norm_below_epsilon"
                for n in range(5, 11) for d in serie(por, 1, "warm", n))
    m["ConvergenLUno"] = conv1

    # validacion contra Joaquin
    s = json.load(open(RES / "json" / "sanidad_joaquin.json", encoding="utf-8"))
    m["SanidadMaxDif"] = _sci(s["max_dif_energia"])
    cj = json.load(open(RES / "json" / "comparacion_joaquin.json", encoding="utf-8"))
    ps = np.array([f["suyo"]["p_exito"] for f in cj]); pn = np.array([f["nuestro"]["p_exito"] for f in cj])
    m["CompEmpates"] = int(np.sum(np.abs(ps - pn) <= 0.05))
    m["CompGanamos"] = int(np.sum(pn > ps + 0.05))
    m["CompGanaEl"] = int(np.sum(ps > pn + 0.05))
    m["CompTotal"] = len(cj)

    # varianza
    v = json.load(open(RES / "json" / "varianza_mwnp.json", encoding="utf-8"))["filas"]
    sub = sorted([f for f in v if f["k"] == 10], key=lambda f: f["n"])
    ns = np.array([f["n"] for f in sub], float); lv = np.log([f["var_total"] for f in sub])
    m["AlfaVar"] = f"{-np.polyfit(ns, lv, 1)[0]:.2f}"
    m["RazonVarDiseno"] = f"{sub[-1]['var_total'] / (sub[0]['var_total'] * 3.0 ** (-(sub[-1]['n'] - 6))):.0f}"

    # n = 13
    r13 = json.load(open(RES / "json" / "mwnp_n13_largo.json", encoding="utf-8"))
    tr = r13["p_optimo_trace"]
    m["NTreceK"] = r13["num_ansatz_ops"]
    m["NTrecePmax"] = _sci(max(tr))
    m["NTreceKmax"] = int(np.argmax(tr))
    m["NTrecePfin"] = _sci(r13["prob_subespacio_optimo"])
    m["NTreceHoras"] = f"{r13['runtime_s']/3600:.1f}"
    return m


def main():
    estilo()
    por, todas = cargar_corridas()
    INF.mkdir(parents=True, exist_ok=True)

    fig_exito_vs_n(por)
    fig_pareado(por)
    fig_trazas(por, 5, 0, "trazas_n5_i00")
    fig_trazas(por, 6, 0, "trazas_n6_i00")
    fig_varianza()
    fig_tiempos(por)

    (INF / "tabla_instancias.tex").write_text(tabla_instancias(), encoding="utf-8")
    (INF / "tabla_fase1.tex").write_text(tabla_fase1(por), encoding="utf-8")
    (INF / "tabla_fase2.tex").write_text(tabla_fase2(por), encoding="utf-8")
    m = cifras(por, todas)
    (INF / "numeros_mwnp.tex").write_text(
        "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items()) + "\n", encoding="utf-8")
    for k, v in m.items():
        print(f"  {k:18s} {v}")
    print(f"\nfiguras en {FIG.relative_to(PROJECT_ROOT)}, tablas y cifras en {INF.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
