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
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}\usepackage{txfonts}",
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
    b.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    b.text(-0.18, 1.02, r"(b)", transform=b.transAxes, fontsize=10)
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


def cargar_ordenes():
    por = defaultdict(dict)
    for f in glob.glob(str(RES / "mwnp_ordenes" / "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        i = d["instancia"]
        por[i["instancia_base"]][i["orden"]] = d["resultado"]["p_exito"]
    return por


def fig_ordenes(ordenes):
    bases = sorted(ordenes)
    M = np.array([[ordenes[b][r] for r in range(10)] for b in bases])
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(10))
    ax.set_xticklabels(["asc."] + [str(r) for r in range(1, 10)])
    ax.set_yticks(range(len(bases)))
    ax.set_yticklabels([str(b) for b in bases], fontsize=7)
    ax.set_xlabel(r"orden de los n\'umeros en los qutrits")
    ax.set_ylabel(r"instancia ($n=6$)")
    ax.axvline(0.5, color="k", lw=0.8)
    for s in ax.spines.values():
        s.set_visible(True)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(r"$p_{\rm \acute{e}xito}$")
    fig.tight_layout()
    guardar(fig, "ordenes")


def fig_rango():
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    etiquetas, x = [], 0
    for l, c in ((1, C_L1), (2, C_L2)):
        for nombre, pat in ((r"$[1,8]$", "mwnp_rango8/n6_l{l}_i??.json"),
                            (r"$[1,18]$", "mwnp/n6_l{l}_i??.json")):
            ps = [json.load(open(f))["resultado"]["p_exito"] for f in glob.glob(str(RES / pat.format(l=l)))]
            fr = np.mean(np.array(ps) >= UMBRAL)
            ax.bar(x, fr, color=c, alpha=0.55 if "8]" in nombre else 0.95, width=0.7)
            ax.text(x, fr + 0.03, f"{int(np.sum(np.array(ps) >= UMBRAL))}/{len(ps)}",
                    ha="center", fontsize=8)
            etiquetas.append(nombre + rf"\\$\ell={l}$")
            x += 1
        x += 0.5
    ax.set_xticks([0, 1, 2.5, 3.5])
    ax.set_xticklabels([r"$[1,8]$" "\n" r"$\ell=1$", r"$[1,18]$" "\n" r"$\ell=1$",
                        r"$[1,8]$" "\n" r"$\ell=2$", r"$[1,18]$" "\n" r"$\ell=2$"], fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(r"fracci\'on resuelta")
    fig.tight_layout()
    guardar(fig, "rango")



# --------------------------------------------------------------------------
# l = 1 contra l = 2 con 5 ordenes por instancia (n = 5..9)
# --------------------------------------------------------------------------

def cargar_ordenes_5a9(estrategia="warm"):
    """(n, l, instancia_base, orden) -> (p_éxito final, traza de p_éxito, dict resultado, tiempo)."""
    runs = {}
    for f in glob.glob(str(RES / "mwnp_ordenes_5a9" / "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        if d["config"].get("estrategia", "warm") != estrategia:
            continue
        i = d["instancia"]
        runs[(i["n"], d["config"]["l"], i["instancia_base"], i["orden"])] = (
            d["resultado"]["p_exito"], d["trazas"]["p_exito"], d["resultado"], d["ejecucion"]["runtime_s"])
    return runs


def fig_ordenes_5a9(res):
    """(a) tasa de éxito promediada sobre órdenes contra n; (b) l = 1 contra l = 2 por instancia."""
    filas = res["por_n_l"]
    ns = sorted({f["n"] for f in filas})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for l, c, m, dx in ((1, C_L1, "o", -0.06), (2, C_L2, "s", 0.06)):
        fl = [f for f in filas if f["l"] == l]
        y = np.array([f["tasa"] for f in fl])
        lo = y - np.array([f["ic95"][0] for f in fl])
        hi = np.array([f["ic95"][1] for f in fl]) - y
        a.errorbar(np.array(ns) + dx, y, yerr=[lo, hi], marker=m, color=c, lw=1.6, ms=5,
                   capsize=2.5, label=rf"$\ell={l}$")
    a.set_xlabel(r"n\'umero de qutrits $n$")
    a.set_ylabel(r"tasa de \'exito")
    a.set_ylim(-0.03, 1.03)
    a.set_xticks(ns)
    a.legend(frameon=False, loc="lower left")
    a.set_title(r"(a)", loc="left", fontsize=10)

    rng = np.random.default_rng(1)
    cmap = plt.get_cmap("viridis")
    for j, n in enumerate(ns):
        t1 = [x for f in filas if f["n"] == n and f["l"] == 1 for x in f["tasa_por_instancia"].values()]
        t2 = [x for f in filas if f["n"] == n and f["l"] == 2 for x in f["tasa_por_instancia"].values()]
        b.scatter(np.array(t1) + rng.uniform(-0.03, 0.03, len(t1)),
                  np.array(t2) + rng.uniform(-0.03, 0.03, len(t2)),
                  s=14, color=cmap(j / (len(ns) - 1) * 0.9), alpha=0.8, lw=0, label=rf"$n={n}$")
    b.plot([0, 1], [0, 1], color="0.6", lw=0.8)
    b.set_xlim(-0.06, 1.06)
    b.set_ylim(-0.06, 1.06)
    b.set_aspect("equal")
    b.set_xlabel(r"tasa por instancia, $\ell=1$")
    b.set_ylabel(r"tasa por instancia, $\ell=2$")
    b.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5), handletextpad=0.2)
    b.set_title(r"(b)", loc="left", fontsize=10)
    fig.tight_layout(w_pad=2.0)
    guardar(fig, "ordenes_5a9")


def fig_ordenes_iteracion(runs):
    """Fracción de corridas con p_éxito >= umbral en función de la iteración ADAPT."""
    ns = sorted({k[0] for k in runs})
    kmax = max(len(v[1]) for v in runs.values()) - 1
    cmap = plt.get_cmap("viridis")
    fig, ejes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, l in zip(ejes, (1, 2)):
        for j, n in enumerate(ns):
            trazas = [v[1] for k, v in runs.items() if k[0] == n and k[1] == l]
            M = np.array([t + [t[-1]] * (kmax + 1 - len(t)) for t in trazas]) >= UMBRAL
            ax.plot(np.arange(kmax + 1), M.mean(axis=0), color=cmap(j / (len(ns) - 1) * 0.9),
                    lw=1.5, label=rf"$n={n}$")
        ax.set_xlabel(r"iteraci\'on ADAPT $k$")
        ax.set_title(rf"$\ell={l}$", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
    ejes[0].set_ylabel(r"fracci\'on con $p_{\rm \acute{e}xito}\geq 0.1$")
    ejes[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(w_pad=1.0)
    guardar(fig, "ordenes_iteracion")


def tabla_ordenes(res):
    lineas = []
    filas = res["por_n_l"]
    ns = sorted({f["n"] for f in filas})
    for n in ns:
        for l in (1, 2):
            f = [x for x in filas if x["n"] == n and x["l"] == l][0]
            pref = rf"\multirow{{2}}{{*}}{{{n}}}" if l == 1 else ""
            lineas.append(rf"{pref} & {l} & {f['tasa']:.2f} & [{f['ic95'][0]:.2f}, {f['ic95'][1]:.2f}] & "
                          rf"{f['siempre']} / {f['depende']} / {f['nunca']} & {f['convergidas']}/{f['corridas']} & "
                          rf"{f['k_mediana']:.0f} & {f['nativas_mediana']:.0f} & {f['ms_mediana']:.0f} & "
                          rf"{_tiempo(f['t_mediana_s'])} \\")
        if n < ns[-1]:
            lineas.append(r"\midrule")
    return "\n".join(lineas)


def fig_estrategias(res):
    """Tasa de éxito de warm, (ii) y (i) contra n, para l = 1 y l = 2."""
    filas = [f for f in res.get("contra_warm", []) if f["n"] != "todos"]
    if not filas:
        return
    fig, ejes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, l in zip(ejes, (1, 2)):
        for otra, c, m, dx, nombre in (("cold", C_COLD, "^", 0.07, r"(ii) ADAPT desde $\bm\theta=0$"),
                                       ("fija0", C_FIJA, "D", 0.14, r"(i) secuencia fija desde $\bm\theta=0$")):
            fs = sorted([f for f in filas if f["l"] == l and f["otra"] == otra], key=lambda f: f["n"])
            if not fs:
                continue
            ns = np.array([f["n"] for f in fs])
            if otra == "cold":
                y = np.array([f["tasa_warm"] for f in fs])
                ic = np.array([f["ic_warm"] for f in fs])
                ax.errorbar(ns, y, yerr=[y - ic[:, 0], ic[:, 1] - y], marker="o", color=C_WARM,
                            lw=1.6, ms=4.5, capsize=2, label="warm start")
            y = np.array([f["tasa_otra"] for f in fs])
            ic = np.array([f["ic_otra"] for f in fs])
            ax.errorbar(ns + dx, y, yerr=[y - ic[:, 0], ic[:, 1] - y], marker=m, color=c,
                        lw=1.3, ms=4.5, capsize=2, label=nombre)
        ax.set_title(rf"$\ell={l}$", fontsize=10)
        ax.set_xlabel(r"n\'umero de qutrits $n$")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(range(5, 10))
    ejes[0].set_ylabel(r"tasa de \'exito")
    ejes[1].legend(frameon=False, loc="lower left")
    fig.tight_layout(w_pad=1.0)
    guardar(fig, "estrategias")


def tabla_estrategias(res):
    filas = res.get("contra_warm", [])
    lineas = []
    for l in (1, 2):
        ns = sorted({f["n"] for f in filas if f["l"] == l and f["n"] != "todos"})
        for j, n in enumerate(ns):
            c = [f for f in filas if f["l"] == l and f["n"] == n and f["otra"] == "cold"]
            fi = [f for f in filas if f["l"] == l and f["n"] == n and f["otra"] == "fija0"]
            pref = rf"\multirow{{{len(ns)}}}{{*}}{{{l}}}" if j == 0 else ""
            celda = lambda x, k: f"{x[0][k]:.2f}" if x else "--"
            par = lambda x: f"{x[0]['gana_warm']} / {x[0]['gana_otra']} / {x[0]['iguales']}" if x else "--"
            tw = c[0]["tasa_warm"] if c else (fi[0]["tasa_warm"] if fi else float("nan"))
            lineas.append(rf"{pref} & {n} & {tw:.2f} & {celda(c, 'tasa_otra')} & {celda(fi, 'tasa_otra')}"
                          rf" & {par(c)} & {par(fi)} \\")
        tc = [f for f in filas if f["l"] == l and f["n"] == "todos" and f["otra"] == "cold"]
        tf = [f for f in filas if f["l"] == l and f["n"] == "todos" and f["otra"] == "fija0"]
        tot = lambda x: (rf"{x[0]['gana_warm']} / {x[0]['gana_otra']} ($p={_sci(x[0]['p_signo']).strip('$')}$)"
                         if x else "--")
        lineas.append(rf"\cmidrule(lr){{2-7}} & todos & & & & {tot(tc)} & {tot(tf)} \\")
        if l == 1:
            lineas.append(r"\midrule")
    return "\n".join(lineas)


# --------------------------------------------------------------------------
# pool y rangos
# --------------------------------------------------------------------------

def n_pool(n, l):
    """Operadores distintos del pool de K_n: conteo por tamaño de soporte."""
    from math import comb
    if l == 1:
        return n + 4 * comb(n, 2)
    return 3 * n + 40 * comb(n, 2) + 78 * comb(n, 3) + 32 * comb(n, 4)


def tabla_pool():
    """Tamaños del pool; se verifican contra las etiquetas en caché que existan."""
    lineas = []
    for n in range(5, 11):
        n1, n2 = n_pool(n, 1), n_pool(n, 2)
        for l, esperado in ((1, n1), (2, n2)):
            ruta = RES / "cache_pools" / f"kn_{n}_l{l}.json"
            if ruta.exists():
                lab = json.load(open(ruta, encoding="utf-8"))
                assert len(set(lab)) == esperado, (n, l, len(set(lab)), esperado)
                if l == 2:
                    assert len(lab) == n1 + n2, (n, len(lab))
        lineas.append(rf"{n} & {n1} & {n2} & {n1 + n2} & {n2 / n1:.0f} \\")
    return "\n".join(lineas)


def tabla_rangos():
    r = json.load(open(RES / "json" / "rangos_instancias.json", encoding="utf-8"))
    cols = ["2n", "3n", "4n", "6n", "15"]
    lineas = []
    for n in range(5, 11):
        fs = {f["rango"]: f for f in r["filas"] if f["n"] == n}
        celdas = []
        for c in cols:
            v = f"{100 * fs[c]['frac_unica']:.1f}"
            celdas.append(rf"\textbf{{{v}}}" if c == "3n" else v)
        lineas.append(rf"{n} & " + " & ".join(celdas) + r" \\")
    return "\n".join(lineas)


# --------------------------------------------------------------------------
# QAOA sobre las instancias de Joaquin
# --------------------------------------------------------------------------

def tabla_qaoa():
    suyo = json.load(open(RES / "json" / "qaoa_joaquin.json", encoding="utf-8"))["instancias"]
    reop = {tuple(f["a"]): f for f in
            json.load(open(RES / "json" / "qaoa_reoptimizado.json", encoding="utf-8"))["instancias"]}
    lineas = []
    for n in (5, 6):
        fs = [f for f in suyo if f["n"] == n]
        col = {
            "suyo": [f["qaoa"][-1]["p_exito"] for f in fs],
            "reop": [reop[tuple(f["a"])]["curva"][-1]["p_exito"] for f in fs],
            "l1": [f["adapt"]["1"]["nuestro"]["p_exito"] for f in fs],
            "l2": [f["adapt"]["2"]["nuestro"]["p_exito"] for f in fs],
        }
        k1 = np.median([f["adapt"]["1"]["nuestro"]["k"] for f in fs])
        k2 = np.median([f["adapt"]["2"]["nuestro"]["k"] for f in fs])
        celdas = [f"{np.median(v):.2f} ({int(np.sum(np.array(v) >= UMBRAL))}/{len(v)})" for v in col.values()]
        lineas.append(rf"{n} & {celdas[0]} & {celdas[1]} & {celdas[2]} & {k1:.0f} & {celdas[3]} & {k2:.0f} \\")
    return "\n".join(lineas)

# --------------------------------------------------------------------------
# tablas y cifras
# --------------------------------------------------------------------------

def tabla_instancias():
    inst = json.load(open(PROJECT_ROOT / "datos" / "mwnp_instancias.json", encoding="utf-8"))["instancias"]
    lineas = []
    for n in range(5, 11):
        ej = [i for i in inst if i["n"] == n][0]
        cajas = r" $\,|\,$ ".join(", ".join(map(str, c)) for c in ej["cajas_optimas"])
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
    # orden de los numeros
    o = cargar_ordenes()
    fr = np.array([np.mean(np.array([o[b][r] for r in range(10)]) >= UMBRAL) for b in sorted(o)])
    m["OrdSiempre"], m["OrdNunca"] = int((fr == 1).sum()), int((fr == 0).sum())
    m["OrdDepende"] = int(((fr > 0) & (fr < 1)).sum())
    m["OrdTasaAsc"] = f"{100*np.mean([o[b][0] >= UMBRAL for b in sorted(o)]):.0f}"
    m["OrdTasaMedia"] = f"{100*np.mean([o[b][r] >= UMBRAL for b in o for r in range(10)]):.0f}"
    dep = [b for b, f in zip(sorted(o), fr) if 0 < f < 1]
    m["OrdAscFallaEnDep"] = sum(o[b][0] < UMBRAL for b in dep)

    # rango
    def res(pat):
        ps = np.array([json.load(open(f))["resultado"]["p_exito"] for f in glob.glob(str(RES / pat))])
        return int((ps >= UMBRAL).sum()), len(ps)
    for l in (1, 2):
        a, na = res(f"mwnp_rango8/n6_l{l}_i??.json"); b, nb = res(f"mwnp/n6_l{l}_i??.json")
        m[f"RangoOchoL{'Uno' if l==1 else 'Dos'}"] = f"{a}/{na}"
        m[f"RangoDiecL{'Uno' if l==1 else 'Dos'}"] = f"{b}/{nb}"
    b1 = sum(json.load(open(f))["resultado"]["p_exito"] >= UMBRAL
             for f in glob.glob(str(RES / "mwnp" / "n6_l1_i0?.json")))
    b2 = sum(json.load(open(f))["resultado"]["p_exito"] >= UMBRAL
             for f in glob.glob(str(RES / "mwnp" / "n6_l1_i1?.json")))
    m["NSeisTandaUno"], m["NSeisTandaDos"] = b1, b2

    # sensibilidad a la maquina: n = 5, 6, l = 1, warm
    def pe(carp, n, i):
        return json.load(open(RES / carp / f"n{n}_l1_i{i:02d}.json"))["resultado"]["p_exito"] >= UMBRAL
    pares = [(n, i) for n in (5, 6) for i in range(10)]
    m["HilosFlips"] = sum(pe("mwnp_laptop_1h", n, i) != pe("mwnp_laptop_2h", n, i) for n, i in pares)
    m["MaquinaFlips"] = sum(pe("mwnp_laptop_2h", n, i) != pe("mwnp", n, i) for n, i in pares)
    m["MaquinaTotal"] = len(pares)
    # l = 1 contra l = 2 con ordenes (n = 5..9)
    o5 = json.load(open(RES / "json" / "ordenes_5a9.json", encoding="utf-8"))
    runs = cargar_ordenes_5a9()
    m["OcSignoDos"] = o5["signo_total"]["l2_mejor"]
    m["OcSignoUno"] = o5["signo_total"]["l1_mejor"]
    m["OcSignoP"] = _sci(o5["signo_total"]["p"])
    m["OcCorridas"] = len(runs)
    viejos = {(n, l, int(Path(f).stem.split("_i")[1])): json.load(open(f))["resultado"]["p_exito"]
              for n in range(5, 10) for l in (1, 2)
              for f in glob.glob(str(RES / "mwnp" / f"n{n}_l{l}_i??.json"))}
    comunes = [(k, runs[(k[0], k[1], k[2], 0)][0]) for k in viejos if (k[0], k[1], k[2], 0) in runs]
    m["OcReproVeredicto"] = sum((viejos[k] >= UMBRAL) == (p >= UMBRAL) for k, p in comunes)
    m["OcReproTotal"] = len(comunes)
    m["OcReproExactas"] = o5["reproducibilidad"]["iguales"]
    medio = [v[0] for k, v in runs.items() if k[1] == 2 and abs(v[0] - 0.5) < 0.02]
    m["OcMitad"] = len(medio)
    m["OcLDosCorridas"] = sum(k[1] == 2 for k in runs)
    asc = {(f["n"], f["l"]): f for f in o5["orden_ascendente"]}
    m["OcAscSeisAsc"] = f"{asc[(6, 1)]['tasa_asc']:.2f}"
    m["OcAscSeisOtros"] = f"{asc[(6, 1)]['tasa_otros']:.2f}"
    m["OcAscDifMax"] = f"{max(abs(f['tasa_asc'] - f['tasa_otros']) for k, f in asc.items() if k != (6, 1)):.2f}"
    for l, nom in ((1, "Uno"), (2, "Dos")):
        fs = [f for f in o5["por_n_l"] if f["l"] == l]
        m[f"OcPerdidas{nom}"] = sum(f["perdidas"] for f in fs)
        m[f"OcFallas{nom}"] = sum(f["fallas"] for f in fs)
    for l, nom in ((1, "Uno"), (2, "Dos")):
        fr = [f["frac_E_otra_menor"] for f in o5.get("contra_warm", [])
              if f["otra"] == "fija0" and f["l"] == l and f["n"] != "todos"]
        if fr:
            m[f"EFija{nom}Min"], m[f"EFija{nom}Max"] = f"{100 * min(fr):.0f}", f"{100 * max(fr):.0f}"
    for f in o5.get("contra_warm", []):
        if f["n"] == "todos":
            clave = f"Est{'Cold' if f['otra'] == 'cold' else 'Fija'}L{'Uno' if f['l'] == 1 else 'Dos'}"
            m[clave + "Warm"], m[clave + "Otra"] = f["gana_warm"], f["gana_otra"]
            m[clave + "P"] = _sci(f["p_signo"])
    # warm contra (ii): presupuesto y respuestas perdidas, l = 2
    for est, nom in (("warm", "Warm"), ("cold", "Cold")):
        rr = cargar_ordenes_5a9(est)
        if not rr:
            continue
        dos = {k: v for k, v in rr.items() if k[1] == 2}
        m[f"Conv{nom}Dos"] = sum(v[2]["stop_reason"] == "gradient_norm_below_epsilon" for v in dos.values())
        m[f"Tot{nom}Dos"] = len(dos)
        ocho = [v for k, v in dos.items() if k[0] == 8]
        m[f"K{nom}Ocho"] = f"{np.median([v[2]['num_parametros'] for v in ocho]):.0f}"
        m[f"Pmax{nom}Ocho"] = f"{np.mean([max(v[1]) >= UMBRAL for v in ocho]):.2f}"
        m[f"Fin{nom}Ocho"] = f"{np.mean([v[0] >= UMBRAL for v in ocho]):.2f}"
    for f in o5["por_n_l"]:
        if f["l"] == 2 and f["n"] in (8, 9):
            m[f"OcConvDos{'Ocho' if f['n'] == 8 else 'Nueve'}"] = f["convergidas"]

    # QAOA de Joaquin
    q = json.load(open(RES / "json" / "qaoa_joaquin.json", encoding="utf-8"))
    sube = pasos = 0
    for f in q["instancias"]:
        E = [c["E_j"] for c in sorted(f["qaoa"], key=lambda c: c["p"])]
        sube += sum(E[i + 1] > E[i] + 1e-6 for i in range(len(E) - 1))
        pasos += len(E) - 1
    m["QaoaSube"], m["QaoaPasos"] = sube, pasos
    m["QaoaDif"] = _sci(q["max_dif_energia"])
    r = json.load(open(RES / "json" / "qaoa_reoptimizado.json", encoding="utf-8"))["instancias"]
    baja = [f["curva"][0]["E_j"] / f["curva"][-1]["E_j"] for f in r]
    m["QaoaBajaMin"], m["QaoaBajaMax"] = f"{min(baja):.0f}", f"{max(baja):.0f}"

    r = json.load(open(RES / "json" / "rangos_instancias.json", encoding="utf-8"))
    m["RangoMuestras"] = r["muestras"]
    m["RangoErrorMax"] = f"{100 * max(f.get('error_unica', 0.0) for f in r['filas']):.1f}"
    m["RangoQuinceDiez"] = f"{100 * [f for f in r['filas'] if f['n'] == 10 and f['rango'] == '15'][0]['frac_unica']:.1f}"
    m["RangoDosnDiez"] = f"{100 * [f for f in r['filas'] if f['n'] == 10 and f['rango'] == '2n'][0]['frac_unica']:.1f}"
    tres = [f["frac_unica"] for f in r["filas"] if f["rango"] == "3n"]
    m["RangoTresMin"], m["RangoTresMax"] = f"{100 * min(tres):.1f}", f"{100 * max(tres):.1f}"

    m["TotalTodo"] = (len(todas) + len(glob.glob(str(RES / "mwnp_rango8" / "*.json")))
                      + len(glob.glob(str(RES / "mwnp_ordenes" / "*.json")))
                      # todas las estrategias de n = 5..9 con órdenes, menos las 100 de
                      # n = 6, l = 1, warm, que son copias de mwnp_ordenes
                      + len(glob.glob(str(RES / "mwnp_ordenes_5a9" / "*.json")))
                      - sum(1 for k in runs if k[0] == 6 and k[1] == 1))
    return m



# --------------------------------------------------------------------------
# benchmark ampliado: ADAPT (k <= 80) contra QAOA (p = 40)
# --------------------------------------------------------------------------

C_QAOA, C_INTERP = "#6b7280", "#7c3aed"


def cargar_k80():
    return json.load(open(RES / "json" / "k80_qaoa.json", encoding="utf-8"))


def _boot(t, reps=10000):
    t = np.asarray(t, dtype=float)
    medias = np.random.default_rng(0).choice(t, (reps, len(t))).mean(axis=1)
    return float(t.mean()), np.percentile(medias, [2.5, 97.5])


def k80_por_n(d):
    """Por n: tasas (con IC por instancia) y p_éxito medio de cada método."""
    fuera = {}
    for n in sorted({r["n"] for r in d["adapt"]} | {r["n"] for r in d["qaoa"]}):
        f = {"azar": 6 / 3 ** n}
        for l in (1, 2):
            rows = [r for r in d["adapt"] if r["n"] == n and r["l"] == l]
            if not rows:
                continue
            bases = sorted({r["base"] for r in rows})
            t = [np.mean([r["p"] >= UMBRAL for r in rows if r["base"] == b]) for b in bases]
            f[f"l{l}"] = {"tasa": _boot(t), "media": float(np.mean([r["p"] for r in rows])),
                          "tope": float(np.mean([r["k"] >= 80 for r in rows])), "corridas": len(rows)}
        q = [r for r in d["qaoa"] if r["n"] == n]
        if q:
            ids = sorted({r["id"] for r in q})
            mejor = [min((r for r in q if r["id"] == i), key=lambda r: r["E_j"])["p"] for i in ids]
            f["qaoa"] = {"tasa_mejor": _boot([m >= UMBRAL for m in mejor]),
                         "media": float(np.mean([r["p"] for r in q])),
                         "media_mejor": float(np.mean(mejor)),
                         "conv": float(np.mean([r["exito"] for r in q])), "reinicios": len(q)}
        it = [r["curva"][-1]["p_exito"] for r in d["interp"] if r["n"] == n]
        if it:
            f["interp"] = {"tasa": _boot([x >= UMBRAL for x in it]), "media": float(np.mean(it)),
                           "mediana": float(np.median(it))}
        fuera[n] = f
    return fuera


def fig_k80(por_n):
    ns = sorted(por_n)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    series = [("l1", r"ADAPT $\ell=1$", C_L1, "o", -0.09), ("l2", r"ADAPT $\ell=2$", C_L2, "s", -0.03),
              ("qaoa", r"QAOA $p=40$, mejor de 10", C_QAOA, "^", 0.03),
              ("interp", r"QAOA $p=40$, INTERP", C_INTERP, "D", 0.09)]
    for clave, nombre, c, m, dx in series:
        xs = [n for n in ns if clave in por_n[n]]
        if not xs:
            continue
        campo = "tasa_mejor" if clave == "qaoa" else "tasa"
        y = np.array([por_n[n][clave][campo][0] for n in xs])
        ic = np.array([por_n[n][clave][campo][1] for n in xs])
        a.errorbar(np.array(xs) + dx, y, yerr=[y - ic[:, 0], ic[:, 1] - y], color=c, marker=m,
                   ms=4.5, lw=1.4, capsize=2, label=nombre)
        med = [por_n[n][clave]["media"] for n in xs]
        b.plot(xs, med, color=c, marker=m, ms=4.5, lw=1.4, label=nombre)
    b.plot(ns, [por_n[n]["azar"] for n in ns], "k--", lw=1, label=r"azar, $6/3^n$")
    a.set_ylim(-0.03, 1.03)
    a.set_ylabel(r"fracci\'on con $p_{\rm \acute{e}xito}\geq 0.1$")
    b.set_yscale("log")
    b.set_ylabel(r"$p_{\rm \acute{e}xito}$ medio")
    for ax, t in ((a, "(a)"), (b, "(b)")):
        ax.set_xlabel(r"n\'umero de qutrits $n$")
        ax.set_xticks(ns)
        ax.set_title(t, loc="left", fontsize=10)
    b.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout(w_pad=2.0)
    guardar(fig, "k80_qaoa")


def _trazas_adapt(n, l):
    tr = []
    for carpeta in ("mwnp_ordenes_5a9", "mwnp_k80"):
        for f in glob.glob(str(RES / carpeta / f"n{n}_l{l}_i*.json")):
            if "_cold" in f or "_fija0" in f:
                continue
            t = json.load(open(f, encoding="utf-8"))["trazas"]["p_exito"][:81]
            tr.append(t + [t[-1]] * (81 - len(t)))
    return np.array(tr)


def fig_k80_parametros(d, ns=(6, 8)):
    """p_éxito medio contra número de parámetros: ADAPT (k) y QAOA INTERP (2p)."""
    fig, ejes = plt.subplots(1, len(ns), figsize=(7.2, 2.9), sharey=False)
    for ax, n in zip(ejes, ns):
        for l, c, nombre in ((1, C_L1, r"ADAPT $\ell=1$"), (2, C_L2, r"ADAPT $\ell=2$")):
            T = _trazas_adapt(n, l)
            if len(T):
                ax.plot(np.arange(81), T.mean(axis=0), color=c, lw=1.6, label=nombre)
        cur = [r["curva"] for r in d["interp"] if r["n"] == n]
        if cur:
            M = np.array([[e["p_exito"] for e in c] for c in cur])
            ax.plot(2 * np.arange(1, M.shape[1] + 1), M.mean(axis=0), color=C_INTERP, lw=1.6,
                    label=r"QAOA, INTERP")
        q = [r["p"] for r in d["qaoa"] if r["n"] == n]
        if q:
            ax.plot([80], [np.mean(q)], "^", color=C_QAOA, ms=7, label=r"QAOA $p=40$, reinicios")
        ax.axhline(6 / 3 ** n, color="k", ls="--", lw=1, label="azar")
        ax.set_yscale("log")
        ax.set_xlabel(r"n\'umero de par\'ametros")
        ax.set_title(rf"$n={n}$", fontsize=10)
    ejes[0].set_ylabel(r"$p_{\rm \acute{e}xito}$ medio")
    h, e = ejes[0].get_legend_handles_labels()
    fig.legend(h, e, frameon=False, fontsize=8, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=1.5, rect=(0, 0.08, 1, 1))
    guardar(fig, "k80_parametros")


def tabla_k80(por_n):
    lineas = []
    for n in sorted(por_n):
        f = por_n[n]
        def ad(l):
            x = f.get(f"l{l}")
            return "-- & --" if not x else rf"{x['tasa'][0]:.2f} & {_sci(x['media'])}"
        q = f.get("qaoa")
        qs = rf"{q['tasa_mejor'][0]:.2f} & {_sci(q['media'])}" if q else "-- & --"
        it = f.get("interp")
        its = rf"{it['tasa'][0]:.2f} & {_sci(it['media'])}" if it else "-- & --"
        tope = f"{f['l2']['tope']:.2f}" if "l2" in f else "--"
        lineas.append(rf"{n} & {_sci(f['azar'])} & {ad(1)} & {ad(2)} & {tope} & {qs} & {its} \\")
    return "\n".join(lineas)


# --------------------------------------------------------------------------
# compuertas nativas y figura estilo paper (mediana y rango intercuartil)
# --------------------------------------------------------------------------

def cargar_compuertas_qaoa():
    return {int(k): v for k, v in
            json.load(open(RES / "json" / "compuertas_qaoa_p40.json", encoding="utf-8")).items()}


def fig_compuertas(d, cq):
    ns = sorted({r["n"] for r in d["adapt"]})
    fig, ejes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=False)
    for ax, clave, cq_clave, titulo in zip(
            ejes, ("r_local", "ms", "total"), ("r_dos_niveles", "ms", "total"),
            (r"locales (rotaciones $R^{(i,j)}$)", r"no locales (MS)", r"totales")):
        for l, c, m in ((1, C_L1, "o"), (2, C_L2, "s")):
            q = np.array([np.percentile([r[clave] for r in d["adapt"] if r["n"] == n and r["l"] == l],
                                        [25, 50, 75]) for n in ns])
            ax.plot(ns, q[:, 1], color=c, marker=m, ms=4, lw=1.5, label=rf"ADAPT $\ell={l}$")
            ax.fill_between(ns, q[:, 0], q[:, 2], color=c, alpha=0.2, lw=0)
        ax.plot(ns, [cq[n]["gellmann"][cq_clave] for n in ns], color=C_QAOA, marker="^", ms=4, lw=1.5,
                label=r"QAOA $p=40$")
        ax.set_yscale("log")
        ax.set_title(titulo, fontsize=9.5)
        ax.set_xlabel(r"$n$")
        ax.set_xticks(ns[::2])
    ejes[0].set_ylabel(r"compuertas nativas")
    h, e = ejes[0].get_legend_handles_labels()
    fig.legend(h, e, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=1.0, rect=(0, 0.08, 1, 1))
    guardar(fig, "compuertas")


def fig_compuertas_pexito(d, cq, ns=(7, 10)):
    """Cada corrida: compuertas totales contra p_éxito."""
    rng = np.random.default_rng(3)
    fig, ejes = plt.subplots(1, len(ns), figsize=(7.2, 2.9))
    for ax, n in zip(ejes, ns):
        for l, c, m in ((1, C_L1, "o"), (2, C_L2, "s")):
            r = [x for x in d["adapt"] if x["n"] == n and x["l"] == l]
            ax.scatter([x["total"] for x in r], [max(x["p"], PISO) for x in r], s=9, color=c, marker=m,
                       alpha=0.6, lw=0, label=rf"ADAPT $\ell={l}$ (100 corridas)")
        qs = [max(x["p"], PISO) for x in d["qaoa"] if x["n"] == n]
        xq = cq[n]["gellmann"]["total"] * (1 + 0.04 * rng.uniform(-1, 1, len(qs)))
        ax.scatter(xq, qs, s=9, color=C_QAOA, marker="^", alpha=0.6, lw=0,
                   label=r"QAOA $p=40$ (200 reinicios)")
        ax.axhline(6 / 3 ** n, color="k", ls="--", lw=1, label="azar")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(PISO / 3, 3)
        ax.set_xlabel("compuertas nativas totales")
        ax.set_title(rf"$n={n}$", fontsize=10)
    ejes[0].set_ylabel(r"$p_{\rm \acute{e}xito}$")
    h, e = ejes[0].get_legend_handles_labels()
    fig.legend(h, e, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=1.5, rect=(0, 0.08, 1, 1))
    guardar(fig, "compuertas_pexito")


def tabla_compuertas(d, cq):
    lineas = []
    for n in sorted({r["n"] for r in d["adapt"]}):
        celdas = []
        for l in (1, 2):
            r = [x for x in d["adapt"] if x["n"] == n and x["l"] == l]
            for clave in ("r_local", "ms", "total"):
                celdas.append(f"{np.median([x[clave] for x in r]):.0f}")
        g = cq[n]["gellmann"]
        celdas += [str(g["r_dos_niveles"]), str(g["ms"]), str(g["total"])]
        lineas.append(rf"{n} & " + " & ".join(celdas) + r" \\")
    return "\n".join(lineas)


def _e_unif(a, cache={}):
    from funciones.utilidades_mwnp import hamiltoniano_joaquin
    clave = tuple(a)
    if clave not in cache:
        cache[clave] = float(np.mean(hamiltoniano_joaquin(a)))
    return cache[clave]


def _qaoa_barrido(n):
    """p -> listas (p_éxito, E/E_unif) de todos los reinicios a esa profundidad."""
    fuera = {}
    carpetas = {int(c.name[1:]): c for c in (RES / "qaoa_barrido").glob("p*")} if (RES / "qaoa_barrido").exists() else {}
    carpetas[40] = RES / "qaoa_p40"
    for p, carpeta in carpetas.items():
        ps, es = [], []
        for f in carpeta.glob(f"n{n}_i*_r*.json"):
            dd = json.load(open(f, encoding="utf-8"))
            ps.append(dd["resultado"]["p_exito"])
            es.append(dd["resultado"]["E_j"] / _e_unif(dd["instancia"]["a"]))
        if ps:
            fuera[p] = (np.array(ps), np.array(es))
    return dict(sorted(fuera.items()))


def _trazas_adapt_pe(n, l):
    P, E = [], []
    for carpeta in ("mwnp_ordenes_5a9", "mwnp_k80"):
        for f in glob.glob(str(RES / carpeta / f"n{n}_l{l}_i*.json")):
            if "_cold" in f or "_fija0" in f:
                continue
            t = json.load(open(f, encoding="utf-8"))["trazas"]
            p, e = t["p_exito"][:81], [x / t["energia_j"][0] for x in t["energia_j"][:81]]
            P.append(p + [p[-1]] * (81 - len(p)))
            E.append(e + [e[-1]] * (81 - len(e)))
    return np.array(P), np.array(E)


def fig_estilo_paper(d, ns=(6, 8, 10)):
    """
    Como las Figs. 1-3 del paper: mediana y rango intercuartil contra número de
    parámetros. ADAPT: 100 corridas (20 instancias x 5 órdenes) en cada k.
    QAOA: 200 reinicios (20 instancias x 10) en cada p. Arriba p_éxito, abajo
    la energía relativa a la de la superposición uniforme, E / E_unif.
    """
    fig, ejes = plt.subplots(2, len(ns), figsize=(7.4, 5.0))
    for j, n in enumerate(ns):
        P, E = {}, {}
        for l in (1, 2):
            P[l], E[l] = _trazas_adapt_pe(n, l)
        qb = _qaoa_barrido(n)
        it = [r["curva"] for r in d["interp"] if r["n"] == n]
        for fila, (ax, ylab) in enumerate(((ejes[0, j], r"$p_{\rm \acute{e}xito}$"),
                                          (ejes[1, j], r"$E/E_{\rm unif}$"))):
            for l, c in ((1, C_L1), (2, C_L2)):
                M = P[l] if fila == 0 else E[l]
                if not len(M):
                    continue
                q = np.percentile(np.maximum(M, PISO), [25, 50, 75], axis=0)
                ax.plot(np.arange(81), q[1], color=c, lw=1.5, label=rf"ADAPT $\ell={l}$")
                ax.fill_between(np.arange(81), q[0], q[2], color=c, alpha=0.2, lw=0)
            if qb:
                xs = np.array([2 * p for p in qb])
                q = np.array([np.percentile(np.maximum(v[fila], PISO), [25, 50, 75]) for v in qb.values()])
                ax.plot(xs, q[:, 1], color=C_QAOA, marker="^", ms=3.5, lw=1.3,
                        label=r"QAOA, 10 reinicios por instancia")
                ax.fill_between(xs, q[:, 0], q[:, 2], color=C_QAOA, alpha=0.25, lw=0)
            if it:
                nums = {i["id"]: i["a"] for i in json.load(open(PROJECT_ROOT / "datos" / "mwnp_instancias_5a12.json",
                                                                 encoding="utf-8"))["instancias"] if i["n"] == n}
                Mi = np.array([[e["p_exito"] if fila == 0 else e["E_j"] / _e_unif(nums[r["id"]])
                                for e in r["curva"]] for r in d["interp"] if r["n"] == n])
                ax.plot(2 * np.arange(1, Mi.shape[1] + 1), np.median(Mi, axis=0), color=C_INTERP,
                        lw=1.2, ls="--", label="QAOA INTERP (mediana)")
            if fila == 0:
                ax.axhline(6 / 3 ** n, color="k", ls=":", lw=1, label="azar")
                ax.set_title(rf"$n={n}$", fontsize=10)
            ax.set_yscale("log")
            if j == 0:
                ax.set_ylabel(ylab)
            if fila == 1:
                ax.set_xlabel(r"n\'umero de par\'ametros")
    h, e = ejes[0, 0].get_legend_handles_labels()
    fig.legend(h, e, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(h_pad=0.8, w_pad=1.0, rect=(0, 0.07, 1, 1))
    guardar(fig, "estilo_paper")


# --------------------------------------------------------------------------
# métrica normalizada, varios órdenes y diagnóstico de "peor que el azar"
# --------------------------------------------------------------------------

def _g(p, n):
    """G = (p_f - p_i) / (p_M - p_i), con p_i = 6/3^n (azar) y p_M = 1."""
    pi = 6 / 3 ** n
    return (np.asarray(p) - pi) / (1 - pi)


def fig_metrica_g(d):
    ns = sorted({r["n"] for r in d["adapt"]})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    series = []
    for l, c, m in ((1, C_L1, "o"), (2, C_L2, "s")):
        series.append((rf"ADAPT $\ell={l}$", c, m,
                       {n: [r["p"] for r in d["adapt"] if r["n"] == n and r["l"] == l] for n in ns}))
    series.append((r"QAOA $p=40$, reinicios", C_QAOA, "^",
                   {n: [r["p"] for r in d["qaoa"] if r["n"] == n] for n in ns}))
    series.append((r"QAOA INTERP", C_INTERP, "D",
                   {n: [r["curva"][-1]["p_exito"] for r in d["interp"] if r["n"] == n] for n in ns}))
    for nombre, c, m, dat in series:
        xs = [n for n in ns if dat[n]]
        a.plot(xs, [np.mean(_g(dat[n], n)) for n in xs], color=c, marker=m, ms=4, lw=1.4, label=nombre)
        b.plot(xs, [np.mean(_g(dat[n], n) < 0) for n in xs], color=c, marker=m, ms=4, lw=1.4)
    a.axhline(0, color="k", lw=0.8)
    a.set_yscale("symlog", linthresh=1e-4)
    a.set_ylabel(r"$G$ medio")
    b.set_ylabel(r"fracci\'on con $G<0$ (peor que el azar)")
    b.set_ylim(-0.03, 1.03)
    for ax, t in ((a, "(a)"), (b, "(b)")):
        ax.set_xlabel(r"n\'umero de qutrits $n$")
        ax.set_xticks(ns)
        ax.set_title(t, loc="left", fontsize=10)
    h, e = a.get_legend_handles_labels()
    fig.legend(h, e, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=2.0, rect=(0, 0.08, 1, 1))
    guardar(fig, "metrica_g")


def tabla_multiorden(d):
    g = defaultdict(dict)
    for r in d["adapt"]:
        g[(r["n"], r["l"], r["base"])][r["orden"]] = r
    lineas = []
    for n in sorted({r["n"] for r in d["adapt"]}):
        celdas = []
        for l in (1, 2):
            bases = sorted({k[2] for k in g if k[0] == n and k[1] == l})
            tasa = np.mean([np.mean([x["p"] >= UMBRAL for x in g[(n, l, b)].values()]) for b in bases])
            mejor = np.mean([min(g[(n, l, b)].values(), key=lambda x: x["E_j"])["p"] >= UMBRAL for b in bases])
            alguno = np.mean([any(x["p"] >= UMBRAL for x in g[(n, l, b)].values()) for b in bases])
            celdas += [f"{tasa:.2f}", f"{mejor:.2f}", f"{alguno:.2f}"]
        q = [x for x in d["qaoa"] if x["n"] == n]
        ids = sorted({x["id"] for x in q})
        qm = np.mean([min((x for x in q if x["id"] == i), key=lambda x: x["E_j"])["p"] >= UMBRAL for i in ids])
        lineas.append(rf"{n} & " + " & ".join(celdas) + rf" & {qm:.2f} \\")
    return "\n".join(lineas)


def cifras_diagnostico(d):
    """ADAPT l = 1 en n = 12: energía, concentración y desbalance de lo que encuentra."""
    m = {}
    E, top, des, conv = [], [], [], []
    for f in glob.glob(str(RES / "mwnp_k80" / "n12_l1_i*.json")):
        dd = json.load(open(f, encoding="utf-8"))
        t, r = dd["trazas"], dd["resultado"]
        E.append(t["energia_j"][-1] / t["energia_j"][0])
        top.append(sum(x["probabilidad"] for x in r["top_particiones"]))
        des.append(r["top_particiones"][0]["desbalance"])
        conv.append(r["stop_reason"] == "gradient_norm_below_epsilon")
    m["DgEunif"] = _sci(float(np.median(E)))
    m["DgTop"] = f"{100 * np.median(top):.0f}"
    m["DgTopN"] = len(r["top_particiones"])
    q = np.percentile(des, [25, 75])
    m["DgDesLo"], m["DgDesHi"] = f"{q[0]:.0f}", f"{q[1]:.0f}"
    m["DgConv"] = f"{100 * np.mean(conv):.0f}"
    m["DgPMax"] = _sci(float(max(r2["p"] for r2 in d["adapt"] if r2["n"] == 12 and r2["l"] == 1)))
    m["DgAzar"] = _sci(6 / 3 ** 12)
    return m


# --------------------------------------------------------------------------
# mapa de la varianza del gradiente (n, k)
# --------------------------------------------------------------------------

def cargar_varianza_mapa():
    filas = []
    for f in (RES / "varianza_mapa").glob("*.json"):
        d = json.load(open(f, encoding="utf-8"))
        for r in d["filas"]:
            filas.append({"n": d["n"], "l": d["l"], "familia": d["familia"], "inst": d["instancia"], **r})
    return filas


def _mapa(filas, l, familia):
    ns = sorted({r["n"] for r in filas})
    ks = sorted({r["k"] for r in filas})
    M = np.full((len(ks), len(ns)), np.nan)
    for i, k in enumerate(ks):
        for j, n in enumerate(ns):
            v = [r["var"] for r in filas if r["n"] == n and r["k"] == k and r["l"] == l and r["familia"] == familia]
            if len(v) >= 5:
                M[i, j] = np.median(v)
    return ns, ks, M


def fig_varianza_mapa(filas):
    if not filas:
        return
    fig, ejes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    vmin = np.log10(min(r["var"] for r in filas if r["var"] > 0))
    vmax = np.log10(max(r["var"] for r in filas))
    for i, familia in enumerate(("aleatoria", "adapt")):
        for j, l in enumerate((1, 2)):
            ns, ks, M = _mapa(filas, l, familia)
            ax = ejes[i, j]
            im = ax.imshow(np.log10(M), origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(ns)))
            ax.set_xticklabels(ns)
            ax.set_yticks(range(len(ks)))
            ax.set_yticklabels(ks)
            ax.set_title(("circuitos aleatorios" if familia == "aleatoria" else "circuitos de ADAPT")
                         + rf", $\ell={l}$", fontsize=9.5)
            if i == 1:
                ax.set_xlabel(r"n\'umero de qutrits $n$")
            if j == 0:
                ax.set_ylabel(r"capas $k$")
    cb = fig.colorbar(im, ax=ejes, fraction=0.03, pad=0.02)
    cb.set_label(r"$\log_{10}$ Var$[\partial_\theta E]$ (mediana)")
    guardar(fig, "varianza_mapa")

    # cortes: Var contra n para cada k, l = 1 y 2, circuitos aleatorios, con la referencia 3^{-n}
    fig, ejes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, l in zip(ejes, (1, 2)):
        ns, ks, M = _mapa(filas, l, "aleatoria")
        for i, k in enumerate(ks):
            ax.plot(ns, M[i], marker="o", ms=3, lw=1.3, color=cmap(i / (len(ks) - 1) * 0.9), label=rf"$k={k}$")
        ref = M[-1, 0] * 3.0 ** (-(np.array(ns) - ns[0]))
        ax.plot(ns, ref, "r--", lw=1, label=r"$\propto3^{-n}$")
        ax.set_yscale("log")
        ax.set_xlabel(r"$n$")
        ax.set_title(rf"circuitos aleatorios, $\ell={l}$", fontsize=9.5)
    ejes[0].set_ylabel(r"Var$[\partial_\theta E]$ (mediana)")
    ejes[1].legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(w_pad=1.0)
    guardar(fig, "varianza_cortes")

    # superficie 3D pedida: l = 1, circuitos aleatorios
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    ns, ks, M = _mapa(filas, 1, "aleatoria")
    X, Y = np.meshgrid(ns, np.log2(ks))
    fig = plt.figure(figsize=(5.0, 3.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, np.log10(M), cmap="viridis", edgecolor="k", lw=0.3, alpha=0.9)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\log_2 k$")
    ax.set_zlabel(r"$\log_{10}$ Var")
    ax.view_init(elev=25, azim=-130)
    guardar(fig, "varianza_superficie")


def cifras_varianza_mapa(filas):
    m = {}
    if not filas:
        return m
    for l, nom in ((1, "Uno"), (2, "Dos")):
        ns, ks, M = _mapa(filas, l, "aleatoria")
        i = ks.index(16) if 16 in ks else len(ks) - 1
        ok = ~np.isnan(M[i])
        alfa = -np.polyfit(np.array(ns)[ok], np.log(M[i][ok]), 1)[0]
        m[f"VmAlfa{nom}"] = f"{alfa:.2f}"
        m[f"VmRazon{nom}"] = f"{M[i][ok][-1] / (M[i][ok][0] * 3.0 ** (-(np.array(ns)[ok][-1] - np.array(ns)[ok][0]))):.0f}"
    m["VmInstancias"] = len({(r["n"], r["inst"]) for r in filas}) // len({r["n"] for r in filas})
    return m


def main():
    estilo()
    por, todas = cargar_corridas()
    INF.mkdir(parents=True, exist_ok=True)

    fig_exito_vs_n(por)
    fig_pareado(por)
    fig_trazas(por, 6, 3, "trazas_n6_i03")
    fig_trazas(por, 7, 4, "trazas_n7_i04")
    fig_varianza()
    fig_tiempos(por)
    fig_ordenes(cargar_ordenes())
    fig_rango()

    o5 = json.load(open(RES / "json" / "ordenes_5a9.json", encoding="utf-8"))
    fig_ordenes_5a9(o5)
    fig_ordenes_iteracion(cargar_ordenes_5a9())
    (INF / "tabla_ordenes.tex").write_text(tabla_ordenes(o5), encoding="utf-8")
    fig_estrategias(o5)
    (INF / "tabla_estrategias.tex").write_text(tabla_estrategias(o5), encoding="utf-8")
    (INF / "tabla_qaoa.tex").write_text(tabla_qaoa(), encoding="utf-8")
    (INF / "tabla_pool.tex").write_text(tabla_pool(), encoding="utf-8")
    (INF / "tabla_rangos.tex").write_text(tabla_rangos(), encoding="utf-8")
    (INF / "tabla_instancias.tex").write_text(tabla_instancias(), encoding="utf-8")
    (INF / "tabla_fase1.tex").write_text(tabla_fase1(por), encoding="utf-8")
    (INF / "tabla_fase2.tex").write_text(tabla_fase2(por), encoding="utf-8")
    k80 = cargar_k80()
    por_n = k80_por_n(k80)
    fig_k80(por_n)
    fig_k80_parametros(k80)
    (INF / "tabla_k80.tex").write_text(tabla_k80(por_n), encoding="utf-8")
    cq = cargar_compuertas_qaoa()
    fig_compuertas(k80, cq)
    fig_compuertas_pexito(k80, cq)
    (INF / "tabla_compuertas.tex").write_text(tabla_compuertas(k80, cq), encoding="utf-8")
    fig_estilo_paper(k80)
    fig_metrica_g(k80)
    (INF / "tabla_multiorden.tex").write_text(tabla_multiorden(k80), encoding="utf-8")
    vm = cargar_varianza_mapa()
    fig_varianza_mapa(vm)
    m = cifras(por, todas)
    nmax = max(n for n in por_n if "qaoa" in por_n[n] and "l2" in por_n[n])
    m["KNmax"] = nmax
    m["KMediaDosNmax"] = _sci(por_n[nmax]["l2"]["media"])
    m["KMediaQaoaNmax"] = _sci(por_n[nmax]["qaoa"]["media"])
    m["KRazonNmax"] = f"{por_n[nmax]['l2']['media'] / por_n[nmax]['qaoa']['media']:.0f}"
    topes = [por_n[n]["l2"]["tope"] for n in por_n if n >= 10 and "l2" in por_n[n]]
    m["KTopeMin"], m["KTopeMax"] = f"{100 * min(topes):.0f}", f"{100 * max(topes):.0f}"
    conv = [por_n[n]["qaoa"]["conv"] for n in por_n if "qaoa" in por_n[n]]
    m["KQaoaConvMin"], m["KQaoaConvMax"] = f"{100 * min(conv):.0f}", f"{100 * max(conv):.0f}"
    it_n = max(n for n in por_n if "interp" in por_n[n])
    m["KInterpN"], m["KInterpMediana"] = it_n, _sci(por_n[it_n]["interp"]["mediana"])
    m["KInterpRazon"] = f"{por_n[it_n]['interp']['mediana'] / por_n[it_n]['azar']:.0f}"
    # umbral 0.5: ADAPT l = 2 contra INTERP donde INTERP gana con umbral 0.1
    for n, nom in ((5, "Cinco"), (6, "Seis"), (7, "Siete")):
        pa = [r["p"] for r in k80["adapt"] if r["n"] == n and r["l"] == 2]
        pi = [r["curva"][-1]["p_exito"] for r in k80["interp"] if r["n"] == n]
        m[f"KMedioAdapt{nom}"] = f"{np.mean(np.array(pa) >= 0.5):.2f}"
        m[f"KMedioInterp{nom}"] = f"{np.mean(np.array(pi) >= 0.5):.2f}"
        m[f"KMinInterp{nom}"] = f"{min(pi):.2f}"
    m.update(cifras_diagnostico(k80))
    m.update(cifras_varianza_mapa(vm))
    m["KCorridasAdapt"] = len(k80["adapt"])
    m["KReinicios"] = len(k80["qaoa"])
    (INF / "numeros_mwnp.tex").write_text(
        "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items()) + "\n", encoding="utf-8")
    for k, v in m.items():
        print(f"  {k:18s} {v}")
    print(f"\nfiguras en {FIG.relative_to(PROJECT_ROOT)}, tablas y cifras en {INF.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
