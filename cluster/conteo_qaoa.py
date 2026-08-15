"""
Native-gate cost of QAOA at a matched parameter budget.

The point of this script is a circuit-structure comparison, not a variational
one: no QAOA optimisation is performed and no optimal angles are needed. The
number of native gates of a QAOA circuit depends only on the graph and on the
number of layers, so fixing every angle to an arbitrary value (0.1 here) is
enough. What is being asked is how many gates QAOA needs to *offer* the same
number of variational parameters that Qudit-ADAPT ends up using.

Both Hamiltonians decompose into mutually commuting terms, so the layers are
exact products with no Trotter error:

    exp(-i beta H_M)  = prod_j exp(-i beta Jx_j)          single-qudit
    exp(-i gamma H_C) = prod_terms exp(-i gamma T)        T diagonal

The cost layer admits two decompositions, and they do not cost the same:

    angular    Jz_i Jz_j - 2 Jz_i^2 - 2 Jz_j^2 + 3 Jz_i^2 Jz_j^2   (4 terms)
    Gell-Mann  l3_i l3_j + l8_i l8_j - (4/3) I                     (2 terms)

The Gell-Mann form is the cheaper of the two and is the one quoted in the
appendix, so that QAOA is counted in its most favourable encoding.

Gate counting reuses `conteo_compuertas`, the same ladder model applied to the
Qudit-ADAPT pool: each local factor is diagonalised (two-level rotations read
off the matrix, exactly), a parametrised diagonal phase is applied, and the
diagonalisations are undone, with 2(w-1) Mølmer-Sørensen gates for a generator
supported on w sites.

    python cluster/conteo_qaoa.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import numpy as np

from funciones.utilidades_bp import conteo_compuertas, load_bp_result

JSON_DIR = PROJECT_ROOT / "resultados" / "json"

# Instancias del manuscrito. G_1 es la de la Fig. 1a y de la Tabla I; G_2 es la
# del apéndice, la misma sobre la que se comparan los pools.
G1 = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 4), (3, 6), (4, 5), (5, 6)]
G2 = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5), (5, 6)]


def _etiqueta(*factores):
    """Etiqueta canónica de un monomio, en el formato que lee conteo_compuertas."""
    return str(tuple(sorted(factores)))


def terminos_mixer(n):
    """
    Términos del mixer H_M = sum_j Jx_j.

    Cada uno actúa sobre un solo sitio, de modo que exp(-i beta Jx_j) es una
    unitaria de un qutrit y su costo no depende de la base en que se escriba
    el pool: es la misma operación física en ambos casos.
    """
    return [_etiqueta((j, "x")) for j in range(1, n + 1)]


def terminos_costo(edges, base):
    """Términos de H_C en la base pedida. Todos son diagonales y conmutan."""
    if base == "angular":
        terminos = []
        for i, j in edges:
            terminos.append(_etiqueta((i, "z"), (j, "z")))
            terminos.append(_etiqueta((i, "z"), (i, "z")))
            terminos.append(_etiqueta((j, "z"), (j, "z")))
            terminos.append(_etiqueta((i, "z"), (i, "z"), (j, "z"), (j, "z")))
        return terminos

    if base == "gellmann":
        # H_C|_(i,j) = l3_i l3_j + l8_i l8_j - (4/3) I. El término identidad es
        # una fase global y no cuesta compuertas.
        terminos = []
        for i, j in edges:
            terminos.append(_etiqueta((i, 3), (j, 3)))
            terminos.append(_etiqueta((i, 8), (j, 8)))
        return terminos

    raise ValueError("base debe ser 'angular' o 'gellmann'")


def verificar_costo_gellmann(n, edges, tol=1e-10):
    """
    Comprueba numéricamente la identidad del apéndice,
        H_C|_(i,j) = l3_i l3_j + l8_i l8_j - (4/3) I,
    reconstruyendo H_C por aristas en ambas formas y comparándolas.
    """
    import qutip as qt

    from funciones.utilidades import Hp_qutip
    from funciones.utilidades_gellmann import GELLMANN

    def embeber(M, sitio):
        ops = [qt.qeye(3)] * n
        ops[sitio - 1] = qt.Qobj(np.asarray(M, dtype=complex))
        return qt.tensor(ops)

    H_gm = 0
    for i, j in edges:
        for a in (3, 8):
            H_gm += embeber(GELLMANN[a], i) * embeber(GELLMANN[a], j)
        H_gm += -(4.0 / 3.0) * qt.tensor([qt.qeye(3)] * n)

    return float(np.max(np.abs((Hp_qutip(n, edges) - H_gm).full()))) < tol


def _sumar(labels, base):
    r = ms = 0
    for label in labels:
        c = conteo_compuertas(label, base=base)
        r += c["r_dos_niveles"]
        ms += c["ms"]
    return r, ms


def conteo_qaoa(n, edges, num_params, base="gellmann"):
    """
    Compuertas nativas de un circuito QAOA que ofrece `num_params` parámetros.

    Con p capas completas el circuito tiene 2p parámetros. Para un número impar
    se añade media capa —sólo el término de costo— al final, que es la
    convención habitual y la que permite igualar exactamente presupuestos como
    los 21 parámetros de Qudit-ADAPT con l = 1.
    """
    # El mixer es un producto de unitarias de un qutrit, así que se cuenta con
    # la maquinaria angular en ambos casos.
    r_mix, ms_mix = _sumar(terminos_mixer(n), "angular")
    r_cost, ms_cost = _sumar(terminos_costo(edges, base), base)

    capas = num_params // 2
    media = num_params % 2

    r = capas * (r_mix + r_cost) + media * r_cost
    ms = capas * (ms_mix + ms_cost) + media * ms_cost

    return {
        "num_params": int(num_params),
        "capas": int(capas),
        "media_capa": bool(media),
        "base": str(base),
        "r_mixer_capa": int(r_mix),
        "ms_mixer_capa": int(ms_mix),
        "r_costo_capa": int(r_cost),
        "ms_costo_capa": int(ms_cost),
        "r_dos_niveles": int(r),
        "ms": int(ms),
        "total": int(r + ms),
    }


def conteo_adapt(archivo):
    """Compuertas nativas acumuladas del ansatz final de una corrida ADAPT."""
    from funciones.utilidades_bp import conteo_compuertas_ansatz

    res = load_bp_result(archivo)
    c = conteo_compuertas_ansatz(res, acumulado=True)
    return {
        "num_params": int(res["num_ansatz_ops"]),
        "r_dos_niveles": int(c["r_dos_niveles"][-1]),
        "ms": int(c["ms"][-1]),
        "total": int(c["total"][-1]),
        "base": res.get("base") or "angular",
        "pool_size": int(res["pool_size"]),
    }


def conteo_adapt_en_k(archivo, k):
    """Compuertas acumuladas del ansatz truncado a los primeros k operadores."""
    from funciones.utilidades_bp import conteo_compuertas_ansatz

    res = load_bp_result(archivo)
    c = conteo_compuertas_ansatz(res, acumulado=True)
    return {
        "num_params": int(k),
        "r_dos_niveles": int(c["r_dos_niveles"][k]),
        "ms": int(c["ms"][k]),
        "total": int(c["total"][k]),
        "base": res.get("base") or "angular",
    }


def _fila(nombre, d):
    return (f"  {nombre:34s}{d['num_params']:>7d}{d['r_dos_niveles']:>10d}"
            f"{d['ms']:>8d}{d['total']:>9d}")


def _cabecera(titulo):
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    print(f"  {'':34s}{'params':>7s}{'R(i,j)':>10s}{'MS':>8s}{'total':>9s}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_qaoa", default="gellmann",
                   choices=["gellmann", "angular"],
                   help="decomposición del layer de costo (gellmann es la más barata)")
    args = p.parse_args()

    n = 6

    for nombre, edges in (("G_1", G1), ("G_2", G2)):
        ok = verificar_costo_gellmann(n, edges)
        print(f"identidad H_C = l3 l3 + l8 l8 - (4/3)I verificada en {nombre}: {ok}")
        if not ok:
            raise SystemExit("la identidad de Gell-Mann para H_C no se cumple")

    # ---------------------------------------------------------------- G_1 ---
    # Instancia de la Fig. 1a. Los presupuestos son los de la Tabla I: 21
    # parámetros con l = 1 y 40 con l = 2.
    _cabecera(f"G_1  (Fig. 1a, {len(G1)} aristas)")
    # Se usan las corridas de la Tabla I (`*_curva.json`): la de l = 2 del
    # barrido de BP se detuvo en 35 parámetros por su tope de iteraciones.
    for archivo, l, k in (("fig1a_l1_curva.json", 1, 21),
                          ("fig1a_l2_curva.json", 2, 40)):
        a = conteo_adapt(archivo)
        q = conteo_qaoa(n, G1, k, base=args.base_qaoa)
        assert a["num_params"] == k, f"{archivo}: k={a['num_params']}, se esperaba {k}"
        print(_fila(f"Qudit-ADAPT  l={l}", a))
        print(_fila(f"QAOA         ({q['capas']} capas)", q))
        print(f"  {'':34s}{'':7s}{'razón':>10s}"
              f"{q['ms'] / a['ms']:>7.1f}x{q['total'] / a['total']:>8.1f}x")

    # ---------------------------------------------------------------- G_2 ---
    # Instancia del apéndice. Los presupuestos son los de sus dos tablas.
    _cabecera(f"G_2  (apéndice, {len(G2)} aristas)")
    ang = conteo_adapt("bp_n6_l2_grafo2_r20_k30.json")
    gm21 = conteo_adapt_en_k("bp_n6_l2_grafo2_r20_k30_gellmann.json", 21)
    gm30 = conteo_adapt("bp_n6_l2_grafo2_r20_k30_gellmann.json")

    print(_fila("Qudit-ADAPT  angular  l=2", ang))
    print(_fila("Qudit-ADAPT  Gell-Mann  (eps<1e-3)", gm21))
    print(_fila("Qudit-ADAPT  Gell-Mann  (final)", gm30))
    for k in (16, 21, 30):
        q = conteo_qaoa(n, G2, k, base=args.base_qaoa)
        print(_fila(f"QAOA  ({q['capas']} capas)", q))

    q16 = conteo_qaoa(n, G2, 16, base=args.base_qaoa)
    q21 = conteo_qaoa(n, G2, 21, base=args.base_qaoa)
    print(f"\n  a igual presupuesto de parámetros:")
    print(f"    16 params: QAOA {q16['total']:5d}  vs  angular   {ang['total']:4d}"
          f"   ({q16['total'] / ang['total']:.1f}x)")
    print(f"    21 params: QAOA {q21['total']:5d}  vs  Gell-Mann {gm21['total']:4d}"
          f"   ({q21['total'] / gm21['total']:.1f}x)")

    # --------------------------------------------------------- desglose ---
    print("\n" + "=" * 70)
    print("desglose por capa (G_2), en las dos decomposiciones del costo")
    print("=" * 70)
    for base in ("angular", "gellmann"):
        q = conteo_qaoa(n, G2, 2, base=base)
        print(f"  {base:10s} mixer  R={q['r_mixer_capa']:3d}  MS={q['ms_mixer_capa']:3d}"
              f"   |  costo  R={q['r_costo_capa']:3d}  MS={q['ms_costo_capa']:3d}"
              f"   |  capa  R={q['r_dos_niveles']:3d}  MS={q['ms']:3d}")


if __name__ == "__main__":
    main()
