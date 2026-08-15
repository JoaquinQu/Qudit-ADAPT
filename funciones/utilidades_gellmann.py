"""
Pool contradiabático en la base de Gell-Mann
============================================

Reformulación del pool de CD-ADAPT-VQE para qutrits usando las matrices de
Gell-Mann {lambda_1, ..., lambda_8} en vez de la base de momento angular
{Jx, Jy, Jz} y sus productos.

Motivación
----------
El pool del algoritmo original se construye tomando los términos individuales
de los conmutadores anidados [H_ad, [H_ad, ... dH_ad]] expresados en monomios
de momento angular. Ahí aparecen productos como Jy^(1) Jz^(1) en un MISMO
sitio, que no son hermíticos, y por eso el código original los "hermitiza" al
final con (O + O^dag)/2.

En la base de Gell-Mann eso no hace falta, porque el álgebra cierra
linealmente:

    [lambda_a, lambda_b] = 2i sum_c f_abc lambda_c

Cada conmutador devuelve otra vez una combinación lineal de lambdas del mismo
sitio, así que todo término del pool queda como un producto de lambdas sobre
sitios DISTINTOS —que conmutan entre sí y son hermíticas— multiplicado por i.
O sea: hermítico por construcción, sin parche posterior.

Punto clave para la comparación
-------------------------------
Las dos bases generan el MISMO espacio (su(3)), así que H_M y H_C son
exactamente los mismos operadores en ambas. Lo que cambia es cómo se
DESCOMPONEN, y por lo tanto qué términos individuales produce el conmutador.
Ése es el origen de que el pool sea distinto: el pool es dependiente de la
base elegida.

Convenciones
------------
Base computacional |+1> = (1,0,0)^T, |0> = (0,1,0)^T, |-1> = (0,0,1)^T, que es
la de qutip (`jmat(1,'z') = diag(1,0,-1)`), de modo que todo es directamente
comparable con `utilidades.py`.

Representación
--------------
Un operador es un dict {string: coeficiente}, donde `string` es una tupla de
pares (sitio, a) con a en 1..8, ordenada por sitio y con sitios distintos. La
tupla vacía es la identidad. Como el producto de dos lambdas del mismo sitio
se reexpande en la base, esta representación es cerrada bajo producto y
conmutador.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import qutip as qt


# ============================================================
# 1. Matrices de Gell-Mann y constantes de estructura
# ============================================================

_S3 = np.sqrt(3.0)

GELLMANN = {
    1: np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex),
    2: np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex),
    3: np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex),
    4: np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex),
    5: np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex),
    6: np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex),
    7: np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex),
    8: np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / _S3,
}

I3 = np.eye(3, dtype=complex)


def _estructura():
    """
    Constantes de estructura de su(3), calculadas numéricamente a partir de
    las matrices en vez de tipeadas a mano (que es donde se cometen errores):

        f_abc = -(i/4) Tr([l_a, l_b] l_c)     antisimétrica
        d_abc =  (1/4) Tr({l_a, l_b} l_c)     simétrica
    """
    f = np.zeros((9, 9, 9), dtype=float)
    d = np.zeros((9, 9, 9), dtype=float)

    for a in range(1, 9):
        for b in range(1, 9):
            La, Lb = GELLMANN[a], GELLMANN[b]
            conm = La @ Lb - Lb @ La
            anti = La @ Lb + Lb @ La
            for c in range(1, 9):
                Lc = GELLMANN[c]
                f[a, b, c] = float(np.real(-0.25j * np.trace(conm @ Lc)))
                d[a, b, c] = float(np.real(0.25 * np.trace(anti @ Lc)))

    return f, d


F_ABC, D_ABC = _estructura()


def producto_local(a, b):
    """
    Producto de dos generadores en el mismo sitio, en la base {I, lambda}:

        l_a l_b = (2/3) delta_ab I + sum_c (d_abc + i f_abc) l_c

    Retorna (coef_identidad, {c: coeficiente}).
    """
    coef_I = (2.0 / 3.0) if a == b else 0.0
    resto = {}
    for c in range(1, 9):
        coef = D_ABC[a, b, c] + 1j * F_ABC[a, b, c]
        if abs(coef) > 1e-12:
            resto[c] = coef
    return coef_I, resto


# ============================================================
# 2. Álgebra de strings de Gell-Mann
# ============================================================

def _norm_string(pares):
    """Ordena por sitio. Los sitios deben ser distintos."""
    return tuple(sorted(pares, key=lambda p: p[0]))


def gm_limpiar(op, tol=1e-12):
    return {s: c for s, c in op.items() if abs(c) > tol}


def gm_suma(*ops):
    salida = {}
    for op in ops:
        for s, c in op.items():
            salida[s] = salida.get(s, 0j) + c
    return gm_limpiar(salida)


def gm_escalar(alpha, op):
    if alpha == 0:
        return {}
    return gm_limpiar({s: alpha * c for s, c in op.items()})


def _mult_strings(s1, s2):
    """
    Producto de dos strings. En los sitios compartidos se expande
    l_a l_b en la base local, lo que puede generar varios términos.

    Retorna dict {string: coeficiente}.
    """
    d1 = dict(s1)
    d2 = dict(s2)

    comunes = sorted(set(d1) & set(d2))
    solos = [(s, a) for s, a in d1.items() if s not in d2]
    solos += [(s, a) for s, a in d2.items() if s not in d1]

    # cada sitio compartido aporta (coef_I -> sin lambda) o (c -> lambda_c)
    parciales = [((), 1.0 + 0j)]

    for sitio in comunes:
        coef_I, resto = producto_local(d1[sitio], d2[sitio])

        nuevos = []
        for pares, coef in parciales:
            if coef_I != 0.0:
                nuevos.append((pares, coef * coef_I))
            for c, cc in resto.items():
                nuevos.append((pares + ((sitio, c),), coef * cc))
        parciales = nuevos

    salida = {}
    for pares, coef in parciales:
        s = _norm_string(tuple(solos) + pares)
        salida[s] = salida.get(s, 0j) + coef

    return gm_limpiar(salida)


def gm_producto(A, B):
    salida = {}
    for s1, c1 in A.items():
        for s2, c2 in B.items():
            for s, c in _mult_strings(s1, s2).items():
                salida[s] = salida.get(s, 0j) + c1 * c2 * c
    return gm_limpiar(salida)


def gm_conmutador(A, B):
    """[A, B] = AB - BA."""
    return gm_suma(gm_producto(A, B), gm_escalar(-1.0, gm_producto(B, A)))


def gm_conmutadores_anidados(H, dH, order=3):
    """O_k = [H, [H, ... [H, dH]]], k veces. Mismo esquema que utilidades.py."""
    salida = {}
    actual = dH
    for k in range(1, order + 1):
        actual = gm_conmutador(H, actual)
        salida[k] = actual
    return salida


# ============================================================
# 3. Conversión a matrices
# ============================================================

def string_a_qutip(n, s):
    """String -> operador de n qutrits (qutip.Qobj)."""
    d = dict(s)
    ops = [qt.Qobj(GELLMANN[d[j]]) if j in d else qt.qeye(3)
           for j in range(1, n + 1)]
    return qt.tensor(ops)


def gm_a_qutip(n, op):
    """Operador en representación de strings -> qutip.Qobj."""
    total = 0
    for s, c in op.items():
        total = total + complex(c) * string_a_qutip(n, s)
    if total == 0:
        total = qt.tensor([qt.qeye(3) for _ in range(n)])
    return total


def peso(s):
    """Número de sitios en los que el string actúa no trivialmente."""
    return len(s)


# ============================================================
# 4. Hamiltonianos en base de Gell-Mann
# ============================================================

def Hi_gm(n, omega0=1.0):
    """
    H_i = -omega0 sum_j X~_j,  con  X~ = Jz^2 + sqrt(2) Jx.

    Descomposición (verificada numéricamente contra la matriz explícita):

        X~ = (2/3) I + (1/2) l_3 - (sqrt(3)/6) l_8 + l_1 + l_6
    """
    op = {}
    for j in range(1, n + 1):
        op = gm_suma(op, {
            (): -omega0 * (2.0 / 3.0),
            ((j, 3),): -omega0 * 0.5,
            ((j, 8),): omega0 * (_S3 / 6.0),
            ((j, 1),): -omega0 * 1.0,
            ((j, 6),): -omega0 * 1.0,
        })
    return op


def Hp_gm(n, edges):
    """
    H_p de Max-3-Cut, idéntico al `Hp_qutip` del repo, escrito en Gell-Mann.

    Por arista, la forma con momento angular

        Jz_i Jz_j - 2(Jz_i^2 + Jz_j^2) + 3 Jz_i^2 Jz_j^2

    equivale exactamente a (verificado numéricamente)

        l_3^(i) l_3^(j) + l_8^(i) l_8^(j) - (4/3) I

    o sea 2 términos de interacción por arista en vez de 4. El -4/3 por arista
    es una constante: no afecta al pool ni a la optimización, sólo desplaza
    el cero de energía — y se incluye para que las energías coincidan número a
    número con las del algoritmo original.
    """
    op = {}
    for (i, j) in edges:
        op = gm_suma(op, {
            ((i, 3), (j, 3)): 1.0,
            ((i, 8), (j, 8)): 1.0,
            (): -4.0 / 3.0,
        })
    return op


def Had_gm(n, edges, lam, omega0=1.0):
    """H_ad(lam) = (1-lam) H_i + lam H_p, con lam numérico."""
    return gm_suma(
        gm_escalar(1.0 - lam, Hi_gm(n, omega0)),
        gm_escalar(lam, Hp_gm(n, edges)),
    )


def dHad_dlam_gm(n, edges, omega0=1.0):
    """dH_ad/dlam = -H_i + H_p (independiente de lam)."""
    return gm_suma(gm_escalar(-1.0, Hi_gm(n, omega0)), Hp_gm(n, edges))


# ============================================================
# 5. Pool
# ============================================================

def build_pool_gm(n, edges, l=1, omega0=1.0, tol=1e-10, incluir_locales=False):
    """
    Pool contradiabático en base de Gell-Mann.

    Igual que en el algoritmo original, se toman los conmutadores anidados de
    H_ad con dH_ad/dlam y se usan los TÉRMINOS individuales como operadores
    del pool. l=1 usa O_1; l=2 usa O_1 U O_3.

    La diferencia con `utilidades.build_cd_pool`: cada término es un producto
    de lambdas sobre sitios distintos, que conmutan y son hermíticas, así que
    el operador ya es hermítico. No se aplica (O + O^dag)/2.

    El coeficiente de cada término del conmutador es imaginario puro (el
    conmutador de hermíticos es antihermítico); el operador del pool es el
    string en sí, que es hermítico.

    Nota sobre lambda: el conmutador [H_ad, dH_ad] depende de lam sólo por un
    factor global, porque H_ad = H_i + lam(H_p - H_i) y dH_ad = H_p - H_i, así
    que [H_ad, dH_ad] = [H_i, H_p - H_i] = [H_i, H_p]. El conjunto de strings
    no depende de lam.
    """
    # OJO con el valor de lambda. El código original mantiene lambda simbólico
    # (sympy), así que un término sólo desaparece si su coeficiente es cero
    # como polinomio. Acá se evalúa numéricamente, y en valores especiales de
    # lambda pueden cancelarse términos por accidente: con lam=0.5 el pool
    # l=2 pierde 24 strings respecto del valor genérico. Para evitarlo se usan
    # dos lambdas genéricos y se toma la unión, que reproduce el soporte del
    # polinomio salvo coincidencias de medida nula.
    dH = dHad_dlam_gm(n, edges, omega0)

    def strings_de(op):
        return {s for s, c in op.items() if abs(c) > tol and len(s) > 0}

    acum1, acum3 = set(), set()
    for lam_val in (0.3714159265, 0.6827182818):
        ordenes = gm_conmutadores_anidados(
            Had_gm(n, edges, lam=lam_val, omega0=omega0), dH, order=3
        )
        acum1 |= strings_de(ordenes[1])
        acum3 |= strings_de(ordenes[3])

    s1 = sorted(acum1, key=str)
    s3 = sorted(acum3, key=str)

    if l == 1:
        strings, ordenes_op = s1, [1] * len(s1)
    elif l == 2:
        vistos = set(s1)
        s3_nuevos = [s for s in s3 if s not in vistos]
        strings = s1 + s3_nuevos
        ordenes_op = [1] * len(s1) + [3] * len(s3_nuevos)
    else:
        raise ValueError("Solo se admiten l=1 o l=2.")

    # Con l=1 el pool de Gell-Mann resulta 100% de peso 2: H_p escrito en esta
    # base no tiene parte de un solo sitio (se absorbe en la constante -4/3),
    # así que el conmutador sólo genera acoplamientos de dos sitios. Esta
    # opción agrega las 8n rotaciones locales lambda_a^(j) para poder separar
    # dos causas posibles de que converja peor: la falta de operadores locales,
    # o que los strings sean elementos más atómicos que los monomios angulares
    # hermitizados.
    if incluir_locales:
        vistos = set(strings)
        locales = [((j, a),) for j in range(1, n + 1) for a in range(1, 9)]
        nuevos = [s for s in locales if s not in vistos]
        strings = strings + nuevos
        ordenes_op = ordenes_op + [0] * len(nuevos)

    return {
        "strings": strings,
        "ops": [string_a_qutip(n, s) for s in strings],
        "labels": [str(s) for s in strings],
        "orders": ordenes_op,
        "pesos": [peso(s) for s in strings],
    }
