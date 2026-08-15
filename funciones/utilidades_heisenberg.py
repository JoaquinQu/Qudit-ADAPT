"""
Pool contradiabático en la base de Heisenberg-Weyl (Pauli generalizadas)
=======================================================================

Tercera base para descomponer los conmutadores del pool CD, además de la de
momento angular (`utilidades.py`) y la de Gell-Mann (`utilidades_gellmann.py`).

Para qutrits se definen, con omega = exp(2 pi i / 3),

    X |k> = |k+1 mod 3>          (desplazamiento)
    Z |k> = omega^k |k>          (reloj)

y los nueve operadores X^a Z^b con a, b en {0,1,2} forman una base del espacio
de matrices 3x3. Satisfacen

    Z X = omega X Z   =>   (X^a Z^b)(X^c Z^d) = omega^{bc} X^{a+c} Z^{b+d}

Por qué es la comparación interesante
-------------------------------------
Las tres bases se distinguen en dos ejes independientes:

                    descomposición    elementos
                                      hermíticos
    momento angular    gruesa            no  -> hay que hermitizar
    Gell-Mann          fina              sí  -> no hace falta
    Heisenberg-Weyl    fina              no  -> hay que hermitizar

O sea, HW aísla la variable: si el pool de Gell-Mann converge peor porque sus
elementos son *atómicos* (y no por la hermiticidad), HW debería comportarse
como Gell-Mann. Si en cambio se parece al angular, lo que manda es otra cosa.

Ventaja técnica: los strings de HW forman un grupo salvo fase, así que el
producto de dos strings es UN solo string (no una suma, como en Gell-Mann).
El álgebra es más barata.

Representación
--------------
Un operador es un dict {string: coeficiente}, con `string` una tupla de tríos
(sitio, a, b), ordenada por sitio, sitios distintos y (a,b) != (0,0). La tupla
vacía es la identidad.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import qutip as qt

D = 3
OMEGA = np.exp(2j * np.pi / D)


def _X():
    M = np.zeros((D, D), dtype=complex)
    for k in range(D):
        M[(k + 1) % D, k] = 1.0
    return M


def _Z():
    return np.diag([OMEGA ** k for k in range(D)]).astype(complex)


X_MAT, Z_MAT = _X(), _Z()


def hw_matriz(a, b):
    """X^a Z^b como matriz 3x3."""
    return np.linalg.matrix_power(X_MAT, a % D) @ np.linalg.matrix_power(Z_MAT, b % D)


# ============================================================
# Álgebra de strings
# ============================================================

def _norm(pares):
    return tuple(sorted((s, a % D, b % D) for (s, a, b) in pares if (a % D, b % D) != (0, 0)))


def hw_limpiar(op, tol=1e-12):
    return {s: c for s, c in op.items() if abs(c) > tol}


def hw_suma(*ops):
    salida = {}
    for op in ops:
        for s, c in op.items():
            salida[s] = salida.get(s, 0j) + c
    return hw_limpiar(salida)


def hw_escalar(alpha, op):
    if alpha == 0:
        return {}
    return hw_limpiar({s: alpha * c for s, c in op.items()})


def _mult_strings(s1, s2):
    """
    Producto de dos strings -> (string, fase). Es UN solo string: los X^aZ^b
    forman un grupo salvo fase.

        (X^a Z^b)(X^c Z^d) = omega^{bc} X^{a+c} Z^{b+d}
    """
    d1 = {s: (a, b) for s, a, b in s1}
    d2 = {s: (a, b) for s, a, b in s2}

    fase = 1.0 + 0j
    pares = []

    for sitio in set(d1) | set(d2):
        a1, b1 = d1.get(sitio, (0, 0))
        a2, b2 = d2.get(sitio, (0, 0))
        fase *= OMEGA ** ((b1 * a2) % D)
        pares.append((sitio, (a1 + a2) % D, (b1 + b2) % D))

    return _norm(pares), fase


def hw_producto(A, B):
    salida = {}
    for s1, c1 in A.items():
        for s2, c2 in B.items():
            s, fase = _mult_strings(s1, s2)
            salida[s] = salida.get(s, 0j) + c1 * c2 * fase
    return hw_limpiar(salida)


def hw_conmutador(A, B):
    return hw_suma(hw_producto(A, B), hw_escalar(-1.0, hw_producto(B, A)))


def hw_conmutadores_anidados(H, dH, order=3):
    salida = {}
    actual = dH
    for k in range(1, order + 1):
        actual = hw_conmutador(H, actual)
        salida[k] = actual
    return salida


# ============================================================
# Conversión y descomposición
# ============================================================

def string_a_qutip(n, s):
    d = {sitio: (a, b) for sitio, a, b in s}
    ops = []
    for j in range(1, n + 1):
        if j in d:
            a, b = d[j]
            ops.append(qt.Qobj(hw_matriz(a, b)))
        else:
            ops.append(qt.qeye(D))
    return qt.tensor(ops)


def hw_a_qutip(n, op):
    total = 0
    for s, c in op.items():
        total = total + complex(c) * string_a_qutip(n, s)
    if total == 0:
        total = qt.tensor([qt.qeye(D) for _ in range(n)])
    return total


def descomponer_local(M, tol=1e-12):
    """
    Descompone una matriz 3x3 en la base {X^a Z^b}:

        M = sum_{a,b} c_{ab} X^a Z^b,     c_{ab} = Tr[(X^a Z^b)^dag M] / 3

    (la base es ortogonal con Tr[(X^aZ^b)^dag X^cZ^d] = 3 delta).
    """
    salida = {}
    for a in range(D):
        for b in range(D):
            P = hw_matriz(a, b)
            c = np.trace(P.conj().T @ M) / D
            if abs(c) > tol:
                salida[(a, b)] = complex(c)
    return salida


def _op_de_local(j, local):
    """dict{(a,b): coef} en el sitio j -> operador en representación de strings."""
    salida = {}
    for (a, b), c in local.items():
        s = _norm([(j, a, b)])
        salida[s] = salida.get(s, 0j) + c
    return salida


# ============================================================
# Hamiltonianos
# ============================================================

def Hi_hw(n, omega0=1.0):
    """H_i = -omega0 sum_j X~_j, con X~ = Jz^2 + sqrt(2) Jx, en base HW."""
    Xt = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=complex)
    local = descomponer_local(-omega0 * Xt)
    op = {}
    for j in range(1, n + 1):
        op = hw_suma(op, _op_de_local(j, local))
    return op


def Hp_hw(n, edges):
    """
    H_p de Max-3-Cut en base HW, idéntico al `Hp_qutip` del repo.

    Se descompone Jz y Jz^2 localmente y se arman los productos por arista con
    el álgebra de strings, así que no hay que derivar la forma a mano.
    """
    Jz = np.diag([1.0, 0.0, -1.0]).astype(complex)
    loc_Jz = descomponer_local(Jz)
    loc_Jz2 = descomponer_local(Jz @ Jz)

    op = {}
    for (i, j) in edges:
        Zi, Zj = _op_de_local(i, loc_Jz), _op_de_local(j, loc_Jz)
        Z2i, Z2j = _op_de_local(i, loc_Jz2), _op_de_local(j, loc_Jz2)

        op = hw_suma(
            op,
            hw_producto(Zi, Zj),
            hw_escalar(-2.0, Z2i),
            hw_escalar(-2.0, Z2j),
            hw_escalar(3.0, hw_producto(Z2i, Z2j)),
        )
    return op


def Had_hw(n, edges, lam, omega0=1.0):
    return hw_suma(hw_escalar(1.0 - lam, Hi_hw(n, omega0)),
                   hw_escalar(lam, Hp_hw(n, edges)))


def dHad_dlam_hw(n, edges, omega0=1.0):
    return hw_suma(hw_escalar(-1.0, Hi_hw(n, omega0)), Hp_hw(n, edges))


# ============================================================
# Pool
# ============================================================

def build_pool_hw(n, edges, l=1, omega0=1.0, tol=1e-10):
    """
    Pool contradiabático en base de Heisenberg-Weyl.

    A diferencia de Gell-Mann, X^a Z^b NO es hermítico, así que cada término se
    hermitiza con (O + O^dag)/2 — igual que en el pool de momento angular. Los
    strings que sólo difieren por conjugación dan el mismo operador hermítico,
    así que se deduplican.

    Como en el resto, se evalúa en dos lambdas genéricos y se toma la unión,
    para no perder términos por cancelaciones accidentales.
    """
    dH = dHad_dlam_hw(n, edges, omega0)

    def strings_de(op):
        return {s for s, c in op.items() if abs(c) > tol and len(s) > 0}

    acum1, acum3 = set(), set()
    for lam_val in (0.3714159265, 0.6827182818):
        ordenes = hw_conmutadores_anidados(
            Had_hw(n, edges, lam=lam_val, omega0=omega0), dH, order=3
        )
        acum1 |= strings_de(ordenes[1])
        acum3 |= strings_de(ordenes[3])

    if l == 1:
        base_strings = sorted(acum1, key=str)
        ordenes_op = [1] * len(base_strings)
    elif l == 2:
        s1 = sorted(acum1, key=str)
        s3 = sorted(acum3 - acum1, key=str)
        base_strings = s1 + s3
        ordenes_op = [1] * len(s1) + [3] * len(s3)
    else:
        raise ValueError("Solo se admiten l=1 o l=2.")

    # hermitizar y deduplicar: X^aZ^b y su adjunto dan el mismo (O+O^dag)/2
    vistos = {}
    strings, ops, ordenes_out = [], [], []

    for s, orden in zip(base_strings, ordenes_op):
        O = string_a_qutip(n, s)
        G = (O + O.dag()) / 2

        clave = np.round(G.full(), 9).tobytes()
        if clave in vistos:
            continue
        if float(np.max(np.abs(G.full()))) < 1e-12:
            continue  # parte hermítica nula

        vistos[clave] = True
        strings.append(s)
        ops.append(G)
        ordenes_out.append(orden)

    return {
        "strings": strings,
        "ops": ops,
        "labels": [str(tuple((sitio, f"{a}{b}") for sitio, a, b in s)) for s in strings],
        "orders": ordenes_out,
        "pesos": [len(s) for s in strings],
    }
