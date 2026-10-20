"""
Descomposición de una unitaria de un qudit en rotaciones de dos niveles.

Es el Algoritmo 1 del apéndice de Ringbauer et al., Nat. Phys. 18, 1053 (2022),
con una sola modificación, declarada explícitamente más abajo.

El algoritmo tiene dos fases:

  1. Eliminación tipo Givens del triángulo inferior, columna por columna. Cada
     elemento no nulo cuesta una rotación R(theta,phi). Al terminar, la unitaria
     es diagonal.

  2. Las fases diagonales que quedan NO son gratis. El algoritmo resuelve un
     sistema lineal por los gamma_i y realiza cada gamma_i no nulo con TRES
     pulsos físicos,
         R(pi/2,pi) . R(gamma_i,pi/2) . R(pi/2,0),
     sobre el par correspondiente. No hay compuerta Z virtual.

La fase 2 es la que un conteo ingenuo olvida: cobra 3 pulsos por cada fase
relativa independiente, de modo que un generador diagonal como lambda_3 cuesta
3 y no 0, y lambda_8 o L_z cuestan 6.

MODIFICACIÓN. El algoritmo publicado sólo usa rotaciones entre niveles
ADYACENTES: su fase 1 mezcla las filas r-1 y r, y su fase 2 usa pares (i,i+1).
Aquí se permite en cambio cualquier par (i,j), que es como la propia Ec. (1) de
esa referencia define la compuerta nativa R^(i,j)(theta,phi). La diferencia es
observable: exp(-i theta lambda_4) vive en el par (0,2) y cuesta un solo pulso
si se puede manejar esa transición directamente, frente a los tres que hacen
falta para rodearla pasando por el nivel 1.
"""

from functools import lru_cache

import numpy as np

D = 3


def rotacion(theta, phi, a, b, d=D):
    """Compuerta nativa R(theta,phi) = exp(-i theta sigma_phi / 2) sobre (a,b)."""
    M = np.eye(d, dtype=complex)
    M[a, a] = M[b, b] = np.cos(theta / 2)
    M[a, b] = -1j * np.exp(-1j * phi) * np.sin(theta / 2)
    M[b, a] = -1j * np.exp(+1j * phi) * np.sin(theta / 2)
    return M


def _anular(U, r, c, piv, d, tol):
    """Rotación sobre el par (piv,r) que anula el elemento (r,c) de U."""
    a, b = U[piv, c], U[r, c]
    theta = 2 * np.arctan2(abs(b), abs(a))
    phi = np.angle(b) - np.angle(a) - np.pi / 2
    G = rotacion(theta, phi, piv, r, d)
    if abs((G @ U)[r, c]) > tol:                      # rama a = 0
        G = rotacion(np.pi, phi, piv, r, d)
    return G


def descomponer(U, adyacentes=False, tol=1e-9, verificar=True):
    """
    Rotaciones de dos niveles en que se descompone la unitaria U.

    Retorna dict con 'fase1', 'fase2', 'total' y la lista de pares usados.
    Con adyacentes=True se reproduce el algoritmo publicado tal cual.
    """
    U = np.asarray(U, dtype=complex)
    d = U.shape[0]
    Ut = U.copy()
    pares = []

    # ---- fase 1 ----------------------------------------------------------
    for c in range(d - 1):
        for r in range(d - 1, c, -1):
            if abs(Ut[r, c]) <= tol:
                continue
            # El pivote es la fila r-1 en el algoritmo publicado (sólo pares
            # adyacentes) y la fila c si se admite cualquier par: así el
            # elemento se anula contra la diagonal en una sola rotación.
            piv = r - 1 if adyacentes else c
            G = _anular(Ut, r, c, piv, d, tol)
            Ut = G @ Ut
            pares.append((piv, r))
    n1 = len(pares)

    fuera = np.max(np.abs(Ut - np.diag(np.diag(Ut))))
    if fuera > 1e-7:
        raise RuntimeError(f"la fase 1 no diagonalizó (residuo {fuera:.2e})")

    # ---- fase 2 ----------------------------------------------------------
    # gamma_i son las fases relativas independientes; el sistema lineal es el
    # del algoritmo, con la fase global gamma como incógnita adicional.
    A = np.zeros((d, d))
    for i in range(d - 1):
        A[i, i] = 1.0
        A[i + 1, i] = -1.0
    A[:, d - 1] = 1.0
    gammas = 2 * np.linalg.solve(A, np.angle(np.diag(Ut)))[:d - 1]

    n2 = 0
    for i, g in enumerate(gammas):
        if abs(np.mod(g + np.pi, 2 * np.pi) - np.pi) > tol:
            n2 += 3                                    # los tres pulsos
            pares += [(i, i + 1)] * 3

    if verificar and not _reconstruye(U, adyacentes, tol):
        raise RuntimeError("la descomposición no reconstruye U")

    return {"fase1": int(n1), "fase2": int(n2), "total": int(n1 + n2),
            "pares": pares}


def _reconstruye(U, adyacentes, tol):
    """Rehace U a partir de las rotaciones de la fase 1 y la diagonal restante."""
    U = np.asarray(U, dtype=complex)
    d = U.shape[0]
    Ut = U.copy()
    inversas = []
    for c in range(d - 1):
        for r in range(d - 1, c, -1):
            if abs(Ut[r, c]) <= tol:
                continue
            piv = r - 1 if adyacentes else c
            G = _anular(Ut, r, c, piv, d, tol)
            Ut = G @ Ut
            inversas.append(G.conj().T)

    rec = np.eye(d, dtype=complex)
    for G in inversas:
        rec = rec @ G
    rec = rec @ Ut

    k = np.unravel_index(np.argmax(np.abs(U)), U.shape)
    return np.max(np.abs(rec - (rec[k] / U[k]) * U)) < 1e-8


@lru_cache(maxsize=4096)
def _coste_cacheado(clave, d, adyacentes):
    U = np.frombuffer(clave, dtype=complex).reshape(d, d)
    return descomponer(U, adyacentes=adyacentes, verificar=False)["total"]


def coste(U, adyacentes=False):
    """Número de rotaciones de dos niveles que cuesta la unitaria U."""
    U = np.ascontiguousarray(np.asarray(U, dtype=complex))
    return _coste_cacheado(U.tobytes(), U.shape[0], adyacentes)


@lru_cache(maxsize=4096)
def _coste_diag_cacheado(clave, d, adyacentes):
    from itertools import permutations

    M = np.frombuffer(clave, dtype=complex).reshape(d, d)

    # Un factor ya diagonal no necesita diagonalizarse.
    if np.max(np.abs(M - np.diag(np.diag(M)))) < 1e-10:
        return 0

    _, V = np.linalg.eigh(M)

    # V sólo está definida salvo permutación de columnas y una fase por
    # columna. `eigh` ordena los autovalores de menor a mayor, lo que puede
    # devolver una permutación gratuita de niveles y cobrar rotaciones que un
    # compilador no pagaría. Se recorren las d! ordenaciones —seis para un
    # qutrit— fijando además la fase de cada columna para que su elemento
    # dominante sea real positivo, y se toma la más barata.
    mejor = None
    for perm in permutations(range(d)):
        W = V[:, list(perm)].copy()
        for j in range(d):
            k = np.argmax(np.abs(W[:, j]))
            W[:, j] *= np.exp(-1j * np.angle(W[k, j]))
        # Sólo la fase 1. En la conjugación V M V^dag, la parte diagonal de V
        # conmuta con M —que ahí ya es diagonal— y se cancela contra su
        # adjunta, de modo que las fases de la fase 2 no se pagan nunca.
        c = descomponer(W, adyacentes=adyacentes, verificar=False)["fase1"]
        if mejor is None or c < mejor:
            mejor = c
    return mejor


def coste_diagonalizacion(M, adyacentes=False):
    """
    Rotaciones que cuesta la unitaria que diagonaliza el factor local M.

    Se cuenta sólo la fase 1 del algoritmo: en V M V^dag las fases diagonales
    de V se cancelan. Se minimiza además sobre la libertad de gauge de la
    diagonalización (permutación de autovectores y fase de cada uno), que es
    lo que haría un compilador.
    """
    M = np.ascontiguousarray(np.asarray(M, dtype=complex))
    return _coste_diag_cacheado(M.tobytes(), M.shape[0], adyacentes)
