"""
QAOA para qutrits en multiway number partitioning, a la escala de n = 12.

    |psi(gamma, beta)> = prod_{l=1}^{p} exp(-i beta_l H_M) exp(-i gamma_l H_C) |+3>^{otimes n}

con params = [gamma_1, ..., gamma_p, beta_1, ..., beta_p], la convención de
`funciones/utilidades_QAOA.py` (la del paper para Max 3-Cut).

ESCALA DE H_C
-------------
H_C es el costo de J. Molina, C(z) = sum_i (Sigma_i - mu)^2, diagonal, dividido
por su desviación estándar sobre la base computacional, sigma. Sin normalizar,
las energías son del orden de cientos a miles y el gamma útil es ~1/sigma; un
sorteo en [-pi, pi] cae casi siempre donde exp(-i gamma H_C) oscila demasiado
rápido, que es lo que le pasó al QAOA de J. Molina. Con H_C / sigma, gamma y
beta viven en la misma escala O(1), como en Max 3-Cut. Las energías que se
reportan se devuelven a la escala de J. Molina multiplicando por sigma.

MEZCLADOR
---------
H_M = sum_j G^(j), con G un operador de un qutrit:
    "jx": G = L_x, el del código del paper para Max 3-Cut y el de J. Molina;
    "x":  G = -(L_z^2 + sqrt(2) L_x), el de la ec. (9) del paper, cuyo estado
          fundamental es |+3>.

IMPLEMENTACIÓN
--------------
Nada de matrices de 3^n x 3^n: H_C es un vector y exp(-i beta H_M) es un
producto de compuertas de un qutrit. El
gradiente es analítico por el método adjunto y no guarda estados intermedios:
la pasada hacia atrás deshace cada capa (todas son unitarias), así que la
memoria es O(3^n) para cualquier p. El mezclador se aplica por grupos de 3
qutrits (U^{otimes 3}, de 27 x 27), con n/3 pasadas por el estado en vez de n:
a n = 12 el costo lo manda el tráfico de memoria.
"""

from __future__ import annotations

import time
from functools import reduce

import numpy as np
from scipy.optimize import minimize

D = 3
LX = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / np.sqrt(2)
LZ2 = np.diag([1.0, 0.0, 1.0]).astype(complex)
GENERADORES = {"jx": LX, "x": -(LZ2 + np.sqrt(2) * LX)}
GRUPO = 3           # sitios por grupo en el mezclador: U^{otimes 3} es de 27 x 27


def estado_uniforme(n):
    """|+3>^{otimes n}."""
    return np.full(D ** n, D ** (-n / 2.0), dtype=complex)


def _en_cada_sitio_por_sitio(pila, M, n):
    """(M^{otimes n}) sitio por sitio: n pasadas por el estado. Sólo para verificar."""
    m = pila.shape[0]
    for s in range(n):
        x = pila.reshape(m, D ** s, D, -1)
        pila = np.einsum("ij,majb->maib", M, x).reshape(m, -1)
    return pila


def _grupos(n, g=GRUPO):
    """[(primer sitio, tamaño)] de los grupos de a lo más g sitios consecutivos."""
    return [(s, min(g, n - s)) for s in range(0, n, g)]


def _en_cada_sitio(pila, M, n, g=GRUPO):
    """
    (M^{otimes n}) aplicado a cada estado de una pila de forma (m, 3^n).

    Se aplica M^{otimes t} a grupos de t <= g sitios consecutivos con un
    producto de matrices: n/g pasadas por el estado en vez de n. A n = 12 el
    costo está dominado por el tráfico de memoria, así que eso es lo que manda.
    """
    m = pila.shape[0]
    for s, t in _grupos(n, g):
        Mt = reduce(np.kron, [M] * t)
        A, B = D ** s, D ** (n - s - t)
        if B == 1:
            pila = (pila.reshape(m, A, D ** t) @ Mt.T).reshape(m, -1)
        else:
            pila = np.matmul(Mt, pila.reshape(m, A, D ** t, B)).reshape(m, -1)
    return pila


def _esperado_suma_local(lam, psi, G, n, g=GRUPO):
    """
    <lam| sum_j G^(j) |psi>, por grupos de sitios: para cada grupo se forma la
    matriz de transición reducida R_ab = sum_resto conj(lam_a) psi_b y se
    contrae con la suma de G sobre los sitios del grupo. Una lectura del estado
    por grupo, sin escribir nada del tamaño del estado.
    """
    total = 0j
    for s, t in _grupos(n, g):
        Hg = sum(reduce(np.kron, [G if k == j else np.eye(D) for k in range(t)]) for j in range(t))
        A, B = D ** s, D ** (n - s - t)
        X, Y = lam.reshape(A, D ** t, B), psi.reshape(A, D ** t, B)
        if A == 1:
            R = X[0].conj() @ Y[0].T
        elif B == 1:
            R = X[:, :, 0].conj().T @ Y[:, :, 0]
        else:
            R = np.matmul(X.conj(), Y.transpose(0, 2, 1)).sum(axis=0)
        total += np.sum(Hg * R)
    return total


def _suma_local(psi, G, n):
    """(sum_j G^(j)) |psi>."""
    fuera = np.zeros_like(psi)
    for s in range(n):
        x = psi.reshape(D ** s, D, -1)
        fuera.reshape(D ** s, D, -1)[...] += np.einsum("ij,ajb->aib", G, x)
    return fuera


def _unitario(w, V, angulo):
    """exp(-i angulo G) a partir de la base propia (w, V) de G."""
    return (V * np.exp(-1j * angulo * w)) @ V.conj().T


def evolucionar(params, h, p, mezclador="jx", psi0=None):
    """Estado QAOA de p capas. `h` es la diagonal de H_C en la escala en que viven los gamma."""
    n = round(np.log(len(h)) / np.log(D))
    w, V = np.linalg.eigh(GENERADORES[mezclador])
    psi = estado_uniforme(n) if psi0 is None else np.array(psi0, dtype=complex)
    for l in range(p):
        psi = np.exp(-1j * params[l] * h) * psi
        psi = _en_cada_sitio(psi[None], _unitario(w, V, params[p + l]), n)[0]
    return psi


def energia_y_grad(params, h, p, mezclador="jx"):
    """
    E(params) = <psi|H_C|psi> y su gradiente exacto.

    Con A_l el producto de las capas posteriores a una compuerta y
    |lambda> = A_l^dag H_C |psi>,
        dE/dbeta_l  = 2 Im <lambda| H_M |psi_l>,
        dE/dgamma_l = 2 Im <lambda| H_C |psi_l>,
    donde |psi_l> es el estado justo después de la compuerta. Hacia atrás se
    deshacen las capas sobre la pila [psi, lambda], sin guardar nada.
    """
    n = round(np.log(len(h)) / np.log(D))
    G = GENERADORES[mezclador]
    w, V = np.linalg.eigh(G)
    psi = evolucionar(params, h, p, mezclador)
    E = float(np.real(np.vdot(psi, h * psi)))

    grad = np.zeros(2 * p)
    pila = np.stack([psi, h * psi])
    for l in range(p - 1, -1, -1):
        grad[p + l] = 2.0 * np.imag(_esperado_suma_local(pila[1], pila[0], G, n))
        pila = _en_cada_sitio(pila, _unitario(w, V, -params[p + l]), n)
        grad[l] = 2.0 * np.imag(np.vdot(pila[1], h * pila[0]))
        pila = pila * np.exp(1j * params[l] * h)[None, :]
    return E, grad


def p_exito(psi, h):
    """Peso del estado sobre el subespacio fundamental de H_C."""
    return float(np.sum(np.abs(psi[np.isclose(h, h.min())]) ** 2))


def optimizar(h, p, x0, mezclador="jx", cota=np.pi, maxiter=10000, gtol=1e-8, ftol=1e-12):
    """
    L-BFGS-B con cotas [-cota, cota] en todos los parámetros, como el QAOA del
    paper, pero con gradiente analítico y sin tope bajo de iteraciones.
    """
    t0 = time.time()
    res = minimize(energia_y_grad, np.asarray(x0, dtype=float), args=(h, p, mezclador),
                   jac=True, method="L-BFGS-B", bounds=[(-cota, cota)] * (2 * p),
                   options={"maxiter": maxiter, "maxfun": 10 * maxiter, "ftol": ftol,
                            "gtol": gtol, "maxls": 50})
    return {"x": res.x, "E": float(res.fun), "nit": int(res.nit), "nfev": int(res.nfev),
            "exito": bool(res.success), "mensaje": str(res.message), "t_s": time.time() - t0}


def interp(x, p):
    """Parámetros óptimos de p capas -> punto inicial de p + 1 (INTERP, Zhou et al. 2020)."""
    def f(v):
        v = np.concatenate([[0.0], v, [0.0]])
        i = np.arange(1, p + 2)
        return (i - 1) / p * v[i - 1] + (p - i + 1) / p * v[i]
    return np.concatenate([f(np.asarray(x[:p])), f(np.asarray(x[p:]))])
