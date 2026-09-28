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
producto de compuertas de un qutrit, que se aplican sitio por sitio. El
gradiente es analítico por el método adjunto y no guarda estados intermedios:
la pasada hacia atrás deshace cada capa (todas son unitarias), así que la
memoria es O(3^n) para cualquier p. Energía y gradiente cuestan unas cuatro
veces la preparación del estado.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

D = 3
LX = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / np.sqrt(2)
LZ2 = np.diag([1.0, 0.0, 1.0]).astype(complex)
GENERADORES = {"jx": LX, "x": -(LZ2 + np.sqrt(2) * LX)}


def estado_uniforme(n):
    """|+3>^{otimes n}."""
    return np.full(D ** n, D ** (-n / 2.0), dtype=complex)


def _en_cada_sitio(pila, M, n):
    """(M^{otimes n}) aplicado a cada estado de una pila de forma (m, 3^n)."""
    m = pila.shape[0]
    for s in range(n):
        x = pila.reshape(m, D ** s, D, -1)
        pila = np.einsum("ij,majb->maib", M, x).reshape(m, -1)
    return pila


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
        grad[p + l] = 2.0 * np.imag(np.vdot(pila[1], _suma_local(pila[0], G, n)))
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
