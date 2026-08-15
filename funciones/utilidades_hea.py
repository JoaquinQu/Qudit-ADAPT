"""
Ansatz hardware-efficient para qutrits: el control positivo
===========================================================

Por qué hace falta
------------------
Medir que la varianza del gradiente del pool contradiabático NO decae
exponencialmente sólo es informativo si el diagnóstico es capaz de DETECTAR un
barren plateau cuando sí lo hay. Sin esa contraparte, una curva plana admite la
explicación trivial de que la métrica es insensible.

Este módulo construye un ansatz hardware-efficient (HEA) de qutrits que sí
debería sufrir barren plateaus al crecer la profundidad, para medirlo con
exactamente el mismo procedimiento y ponerlo en el mismo gráfico.

Es especialmente pertinente acá porque el propio paper cita a Friedrich,
de Souza Farias & Maziero, *Barren plateaus are amplified by the dimension of
qudits*, Quantum Mach. Intell. 7, 56 (2025), que sostiene que subir la
dimensión del qudit EMPEORA el problema. O sea, la afirmación de robustez va
contra una tendencia documentada, y por eso necesita el control.

Construcción
------------
Capa = rotaciones locales aleatorias en los tres subespacios de dos niveles de
cada qutrit, seguidas de una capa entrelazadora en escalera. Se parametriza con
generadores de Gell-Mann para que las rotaciones locales sean nativas del
hardware de iones atrapados (ver `utilidades_bp.conteo_compuertas`).

Con suficientes capas, el circuito se aproxima a un 2-design sobre SU(3^n) y la
varianza del gradiente debe caer como ~exp(-alpha n). Esa es la firma que se
quiere ver aparecer en el control y NO en el pool contradiabático.

Diseño del experimento
----------------------
La comparación tiene que ser a igualdad de condiciones:

  - mismo Hamiltoniano de costo H_C y mismo estado inicial |phi_g>
  - mismo estimador: Var sobre parámetros uniformes en [-pi, pi]
  - mismo número de parámetros en ambos ansatz

Lo único que cambia es de dónde salen los generadores: del pool
contradiabático, o de la construcción hardware-efficient.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.sparse as sps

from funciones.utilidades_gellmann import GELLMANN
from funciones.utilidades_bp import (
    energy_and_grad,
    obtener_pool,
    operator_spectrum,
    pool_to_sparse,
    prepare_problem,
)

# Generadores locales de una capa HEA. Se usan los seis de Gell-Mann que
# conectan pares de niveles (los "transicionales"), que son las rotaciones
# nativas del procesador de iones, más los dos diagonales para las fases.
_LOCALES = [1, 2, 3, 4, 5, 6, 7, 8]


def _op_local(n, sitio, a):
    """lambda_a actuando en `sitio`, embebido en n qutrits, como sparse."""
    M = sps.identity(1, format="csr", dtype=complex)
    for j in range(1, n + 1):
        bloque = sps.csr_matrix(GELLMANN[a]) if j == sitio else sps.identity(3, format="csr", dtype=complex)
        M = sps.kron(M, bloque, format="csr")
    return M


# Pares de niveles del qutrit, y las Gell-Mann que hacen de sigma_x y sigma_y
# dentro de cada par. Son exactamente las transiciones que direcciona el
# hardware de iones.
_SIGMA = {(0, 1): (1, 2), (0, 2): (4, 5), (1, 2): (6, 7)}


def _sigma_phi(niveles, phi):
    """sigma_phi^(i,j) = cos(phi) sigma_x^(i,j) + sin(phi) sigma_y^(i,j)."""
    ax, ay = _SIGMA[niveles]
    return np.cos(phi) * GELLMANN[ax] + np.sin(phi) * GELLMANN[ay]


def _op_ms(n, i, j, niveles=(0, 1), phi=0.0):
    """
    Generador de la compuerta Mølmer-Sørensen entre los qutrits i y j:

        G_MS = (sigma_phi (x) 1 + 1 (x) sigma_phi)^2 / 4,

    de modo que MS(theta) = exp(-i theta G_MS) con la definición de Ringbauer
    et al. Es el entrelazador nativo del procesador de iones atrapados.

    Detalle importante para este módulo: G_MS NO es diagonal. Un entrelazador
    diagonal como lambda_3 (x) lambda_3 conmuta con H_C —que también es
    diagonal— y entonces su gradiente se anula idénticamente, porque
    dE/dtheta = 2 Im<H psi|A|psi> es cero si H y A son diagonales reales. Con
    ese entrelazador el circuito sólo acumula fases, no mezcla, y nunca se
    acerca a un 2-design: sería un control positivo inservible.
    """
    sp_ = _sigma_phi(niveles, phi)

    def _emb(sitio, M3):
        M = sps.identity(1, format="csr", dtype=complex)
        for k in range(1, n + 1):
            bloque = (sps.csr_matrix(M3) if k == sitio
                      else sps.identity(3, format="csr", dtype=complex))
            M = sps.kron(M, bloque, format="csr")
        return M

    S = _emb(i, sp_) + _emb(j, sp_)
    return ((S @ S) / 4.0).tocsr()


def build_hea(n, n_capas, seed=0, niveles_ms=(0, 1)):
    """
    Generadores de un ansatz hardware-efficient de `n_capas` capas.

    Cada capa aporta:
      - una rotación local por qutrit, con el generador de Gell-Mann elegido al
        azar entre los ocho (esto es lo que hace al ansatz "genérico" y lo
        empuja hacia un 2-design al crecer la profundidad);
      - una escalera de compuertas MS sobre los pares vecinos (j, j+1), cada
        una con fase phi aleatoria.

    El generador local aleatorio y la fase aleatoria de cada MS son lo que
    empuja al circuito hacia un 2-design al crecer la profundidad.

    Retorna la lista de generadores hermíticos, en orden de aplicación.
    """
    rng = np.random.default_rng(seed)
    gens = []

    for _ in range(n_capas):
        for sitio in range(1, n + 1):
            a = int(rng.choice(_LOCALES))
            gens.append(_op_local(n, sitio, a))

        for j in range(1, n):
            phi = float(rng.uniform(0.0, 2 * np.pi))
            gens.append(_op_ms(n, j, j + 1, niveles=niveles_ms, phi=phi))

    return gens


def varianza_gradiente(generadores, prob, n_muestras=200, seed=0,
                       theta_range=(-np.pi, np.pi), indice=0):
    """
    Varianza de una componente del gradiente sobre parámetros aleatorios.

    Es el estimador estándar de barren plateau: se muestrea theta uniforme, se
    evalúa dE/dtheta_j exacto, y se mide su varianza. Un barren plateau se
    manifiesta como Var ~ exp(-alpha n).

    Se reporta la varianza de UNA componente fija (`indice`) y también la
    agrupada sobre todas, porque la definición formal es sobre una componente.
    """
    rng = np.random.default_rng(seed)
    m = len(generadores)

    spectra = [operator_spectrum(G) for G in generadores]
    psi0, Hf = prob["psi0"], prob["Hf_mat"]

    una, todas = [], []
    for _ in range(n_muestras):
        theta = rng.uniform(theta_range[0], theta_range[1], size=m)
        _, g = energy_and_grad(theta, spectra, generadores, psi0, Hf)
        una.append(g[min(indice, m - 1)])
        todas.append(g)

    una = np.asarray(una, dtype=float)
    todas = np.concatenate(todas)

    return {
        "var_componente": float(np.var(una)),
        "var_todas": float(np.var(todas)),
        "mean_abs": float(np.mean(np.abs(todas))),
        "n_params": int(m),
        "n_muestras": int(n_muestras),
    }


def escaneo_n(
    ns,
    familia="ciclo",
    n_capas=None,
    l=1,
    n_muestras=200,
    seed=0,
    show=True,
):
    """
    El experimento central: Var(dE/dtheta) contra número de qutrits, para el
    ansatz HEA y para el pool contradiabático, sobre el mismo problema.

    Para que la comparación sea justa ambos ansatz usan el MISMO número de
    parámetros en cada n: el que produce el HEA con `n_capas` capas.

    `n_capas=None` escala la profundidad con n (n_capas = n), que es el régimen
    donde se espera que el HEA se acerque a un 2-design.

    Retorna una lista de dicts, uno por n.
    """
    filas = []

    for n in ns:
        if familia == "ciclo":
            edges = [(1, 2)] if n == 2 else [(i, i % n + 1) for i in range(1, n + 1)]
        elif familia == "completo":
            edges = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
        else:
            raise ValueError("familia debe ser 'ciclo' o 'completo'")

        prob = prepare_problem(n, edges)
        capas = n_capas if n_capas is not None else n

        gens_hea = build_hea(n, capas, seed=seed)
        m = len(gens_hea)

        # mismo número de parámetros, tomados del pool contradiabático
        pool = obtener_pool(n, edges, l, base="angular")
        A_sparse, _ = pool_to_sparse(pool["ops"], prob["Hf_sparse"])
        pool["ops"] = None

        rng = np.random.default_rng(seed + 1)
        idx = rng.integers(0, len(A_sparse), size=m)
        gens_cd = [A_sparse[i] for i in idx]

        st_hea = varianza_gradiente(gens_hea, prob, n_muestras=n_muestras, seed=seed)
        st_cd = varianza_gradiente(gens_cd, prob, n_muestras=n_muestras, seed=seed)

        fila = {
            "n": int(n),
            "dim": int(3 ** n),
            "capas": int(capas),
            "n_params": int(m),
            "pool_size": int(len(A_sparse)),
            "var_hea": st_hea["var_componente"],
            "var_hea_todas": st_hea["var_todas"],
            "var_cd": st_cd["var_componente"],
            "var_cd_todas": st_cd["var_todas"],
        }
        filas.append(fila)

        if show:
            print(f"n={n} | dim={3**n:5d} | capas={capas} | params={m:3d} | "
                  f"Var_HEA={fila['var_hea_todas']:.4e} | "
                  f"Var_CD={fila['var_cd_todas']:.4e}", flush=True)

    return filas


def ajuste_exponencial(filas, clave="var_hea_todas"):
    """
    Ajusta Var ~ exp(-alpha n) y devuelve alpha. Un alpha claramente positivo
    es la firma de barren plateau; alpha ~ 0 indica decaimiento a lo más
    polinomial.
    """
    ns = np.array([f["n"] for f in filas], dtype=float)
    v = np.array([f[clave] for f in filas], dtype=float)
    buenos = v > 0
    if buenos.sum() < 2:
        return float("nan")
    pendiente = np.polyfit(ns[buenos], np.log(v[buenos]), 1)[0]
    return float(-pendiente)
