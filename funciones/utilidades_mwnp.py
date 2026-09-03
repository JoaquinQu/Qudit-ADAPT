"""
Qudit-ADAPT for multiway number partitioning, at sizes the dense engine cannot reach.

Given m numbers a_1..a_m and k = 3 bins, assign each number to a bin so the bin
sums come out as even as possible. One number, one qutrit, three levels.

The cost Hamiltonian is the Max 3-Cut one with edge weights a_i a_j on the
complete graph:

    H_C = sum_{i<j} (a_i a_j / 2) [ Lz_i Lz_j - 2(Lz_i^2 + Lz_j^2)
                                    + 3 Lz_i^2 Lz_j^2 ]

and it relates to the partition objective exactly by

    sum_s S_s^2 = 2 E_C + (sum_i a_i)^2 ,

so minimizing the energy minimizes the spread of the bin sums. Since every pair
is coupled, the interaction graph is complete: the operator pool of an instance
on m numbers is the pool of K_m. The weights change the commutator
coefficients, not which operator strings appear, so the pool is reused as is.

WHY THIS MODULE EXISTS
----------------------
`utilidades_bp.py` is exact and readable but stores things densely, and two of
those hit a wall well before n = 14:

  * `prepare_problem` materializes H_C as a (3^n)^2 dense matrix and calls
    `eigvalsh` on it. At n = 10 that matrix alone is 52 GB. But H_C is diagonal
    in the computational basis, so the ground energy is just its minimum, and
    the reference state is the known product |+3>^{otimes n}. Neither needs a
    matrix at all.

  * `operator_spectrum` diagonalizes every ansatz operator densely and keeps V
    and V^dagger: 1.4 GB per operator at n = 8, 12.4 GB at n = 9.

Here every pool operator is treated as what it is -- a gate on at most four
sites. Its 3^w x 3^w block (81 x 81 at worst) is diagonalized once, and
exp(-i theta A) is applied by contracting that block against the state. Cost is
O(3^w * 3^n) per application with no storage beyond the state vector, and it is
exact for every operator, including the ones whose Hermitization does not
factorize into a tensor product.

Gradients are analytic throughout, both for the ADAPT operator selection and
for the BFGS inner loop (`jac=True`), computed by the adjoint method in a
single backward sweep.

    from funciones.utilidades_mwnp import adapt_mwnp
    res = adapt_mwnp([3, 5, 7, 11, 13, 2], l=1, max_iteration=20)
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

D = 3  # qutrits


# ==========================================================================
# 1. Cost Hamiltonian, as a diagonal
# ==========================================================================

def hamiltoniano_diag(a):
    """
    Diagonal de H_C, como vector de 3^m entradas.

    Se construye desde el objetivo de la partición y no desde los operadores:
        sum_s S_s^2 = 2 E_C + (sum a)^2   =>   E_C = (sum_s S_s^2 - (sum a)^2)/2
    Cuesta O(m * 3^m) y nunca toca una matriz.

    El nivel `c = 0, 1, 2` del qutrit i es el índice de su clase. En la
    convención de qutip el sitio 1 es el eje que varía más lento.
    """
    a = np.asarray(a, dtype=float)
    m = len(a)
    dim = D ** m

    # S[s] = suma de los a_i asignados a la clase s, para cada estado base.
    S = np.zeros((D, dim), dtype=float)
    idx = np.arange(dim, dtype=np.int64)
    for i in range(m):
        clase = (idx // (D ** (m - 1 - i))) % D      # nivel del sitio i+1
        for s in range(D):
            S[s] += a[i] * (clase == s)

    return 0.5 * (np.sum(S * S, axis=0) - a.sum() ** 2)


def energia_a_objetivo(E, a):
    """Convierte una energía de H_C al objetivo real sum_s S_s^2."""
    return 2.0 * E + float(np.sum(a)) ** 2


def particion_desde_indice(idx, m):
    """Asignación de clases (una por número) del estado base `idx`."""
    return [(idx // (D ** (m - 1 - i))) % D for i in range(m)]


# ==========================================================================
# 2. Pool: cada operador, como una compuerta de pocos sitios
# ==========================================================================

def _matriz_local(factores):
    """Producto de los factores de momento angular de un mismo sitio."""
    from funciones.utilidades import Jx1, Jy1, Jz1
    tabla = {"x": np.asarray(Jx1.full(), complex),
             "y": np.asarray(Jy1.full(), complex),
             "z": np.asarray(Jz1.full(), complex)}
    M = np.eye(D, dtype=complex)
    for eje in factores:
        M = M @ tabla[eje]
    return M


def bloque_local(label):
    """
    (sitios, bloque) de un operador del pool a partir de su etiqueta.

    El bloque es la matriz 3^w x 3^w del operador restringido a su soporte, ya
    hermitizada. Esto es exacto también cuando dos o más sitios llevan factor
    local no hermítico: ahí (O + O^dag)/2 no factoriza como producto tensorial,
    pero su restricción al soporte sigue siendo (B + B^dag)/2 con B el producto
    de los factores locales, que es justo lo que se calcula acá.
    """
    porsitio = {}
    for sitio, eje in ast.literal_eval(label):
        porsitio.setdefault(sitio, []).append(eje)

    sitios = sorted(porsitio)
    B = np.eye(1, dtype=complex)
    for s in sitios:
        B = np.kron(B, _matriz_local(porsitio[s]))

    return sitios, 0.5 * (B + B.conj().T)


CACHE_POOL = Path(__file__).resolve().parent.parent / "resultados" / "cache_pools"


def etiquetas_pool(n, l):
    """
    Etiquetas del pool contradiabático de K_n, con caché en disco.

    Dos diferencias con `utilidades_bp.obtener_pool`, y ambas importan al
    crecer n:

    1. Sólo se expanden los conmutadores anidados que el orden l realmente
       necesita. `build_cd_pool` pide siempre `order=3` y descarta O_2 y O_3
       cuando l = 1, que son justo los caros: cada conmutador multiplica el
       número de monomios. Con l = 1 basta O_1.

    2. El resultado se guarda en disco. La expansión es simbólica y no depende
       de los pesos a_i, sólo de (n, l), así que se paga una vez por tamaño y
       todas las instancias de ese n la reutilizan.
    """
    import sympy as sp

    CACHE_POOL.mkdir(parents=True, exist_ok=True)
    ruta = CACHE_POOL / f"kn_{n}_l{l}.json"
    if ruta.exists():
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)

    from funciones.utilidades import Had, dHad_dlam, canonical_op, nested_commutators

    edges = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
    lam = sp.symbols("lam", real=True)
    orden = 1 if l == 1 else 3
    res = nested_commutators(Had(n, edges, lam), dHad_dlam(n, edges), order=orden)

    ordenes = [1] if l == 1 else [1, 3]
    labels = []
    for k in ordenes:
        labels += sorted({str(canonical_op(op)) for op in res[k].keys()})

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    return labels


def preparar_pool(n, l, mostrar=False):
    """
    Pool contradiabático de K_n, con cada operador ya listo para aplicarse.

    Cada entrada trae los sitios (0-indexados) y la base propia del bloque de
    3^w x 3^w. Diagonalizar un bloque de a lo más 81x81 es instantáneo, y con
    (evals, V) la exponencial exp(-i theta A) sale sin exponenciar nada.
    """
    t0 = time.time()
    labels = etiquetas_pool(n, l)

    ops = []
    for label in labels:
        sitios, B = bloque_local(label)
        w, V = np.linalg.eigh(B)
        ops.append({"sitios": [s - 1 for s in sitios], "evals": w, "V": V,
                    "label": label, "peso": len(sitios)})

    if mostrar:
        print(f"pool K_{n}, l={l}: {len(ops)} operadores, "
              f"peso máx {max(o['peso'] for o in ops)}, {time.time()-t0:.1f} s")
    return ops


# ==========================================================================
# 3. Aplicar operadores al estado
# ==========================================================================

def _aplicar_bloque(psi, sitios, M, n):
    """
    Contrae la matriz M (3^w x 3^w) contra los ejes `sitios` del estado.

    El estado se ve como un tensor de n índices; se llevan los ejes objetivo al
    frente, se aplasta a (3^w, resto), se multiplica una sola vez con BLAS y se
    deshace la permutación.
    """
    w = len(sitios)
    psi = psi.reshape((D,) * n)
    psi = np.moveaxis(psi, sitios, range(w))
    forma = psi.shape
    psi = (M @ psi.reshape(D ** w, -1)).reshape(forma)
    return np.moveaxis(psi, range(w), sitios).reshape(-1)


def aplicar_operador(psi, op, n):
    """A|psi>, con A el operador hermítico del pool."""
    M = (op["V"] * op["evals"]) @ op["V"].conj().T
    return _aplicar_bloque(psi, op["sitios"], M, n)


def aplicar_exp(psi, op, theta, n, signo=-1.0):
    """exp(signo * i * theta * A)|psi>, exacto vía la base propia del bloque."""
    fase = np.exp(signo * 1j * theta * op["evals"])
    M = (op["V"] * fase) @ op["V"].conj().T
    return _aplicar_bloque(psi, op["sitios"], M, n)


def estado_referencia(n):
    """|+3>^{otimes n}: el fundamental del mixer, conocido analíticamente."""
    return np.full(D ** n, D ** (-n / 2.0), dtype=complex)


# ==========================================================================
# 4. Energía y gradiente analítico (el `jac` del optimizador)
# ==========================================================================

def energia_y_grad(params, ops, psi0, hdiag, n, guardar=None):
    """
    E(theta) y su gradiente exacto, en una pasada hacia adelante y una atrás.

    Con |psi> = U_k ... U_1 |psi_0> y U_j = exp(-i theta_j A_j),

        dE/dtheta_j = 2 Im <lambda_j| A_j |phi_j>,

    donde |phi_j> es el estado tras aplicar los primeros j operadores y
    |lambda_j> = U_{j+1}^dag ... U_k^dag H |psi>. La recursión hacia atrás
    reutiliza cada lambda, de modo que el gradiente completo cuesta lo mismo
    que dos evaluaciones de la energía, no k+1 como en diferencias finitas.
    """
    k = len(params)

    phis = [psi0]
    psi = psi0
    for j in range(k):
        psi = aplicar_exp(psi, ops[j], params[j], n)
        phis.append(psi)

    E = float(np.real(np.vdot(psi, hdiag * psi)))
    if guardar is not None:
        guardar["psi"] = psi

    grad = np.zeros(k)
    lam = hdiag * psi                       # H es diagonal: producto elemento a elemento
    for j in range(k - 1, -1, -1):
        grad[j] = 2.0 * np.imag(np.vdot(lam, aplicar_operador(phis[j + 1], ops[j], n)))
        lam = aplicar_exp(lam, ops[j], params[j], n, signo=+1.0)

    return E, grad


def gradientes_pool(psi, pool, hdiag, n):
    """
    Gradiente de ADAPT en theta = 0 para todo el pool: g_j = i<psi|[A_j, H]|psi>.

    Con H diagonal se reduce a g_j = -2 Im <A_j psi | H psi>, o sea UNA
    aplicación de operador por elemento del pool.
    """
    Hpsi = hdiag * psi
    return np.array([-2.0 * np.imag(np.vdot(aplicar_operador(psi, op, n), Hpsi))
                     for op in pool])


# ==========================================================================
# 5. El loop ADAPT
# ==========================================================================

def adapt_mwnp(a, l=1, epsilon=1e-2, max_iteration=30, maxiter=1000,
               pool=None, mostrar=True):
    """
    Qudit-ADAPT sobre una instancia de multiway number partitioning.

    `a` es la lista de números. Devuelve un dict con la traza de energía, los
    operadores elegidos, los parámetros óptimos y la partición leída del estado
    final.
    """
    a = np.asarray(a, dtype=float)
    n = len(a)
    t0 = time.time()

    hdiag = hamiltoniano_diag(a)
    E0 = float(hdiag.min())
    degeneracion = int(np.sum(np.isclose(hdiag, E0)))

    if pool is None:
        pool = preparar_pool(n, l, mostrar=mostrar)

    psi0 = estado_referencia(n)
    psi = psi0.copy()
    E_ini = float(np.real(np.vdot(psi, hdiag * psi)))

    ops, params = [], np.zeros(0)
    traza = [E_ini]
    indices, etiquetas = [], []
    razon = "max_iteration_reached"

    if mostrar:
        print(f"n = {n} números | pool = {len(pool)} | E0 = {E0:.6f} "
              f"(degeneración {degeneracion}) | E inicial = {E_ini:.6f}")

    for it in range(max_iteration):
        g = gradientes_pool(psi, pool, hdiag, n)
        norma = float(np.linalg.norm(g))
        if norma < epsilon:
            razon = "gradient_norm_below_epsilon"
            break

        j = int(np.argmax(np.abs(g)))
        ops.append(pool[j])
        indices.append(j)
        etiquetas.append(pool[j]["label"])

        x0 = np.concatenate([params, [0.0]])       # warm start
        guardar = {}
        res = minimize(energia_y_grad, x0,
                       args=(ops, psi0, hdiag, n, guardar),
                       jac=True, method="BFGS",
                       options={"maxiter": maxiter, "gtol": 1e-10})
        params = res.x
        E, _ = energia_y_grad(params, ops, psi0, hdiag, n, guardar)
        psi = guardar["psi"]
        traza.append(E)

        if mostrar:
            eps = abs(E - E0) / abs(E0) if E0 != 0 else abs(E - E0)
            print(f"  k={len(ops):3d}  |g|={norma:.3e}  E={E:.8f}  "
                  f"eps_rel={eps:.3e}  {pool[j]['label']}")

    idx_mejor = int(np.argmax(np.abs(psi) ** 2))
    clases = particion_desde_indice(idx_mejor, n)
    sumas = [float(a[[i for i in range(n) if clases[i] == s]].sum()) for s in range(D)]

    E_final = traza[-1]
    return {
        "n": n, "a": a.tolist(), "l": l,
        "pool_size": len(pool),
        "ground_energy": E0, "ground_degeneracy": degeneracion,
        "initial_energy": E_ini,
        "energy_trace": traza, "final_energy": E_final,
        "rel_error": abs(E_final - E0) / abs(E0) if E0 != 0 else abs(E_final - E0),
        "num_ansatz_ops": len(ops),
        "ansatz_op_indices": indices, "ansatz_op_labels": etiquetas,
        "params": params.tolist(),
        "stop_reason": razon,
        "mejor_probabilidad": float(np.abs(psi[idx_mejor]) ** 2),
        "particion": clases, "sumas": sumas,
        "objetivo": energia_a_objetivo(E_final, a),
        "objetivo_optimo": energia_a_objetivo(E0, a),
        "runtime_s": time.time() - t0,
    }


# ==========================================================================
# 6. Varianza del gradiente: el diagnóstico de barren plateau
# ==========================================================================

def varianza_gradiente(a, ops, n_muestras=100, seed=0, normalizar=True):
    """
    Var(dE/dtheta) en puntos de parámetros ALEATORIOS, no en el óptimo.

    Ésta es la medición que corresponde a un barren plateau: un paisaje
    exponencialmente plano visto por un optimizador que arranca sin
    información. Que el gradiente sea chico cerca del mínimo no dice nada.

    `normalizar` divide H_C por su rango espectral, y NO es opcional en la
    práctica: la escala de H_C crece como (sum a)^2, o sea ~n^2, así que sin
    normalizar la varianza sube con n por puro factor de escala y taparía
    cualquier decaimiento exponencial. Con el rango fijado a 1 la comparación
    entre tamaños es la que corresponde. Como H es diagonal, su máximo y su
    mínimo son exactos y gratis.

    `ops` es la lista de operadores del ansatz. Devuelve la varianza sobre
    todas las componentes y muestras, y la de la primera componente, que es la
    métrica habitual en la literatura de BP.
    """
    a = np.asarray(a, dtype=float)
    n = len(a)
    hdiag = hamiltoniano_diag(a)
    if normalizar:
        rango = float(hdiag.max() - hdiag.min())
        if rango > 0:
            hdiag = hdiag / rango
    psi0 = estado_referencia(n)
    k = len(ops)

    rng = np.random.default_rng(seed)
    grads = np.empty((n_muestras, k))
    for s in range(n_muestras):
        theta = rng.uniform(-np.pi, np.pi, size=k)
        _, grads[s] = energia_y_grad(theta, ops, psi0, hdiag, n)

    return {
        "n": n, "k": k, "n_muestras": n_muestras,
        "var_total": float(np.var(grads)),
        "var_primera": float(np.var(grads[:, 0])),
        "var_por_componente": np.var(grads, axis=0).tolist(),
        "abs_medio": float(np.mean(np.abs(grads))),
    }
