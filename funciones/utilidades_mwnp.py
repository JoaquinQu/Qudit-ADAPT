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
import os
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
    """
    Etiqueta de caja de cada número en el estado base `idx`, en la convención
    de Joaquín (formulacion_qubo.pdf, §3.1): la etiqueta es el autovalor del
    operador dígito d = J_z + I.

    Con J_z = diag(1, 0, -1), d = diag(2, 1, 0), así que el nivel k del qutrit
    lleva la etiqueta 2 - k. Es sólo una convención de lectura: el costo es
    simétrico bajo permutar etiquetas, de modo que energías, probabilidades y
    desbalances no cambian. Lo que cambia es qué número de caja se imprime, y
    adoptar la suya evita leer "caja 0" donde él lee "caja 2".
    """
    return [2 - (idx // (D ** (m - 1 - i))) % D for i in range(m)]


def hamiltoniano_joaquin(a):
    """
    H_p de Joaquín tal como está en formulacion_qubo.pdf, §2.4 y §3.3:

        C(z) = sum_i (Sigma_i(z) - mu)^2,    mu = (1/3) sum_j a_j,

    que vale exactamente 0 en una partición perfecta. Se implementa directo de
    la fórmula y no como transformación de `hamiltoniano_diag`, para que exista
    una versión de su escala que no dependa de nuestra derivación. Equivale a
    2 * hamiltoniano_diag(a) + (2/3) (sum a)^2, lo que se verifica en los tests.
    """
    a = np.asarray(a, dtype=float)
    m = len(a)
    idx = np.arange(D ** m, dtype=np.int64)
    Sigma = np.zeros((D, D ** m))
    for j in range(m):
        etiqueta = 2 - (idx // (D ** (m - 1 - j))) % D
        for i in range(D):
            Sigma[i] += a[j] * (etiqueta == i)
    mu = a.sum() / 3.0
    return np.sum((Sigma - mu) ** 2, axis=0)


def sumas_de_particion(a, clases):
    """Suma de cada una de las tres cajas."""
    a = np.asarray(a, dtype=float)
    return [float(a[[i for i in range(len(a)) if clases[i] == s]].sum())
            for s in range(D)]


def desbalance(sumas):
    """Diferencia entre la caja más llena y la más vacía."""
    return float(max(sumas) - min(sumas))


def leer_estado(psi, a, cuantos=8):
    """
    Traduce el estado final a particiones legibles.

    Devuelve las `cuantos` cadenas de trits más probables, cada una con su
    probabilidad, la partición que codifica, las sumas de las tres cajas y su
    desbalance. Es lo que uno realmente mediría: cada disparo del experimento
    colapsa el estado a una cadena, y esa cadena ES una asignación.
    """
    a = np.asarray(a, dtype=float)
    m = len(a)
    prob = np.abs(psi) ** 2
    orden = np.argsort(prob)[::-1][:cuantos]

    fuera = []
    for idx in orden:
        clases = particion_desde_indice(int(idx), m)
        s = sumas_de_particion(a, clases)
        fuera.append({
            "indice": int(idx),
            "trits": "".join(str(c) for c in clases),
            "probabilidad": float(prob[idx]),
            "clases": clases,
            "cajas": [[int(a[i]) for i in range(m) if clases[i] == c] for c in range(D)],
            "sumas": s,
            "desbalance": desbalance(s),
        })
    return fuera


def instancia_unica(n, indice, seed=0, rango=None, intentos_max=200000):
    """
    La instancia número `indice` de tamaño n, con partición perfecta ÚNICA.

    Criterios, todos verificados por enumeración exacta y no por construcción:
      * n enteros DISTINTOS en [1, rango], con rango = 3n por defecto;
      * existe un reparto en tres cajas de suma idéntica (desbalance 0);
      * ese reparto es único: degeneración exactamente 6, que es el mínimo
        posible porque cada partición aparece una vez por cada permutación de
        las tres etiquetas.

    Los números distintos importan: los repetidos inflan la degeneración y con
    ella la probabilidad de acertar por azar. El rango crece con n porque, fijo,
    a n = 10 ya no queda ninguna instancia de solución única y la dificultad
    dejaría de ser comparable entre tamaños.

    Es DETERMINISTA por (seed, n, indice): la instancia 7 es siempre la misma,
    se generen 10 o 20. Así se pueden agregar instancias sin tocar las previas.
    Con números distintos no hay partición perfecta para n < 5 (dos cajas de
    un solo elemento tendrían que ser iguales).
    """
    if n < 5:
        raise ValueError("con números distintos no existe partición perfecta para n < 5")
    rango = 3 * n if rango is None else rango
    rng = np.random.default_rng([seed, n, indice])

    for _ in range(intentos_max):
        a = sorted(rng.choice(np.arange(1, rango + 1), size=n, replace=False).tolist())
        if sum(a) % 3:
            continue
        h = hamiltoniano_diag(a)
        E0 = float(h.min())
        # desbalance 0  <=>  sum_s S_s^2 = S^2/3  <=>  2 E0 + S^2 = S^2/3
        if not np.isclose(energia_a_objetivo(E0, a), sum(a) ** 2 / 3):
            continue
        optimos = np.flatnonzero(np.isclose(h, E0))
        if len(optimos) != 6:
            continue
        clases = particion_desde_indice(int(optimos[0]), n)
        return {
            "n": n, "indice": indice, "seed": seed, "rango": rango,
            "a": [int(x) for x in a],
            "suma": int(sum(a)), "suma_por_caja": int(sum(a) // 3),
            "degeneracion": 6,
            "particion_optima": "".join(map(str, clases)),
            "cajas_optimas": [[int(a[i]) for i in range(n) if clases[i] == c]
                              for c in range(D)],
            "p_azar": 6.0 / D ** n,
        }
    raise RuntimeError(f"sin instancia única para n={n} tras {intentos_max} intentos")


def instancias_unicas(n, cuantas, seed=0, rango=None):
    """Las primeras `cuantas` instancias únicas de tamaño n, sin duplicados."""
    fuera, vistas = [], set()
    indice = 0
    while len(fuera) < cuantas:
        inst = instancia_unica(n, indice, seed=seed, rango=rango)
        clave = tuple(inst["a"])
        if clave not in vistas:
            vistas.add(clave)
            inst["id"] = len(fuera)
            fuera.append(inst)
        indice += 1
    return fuera


def fuerza_bruta(a):
    """
    Óptimo exacto por enumeración, para verificar. Sólo hasta ~n = 14: son
    3^n asignaciones, que a n = 13 son 1.6 millones y a n = 16 ya 43.
    """
    hdiag = hamiltoniano_diag(a)
    E0 = float(hdiag.min())
    indices = np.flatnonzero(np.isclose(hdiag, E0))
    m = len(a)
    clases = particion_desde_indice(int(indices[0]), m)
    return {
        "energia": E0,
        "objetivo": energia_a_objetivo(E0, a),
        "degeneracion": int(len(indices)),
        "sumas": sumas_de_particion(a, clases),
        "desbalance": desbalance(sumas_de_particion(a, clases)),
        "indices_optimos": indices,
    }


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

    # Escritura atómica: con varios procesos en paralelo, dos podrían pedir el
    # mismo pool a la vez. Se escribe a un temporal y se renombra, que en POSIX
    # es atómico, así nadie lee nunca un archivo a medio escribir.
    tmp = ruta.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    os.replace(tmp, ruta)
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
    reutiliza cada lambda, de modo que energía y gradiente completo cuestan unas
    3k aplicaciones de bloque (unas tres evaluaciones de la energía), no k+1
    evaluaciones como en diferencias finitas.
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
               pool=None, mostrar=True, checkpoint=None, inicializacion="warm"):
    """
    Qudit-ADAPT sobre una instancia de multiway number partitioning.

    `a` es la lista de números. Devuelve un dict con la traza de energía, los
    operadores elegidos, los parámetros óptimos y la partición leída del estado
    final.

    En cada iteración se registra `p_optimo`, el peso del estado sobre el
    subespacio fundamental, o sea la probabilidad de que un disparo del
    experimento entregue una partición óptima. Es la cifra que importa en
    optimización combinatoria, y no coincide con el error de energía: el fondo
    del espectro es exponencialmente denso, así que un error relativo chico
    puede convivir con probabilidad cero de medir la solución.

    Con `checkpoint` se vuelca el estado a ese archivo tras cada iteración, de
    modo que una corrida larga se pueda seguir mientras avanza y no se pierda
    si el proceso muere.
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

    # Mascara del subespacio fundamental: se calcula una vez y sirve para
    # medir en cada iteracion la probabilidad de acertar la particion optima.
    mask_fund = np.isclose(hdiag, E0)

    ops, params = [], np.zeros(0)
    traza = [E_ini]
    traza_p_optimo = [float(np.sum(np.abs(psi[mask_fund]) ** 2))]
    traza_desbalance = [leer_estado(psi, a, cuantos=1)[0]["desbalance"]]
    indices, etiquetas = [], []
    traza_norma, traza_seleccion = [], []
    traza_params, traza_bfgs, traza_tiempo = [], [], []
    razon = "max_iteration_reached"

    if mostrar:
        print(f"n = {n} números | pool = {len(pool)} | E0 = {E0:.6f} "
              f"(degeneración {degeneracion}) | E inicial = {E_ini:.6f}")

    for it in range(max_iteration):
        t_it = time.time()
        g = gradientes_pool(psi, pool, hdiag, n)
        t_barrido = time.time() - t_it
        norma = float(np.linalg.norm(g))
        traza_norma.append(norma)
        if norma < epsilon:
            razon = "gradient_norm_below_epsilon"
            break

        # Selección. Se registran también los empates: en instancias con
        # simetría varios operadores comparten el gradiente máximo hasta
        # redondeo, y cuál elige argmax depende sólo del orden del pool.
        ag = np.abs(g)
        j = int(np.argmax(ag))
        top = np.argsort(ag)[::-1][:20]
        traza_seleccion.append({
            "indice": j,
            "grad": float(g[j]),
            "empates": int(np.sum(np.isclose(ag, ag[j], rtol=1e-9, atol=1e-12))),
            "top20": [[int(i), float(ag[i])] for i in top],
        })
        ops.append(pool[j])
        indices.append(j)
        etiquetas.append(pool[j]["label"])

        # Punto de partida de BFGS en cada paso.
        #   warm: (theta*_{k-1}, 0), el óptimo anterior más el nuevo en cero.
        #         Es el ADAPT estándar, y por construcción E_{k+1}(x0) = E_k.
        #   cold: todo en cero. Con theta = 0 todas las exponenciales son la
        #         identidad, así que cada optimización arranca literalmente
        #         desde la superposición uniforme. Como el estado optimizado
        #         cambia, el barrido del paso siguiente elige operadores
        #         distintos: es otro algoritmo, no sólo otra inicialización.
        if inicializacion == "warm":
            x0 = np.concatenate([params, [0.0]])
        elif inicializacion == "cold":
            x0 = np.zeros(len(ops))
        else:
            raise ValueError("inicializacion debe ser 'warm' o 'cold'")
        guardar = {}
        t_opt = time.time()
        res = minimize(energia_y_grad, x0,
                       args=(ops, psi0, hdiag, n, guardar),
                       jac=True, method="BFGS",
                       options={"maxiter": maxiter, "gtol": 1e-10})
        params = res.x
        E, _ = energia_y_grad(params, ops, psi0, hdiag, n, guardar)
        psi = guardar["psi"]
        traza.append(E)
        traza_params.append(params.tolist())
        traza_bfgs.append({"nit": int(res.nit), "nfev": int(res.nfev),
                           "exito": bool(res.success),
                           "t_barrido": t_barrido, "t_bfgs": time.time() - t_opt})
        traza_tiempo.append(time.time() - t_it)

        # Probabilidad de que un disparo entregue una partición óptima.
        p_opt = float(np.sum(np.abs(psi[mask_fund]) ** 2))
        traza_p_optimo.append(p_opt)
        mejor_k = leer_estado(psi, a, cuantos=1)[0]
        traza_desbalance.append(mejor_k["desbalance"])

        if mostrar:
            eps = abs(E - E0) / abs(E0) if E0 != 0 else abs(E - E0)
            print(f"  k={len(ops):3d}  |g|={norma:.3e}  E={E:.6f}  eps={eps:.2e}  "
                  f"p_opt={p_opt:.4f}  desbal={mejor_k['desbalance']:.0f}  "
                  f"{pool[j]['label']}", flush=True)

        if checkpoint is not None:
            with open(checkpoint, "w", encoding="utf-8") as f:
                json.dump({
                    "a": list(map(float, a)), "n": n, "l": l,
                    "ground_energy": E0, "ground_degeneracy": degeneracion,
                    "num_ansatz_ops": len(ops),
                    "energy_trace": traza,
                    "p_optimo_trace": traza_p_optimo,
                    "desbalance_trace": traza_desbalance,
                    "ansatz_op_labels": etiquetas,
                    "params": list(map(float, params)),
                    "grad_norm": norma,
                    "runtime_s": time.time() - t0,
                    "en_progreso": True,
                }, f, indent=1)

    lectura = leer_estado(psi, a, cuantos=10)
    mejor = lectura[0]
    clases, sumas = mejor["clases"], mejor["sumas"]

    # Peso total que el estado pone sobre el subespacio fundamental: es la
    # probabilidad de que UN disparo del experimento entregue una partición
    # óptima, que para optimización combinatoria es la cifra que importa.
    hmin = float(hdiag.min())
    p_optimo = float(np.sum(np.abs(psi[np.isclose(hdiag, hmin)]) ** 2))

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
        "p_optimo_trace": traza_p_optimo,
        "desbalance_trace": traza_desbalance,
        "grad_norm_trace": traza_norma,
        "seleccion_trace": traza_seleccion,
        "params_trace": traza_params,
        "bfgs_trace": traza_bfgs,
        "tiempo_iter_trace": traza_tiempo,
        "mejor_probabilidad": mejor["probabilidad"],
        "prob_subespacio_optimo": p_optimo,
        "top_particiones": lectura,
        "particion": clases, "sumas": sumas,
        "desbalance": desbalance(sumas),
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


# ==========================================================================
# 7. Reoptimización con la secuencia de operadores fija
# ==========================================================================

def operadores_desde_etiquetas(etiquetas):
    """Operadores listos para aplicar a partir de sus etiquetas."""
    ops = []
    for label in etiquetas:
        sitios, B = bloque_local(label)
        w, V = np.linalg.eigh(B)
        ops.append({"sitios": [s - 1 for s in sitios], "evals": w, "V": V,
                    "label": label, "peso": len(sitios)})
    return ops


def reoptimizar_secuencia(a, etiquetas, maxiter=1000, cada_k=True):
    """
    Reoptimiza desde theta = 0 la MISMA secuencia de operadores que eligió otra
    corrida, tomando sus primeros k para cada k.

    La secuencia queda fija; lo único que cambia respecto del ADAPT original es
    el punto de partida de BFGS. Por eso aísla el efecto de la inicialización:
    si con el mismo circuito, partiendo de la superposición uniforme, se llega a
    la solución que el warm start perdió, el circuito era suficiente y el
    problema fue el camino de optimización. Es el experimento de las Figs. 5 y 6
    del manuscrito.

    Con cada_k=False se reoptimiza sólo el circuito completo, que es la
    reoptimización global final que se propuso tras la auditoría de la Fig. 3b.
    """
    a = np.asarray(a, dtype=float)
    n = len(a)
    hdiag = hamiltoniano_diag(a)
    E0 = float(hdiag.min())
    mask_fund = np.isclose(hdiag, E0)
    psi0 = estado_referencia(n)
    ops = operadores_desde_etiquetas(etiquetas)

    ks = range(1, len(ops) + 1) if cada_k else [len(ops)]
    traza = {"k": [], "energia": [], "p_exito": [], "desbalance_top": [],
             "parametros": [], "bfgs": []}
    for k in ks:
        guardar = {}
        res = minimize(energia_y_grad, np.zeros(k),
                       args=(ops[:k], psi0, hdiag, n, guardar),
                       jac=True, method="BFGS",
                       options={"maxiter": maxiter, "gtol": 1e-10})
        E, _ = energia_y_grad(res.x, ops[:k], psi0, hdiag, n, guardar)
        psi = guardar["psi"]
        traza["k"].append(k)
        traza["energia"].append(E)
        traza["p_exito"].append(float(np.sum(np.abs(psi[mask_fund]) ** 2)))
        traza["desbalance_top"].append(leer_estado(psi, a, cuantos=1)[0]["desbalance"])
        traza["parametros"].append(res.x.tolist())
        traza["bfgs"].append({"nit": int(res.nit), "nfev": int(res.nfev),
                              "exito": bool(res.success)})

    return {"ground_energy": E0, "traza": traza,
            "top_particiones": leer_estado(psi, a, cuantos=10)}
