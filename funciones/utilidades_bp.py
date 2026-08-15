"""
Robustez de Qudit-ADAPT-VQE frente a barren plateaus (BP)
=========================================================

Este módulo reproduce el experimento numérico que sustenta la afirmación del
paper ("ADAPT-VQE with a counterdiabatic operator pool is robust against
barren plateau effects") para Max-3-Cut sobre qutrits.

Idea del experimento (análogo qudit de la Fig. de Grimsley et al.,
npj Quantum Inf. 9, 19 (2023), "Adaptive, problem-tailored VQE mitigates
rough parameter landscapes and barren plateaus"):

Se corre el algoritmo CD-ADAPT-VQE tal cual está en `funciones/utilidades.py`.
Cada vez que ADAPT agrega un operador nuevo al ansatz (y por lo tanto un
parámetro nuevo), se re-optimiza el MISMO ansatz de tres formas distintas:

  1. **Reciclada** (`recycled`): warm start, es decir los parámetros óptimos de
     la iteración anterior más un 0.0 para el parámetro nuevo. Ésta es la
     estrategia del algoritmo real y la que define la trayectoria de ADAPT
     (qué operador se elige en cada paso).
  2. **Fría** (`cold`): todos los parámetros a 0. Como theta = 0 implica
     |psi> = |psi_0>, esto equivale a reinicializar el estado al estado de
     referencia en cada iteración — el análogo qudit de la curva "HF" de la
     figura original.
  3. **Aleatoria** (`random`): `n_random` reinicios independientes con
     theta ~ U(theta_range) en TODOS los parámetros. Se registra la historia
     completa de la función de coste (una entrada por evaluación del
     objetivo), no sólo el valor final.

La nube de valores visitados por los reinicios aleatorios es lo que se grafica
como dashes con gradiente de color: muestra el paisaje de optimización que
"ve" un ansatz de ese tamaño cuando NO se recicla información, mientras que la
curva reciclada baja monótonamente. Si ADAPT fuera sensible a BP, la curva
reciclada quedaría atrapada dentro de la nube.

Notas de implementación
-----------------------
- Los Hamiltonianos y el pool CD se construyen llamando a `funciones.utilidades`
  (mismas funciones que usa el algoritmo original), de modo que el pool es
  bit a bit el mismo que el de `cd_adapt_vqe_algorithm_profundo`.
- El álgebra pesada se hace en numpy/scipy.sparse en vez de qutip, y la
  optimización usa **gradiente analítico** (retropropagación sobre el producto
  de exponenciales) en lugar de diferencias finitas. Esto es el mismo objetivo
  matemático, pero es ~k veces más barato y evita que el optimizador se
  detenga por ruido numérico — algo importante justamente en un estudio de
  barren plateaus, donde lo que se quiere medir son gradientes.
  `verificar_gradiente()` chequea el gradiente analítico contra diferencias
  finitas.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps
import sympy as sp
from scipy.optimize import minimize

from funciones.utilidades import (
    Had,
    Hi_qutip,
    Hp_qutip,
    canonical_op,
    dHad_dlam,
    monomial_to_qutip,
    nested_commutators,
    pool_to_qutip,
    to_jsonable,
)

RESULTADOS_DIR = PROJECT_ROOT / "resultados"
CSV_DIR = RESULTADOS_DIR / "csv"
JSON_DIR = RESULTADOS_DIR / "json"
IMAGES_DIR = RESULTADOS_DIR / "images"


# ============================================================
# 1. Construcción del problema y del pool CD
# ============================================================

def build_cd_pool(n, edges, l):
    """
    Reconstruye exactamente el pool de operadores contradiabáticos que usa
    `cd_adapt_vqe_algorithm_profundo`.

    O_k = conmutadores anidados [H_ad, [H_ad, ... [H_ad, dH_ad/dlam]]].
    l = 1 -> sólo O_1;  l = 2 -> O_1 U O_3.

    El orden del pool se fija con sorted(..., key=str) — misma regla que la
    versión profunda — para que los índices de operador sean reproducibles.
    (Verificado: este pool es idéntico, elemento a elemento, al que construye
    `cd_adapt_vqe_algorithm_profundo`. En grafos simétricos varios operadores
    empatan en el gradiente hasta ~1e-15, así que `argmax` puede elegir un
    índice distinto al del código original; los operadores son equivalentes
    por simetría y las trazas de energía coinciden.)

    Retorna
    -------
    dict con:
        "ops"    : lista de operadores hermíticos (qutip.Qobj)
        "labels" : etiquetas string de cada operador
        "orders" : orden del conmutador anidado (1 o 3) de cada operador
    """
    lam = sp.symbols("lam", real=True)

    H = Had(n, edges, lam)
    dH = dHad_dlam(n, edges)

    resultados = nested_commutators(H, dH, order=3)

    O1 = resultados[1]
    O3 = resultados[3]

    pool_unique_1 = sorted({canonical_op(op) for op in O1.keys()}, key=str)
    pool_unique_3 = sorted({canonical_op(op) for op in O3.keys()}, key=str)

    ops_1 = pool_to_qutip(pool_unique_1, n)
    ops_3 = pool_to_qutip(pool_unique_3, n)

    labels_1 = [str(op) for op in pool_unique_1]
    labels_3 = [str(op) for op in pool_unique_3]

    if l == 1:
        return {
            "ops": ops_1,
            "labels": labels_1,
            "orders": [1] * len(ops_1),
        }

    if l == 2:
        return {
            "ops": ops_1 + ops_3,
            "labels": labels_1 + labels_3,
            "orders": [1] * len(ops_1) + [3] * len(ops_3),
        }

    raise ValueError("Solo se admiten l=1 o l=2.")


def obtener_pool(n, edges, l, base="angular"):
    """
    Devuelve el pool en la base pedida, con la misma estructura en ambos casos
    ('ops', 'labels', 'orders').

    base="angular"  : monomios de momento angular, hermitizados con
                      (O + O^dag)/2. Es el pool del algoritmo original.
    base="gellmann" : strings de matrices de Gell-Mann, hermíticos por
                      construcción (ver funciones/utilidades_gellmann.py).

    Los Hamiltonianos son los MISMOS en ambos casos; lo único que cambia es en
    qué base se descomponen los conmutadores, y por lo tanto qué términos
    individuales forman el pool.
    """
    if base == "angular":
        return build_cd_pool(n, edges, l)

    if base == "gellmann":
        from funciones.utilidades_gellmann import build_pool_gm
        return build_pool_gm(n, edges, l)

    if base == "gellmann_local":
        # Gell-Mann + las 8n rotaciones de un solo sitio. Sirve para separar
        # dos causas del peor desempeño: la ausencia de operadores locales, o
        # que los strings sean más atómicos que los monomios hermitizados.
        from funciones.utilidades_gellmann import build_pool_gm
        return build_pool_gm(n, edges, l, incluir_locales=True)

    if base == "heisenberg":
        from funciones.utilidades_heisenberg import build_pool_hw
        return build_pool_hw(n, edges, l)

    raise ValueError(
        "base debe ser 'angular', 'gellmann', 'gellmann_local' o 'heisenberg'")


def peso_operador(label, base="angular"):
    """
    Número de sitios en los que actúa un operador del pool, a partir de su
    etiqueta. Sirve para estimar el costo de implementar exp(-i theta G).
    """
    import ast as _ast
    partes = _ast.literal_eval(label)
    return len({p[0] for p in partes})


def prepare_problem(n, edges):
    """
    Arma todo lo numérico que no depende del ansatz:

    - Hf = H_p (Max-3-Cut) como matriz densa y sparse
    - E0 = energía exacta del fundamental de Hf (referencia del error)
    - psi_0 = estado fundamental de H_i = -sum_j X_j (superposición uniforme)

    Se usa `Hi_qutip` / `Hp_qutip` del módulo original para garantizar que los
    Hamiltonianos son idénticos a los del algoritmo publicado.
    """
    Hf = Hp_qutip(n, edges)
    Hi = Hi_qutip(n, 1)

    Hf_mat = np.asarray(Hf.full(), dtype=complex)
    Hf_mat = 0.5 * (Hf_mat + Hf_mat.conj().T)  # simetriza ruido numérico

    evals_f = np.linalg.eigvalsh(Hf_mat)
    E0 = float(evals_f[0])

    tol = 1e-10
    ground_degeneracy = int(np.sum(np.abs(evals_f - E0) < tol))
    first_excited = float(evals_f[evals_f > E0 + tol][0]) if np.any(evals_f > E0 + tol) else None
    spectral_gap = (first_excited - E0) if first_excited is not None else None

    Hi_mat = np.asarray(Hi.full(), dtype=complex)
    Hi_mat = 0.5 * (Hi_mat + Hi_mat.conj().T)
    _, evecs_i = np.linalg.eigh(Hi_mat)
    psi0 = np.ascontiguousarray(evecs_i[:, 0].astype(complex))
    psi0 /= np.linalg.norm(psi0)

    return {
        "Hf_mat": Hf_mat,
        "Hf_sparse": sps.csr_matrix(Hf_mat),
        "E0": E0,
        "evals_f": evals_f,
        "first_excited_energy": first_excited,
        "spectral_gap": spectral_gap,
        "ground_degeneracy": ground_degeneracy,
        "psi0": psi0,
        "E_inicial": float(np.real(np.vdot(psi0, Hf_mat @ psi0))),
    }


def pool_to_sparse(pool_ops, Hf_sparse):
    """
    Convierte el pool a scipy.sparse y precomputa los conmutadores
    [H_f, A_j] que definen el gradiente de ADAPT.

    Mantener todo sparse es lo que hace viable un pool de cientos de
    operadores en dimensión 3^n sin reventar la memoria.
    """
    A_sparse = []
    comms = []

    for A in pool_ops:
        A_sp = sps.csr_matrix(np.asarray(A.full(), dtype=complex))
        A_sparse.append(A_sp)
        comms.append((Hf_sparse @ A_sp - A_sp @ Hf_sparse).tocsr())

    return A_sparse, comms


# ============================================================
# 2. Ansatz, energía y gradiente analítico
# ============================================================

def operator_spectrum(A_sparse):
    """
    Descomposición espectral de un operador hermítico del pool, para poder
    aplicar exp(-i theta A) sin exponenciar matrices dentro del optimizador
    (convención 4 del README).
    """
    A_dense = np.asarray(A_sparse.todense(), dtype=complex)
    A_dense = 0.5 * (A_dense + A_dense.conj().T)
    evals, V = np.linalg.eigh(A_dense)
    return (evals.astype(float), V, V.conj().T)


def apply_exp(theta, spectrum, vec, sign=-1.0):
    """
    Aplica exp(sign * i * theta * A) |vec>  usando la base espectral de A.
    sign = -1 -> U = exp(-i theta A)   (evolución del ansatz)
    sign = +1 -> U^dagger
    """
    evals, V, Vdag = spectrum
    coeffs = Vdag @ vec
    coeffs = coeffs * np.exp(sign * 1j * float(theta) * evals)
    return V @ coeffs


def build_state(params, spectra, psi0):
    """|psi(theta)> = exp(-i theta_m A_m) ... exp(-i theta_1 A_1) |psi_0>"""
    vec = psi0
    for theta, spec in zip(params, spectra):
        vec = apply_exp(theta, spec, vec, -1.0)
    return vec


def energy(params, spectra, psi0, Hf_mat):
    psi = build_state(params, spectra, psi0)
    return float(np.real(np.vdot(psi, Hf_mat @ psi)))


def energy_and_grad(params, spectra, A_ansatz, psi0, Hf_mat):
    """
    Energía y gradiente analítico exacto de

        E(theta) = <psi(theta)| H_f |psi(theta)>,
        |psi(theta)> = U_m ... U_1 |psi_0>,  U_j = exp(-i theta_j A_j).

    Derivación (retropropagación / método adjunto):

        d|psi>/d theta_j = R_j (-i A_j) |phi_j>,
            phi_j = U_j ... U_1 |psi_0>,   R_j = U_m ... U_{j+1}

        dE/d theta_j = 2 Re <H_f psi | R_j (-i A_j) | phi_j>
                     = 2 Im <sigma_j | A_j | phi_j>,
            sigma_j = R_j^dagger H_f |psi>,  sigma_m = H_f |psi>,
            sigma_{j-1} = U_j^dagger sigma_j

    Costo: O(m) productos matriz-vector para las m derivadas, en vez de
    O(m^2) que costaría el gradiente por diferencias finitas.
    """
    m = len(params)

    phis = [psi0]
    vec = psi0
    for theta, spec in zip(params, spectra):
        vec = apply_exp(theta, spec, vec, -1.0)
        phis.append(vec)

    psi = phis[-1]
    Hpsi = Hf_mat @ psi
    E = float(np.real(np.vdot(psi, Hpsi)))

    grad = np.empty(m, dtype=float)
    sigma = Hpsi

    for j in range(m - 1, -1, -1):
        grad[j] = 2.0 * float(np.imag(np.vdot(sigma, A_ansatz[j] @ phis[j + 1])))
        sigma = apply_exp(params[j], spectra[j], sigma, +1.0)

    return E, grad


def verificar_gradiente(params, spectra, A_ansatz, psi0, Hf_mat, h=1e-6):
    """
    Chequeo de sanidad: gradiente analítico vs diferencias finitas centradas.
    Retorna (grad_analitico, grad_numerico, error_maximo_absoluto).
    """
    params = np.asarray(params, dtype=float)
    _, g_ana = energy_and_grad(params, spectra, A_ansatz, psi0, Hf_mat)

    g_num = np.empty_like(params)
    for j in range(len(params)):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[j] += h
        p_minus[j] -= h
        g_num[j] = (
            energy(p_plus, spectra, psi0, Hf_mat)
            - energy(p_minus, spectra, psi0, Hf_mat)
        ) / (2 * h)

    return g_ana, g_num, float(np.max(np.abs(g_ana - g_num)))


class _CostRecorder:
    """
    Envuelve el objetivo para registrar TODOS los valores que visita la
    función de coste durante la optimización (una entrada por evaluación).

    Esto es exactamente "el historial de optimización" que después se grafica
    como la nube de dashes con gradiente de color.
    """

    def __init__(self, spectra, A_ansatz, psi0, Hf_mat):
        self.spectra = spectra
        self.A_ansatz = A_ansatz
        self.psi0 = psi0
        self.Hf_mat = Hf_mat
        self.history = []

    def __call__(self, params):
        E, g = energy_and_grad(params, self.spectra, self.A_ansatz, self.psi0, self.Hf_mat)
        self.history.append(E)
        return E, g


def optimizar(x0, spectra, A_ansatz, psi0, Hf_mat, maxiter=1000, gtol=1e-8):
    """
    Optimización BFGS con gradiente analítico, registrando la historia
    completa del coste. Retorna (params, energia, historia, info).
    """
    rec = _CostRecorder(spectra, A_ansatz, psi0, Hf_mat)

    res = minimize(
        rec,
        np.asarray(x0, dtype=float),
        jac=True,
        method="BFGS",
        options={"maxiter": maxiter, "gtol": gtol},
    )

    return (
        [float(p) for p in res.x],
        float(np.real(res.fun)),
        [float(e) for e in rec.history],
        {
            "success": bool(res.success),
            "status": int(res.status),
            "nit": int(getattr(res, "nit", -1)),
            "nfev": int(getattr(res, "nfev", -1)),
        },
    )


# ============================================================
# 3. Experimento principal
# ============================================================

def adapt_bp_scan(
    n,
    edges,
    l=1,
    epsilon=1e-2,
    max_iteration=20,
    n_random=100,
    theta_range=(-np.pi, np.pi),
    seed=0,
    maxiter=1000,
    show=True,
    store_history=False,
    n_jobs=None,
    checkpoint_path=None,
    base="angular",
    resume=True,
):
    """
    Corre CD-ADAPT-VQE y, en cada iteración (cada vez que se agrega un
    operador/parámetro nuevo), re-optimiza el mismo ansatz desde:

      - warm start reciclado   -> curva del algoritmo real
      - todos los thetas en 0  -> curva "fría" (análogo de la curva HF)
      - `n_random` puntos aleatorios uniformes en `theta_range`

    Parámetros
    ----------
    n : int              número de qutrits
    edges : list         aristas del grafo, sitios indexados desde 1
    l : int              1 -> pool O_1 ; 2 -> pool O_1 U O_3
    epsilon : float      corte por norma del gradiente (igual que el original)
    max_iteration : int  máximo de operadores en el ansatz
    n_random : int       reinicios aleatorios por iteración
    theta_range : tuple  rango uniforme de los parámetros aleatorios
    seed : int           semilla del generador aleatorio
    store_history : bool si True guarda todos los valores visitados por el
                         optimizador en cada reinicio; si False (por defecto)
                         guarda sólo el óptimo final de cada instancia, que es
                         lo que se grafica y evita JSONs enormes con n_random
                         grande.
    n_jobs : int         hilos para los reinicios (independientes entre sí).
                         numpy suelta el GIL dentro de BLAS, que es donde se va
                         casi todo el tiempo. Por defecto min(6, n_cpu).
    resume : bool        si hay checkpoint previo y coincide la configuración,
                         retoma desde ahí en vez de empezar de cero. El
                         resultado es idéntico a una corrida sin cortes,
                         porque también se repone el estado del generador
                         aleatorio.
    checkpoint_path      si se da, guarda el resultado parcial al final de cada
                         iteración de ADAPT. Un checkpoint tiene el mismo
                         esquema que un resultado completo (con
                         stop_reason="en_progreso"), así que se puede cargar y
                         graficar aunque la corrida se haya interrumpido.

    Retorna
    -------
    dict serializable con las curvas, las historias de coste y la metadata
    necesaria para reconstruir el ansatz.
    """
    t_start = time.time()
    rng = np.random.default_rng(seed)

    if n_jobs is None:
        n_jobs = min(6, os.cpu_count() or 1)
    workers = max(1, min(int(n_jobs), n_random))

    prob = prepare_problem(n, edges)
    Hf_mat = prob["Hf_mat"]
    psi0 = prob["psi0"]
    E0 = prob["E0"]

    pool = obtener_pool(n, edges, l, base=base)
    A_sparse, comms = pool_to_sparse(pool["ops"], prob["Hf_sparse"])
    pool["ops"] = None  # liberar los Qobj: con l=2 el pool tiene >1000 operadores

    if show:
        print(f"n = {n} | aristas = {len(edges)} | l = {l} | base = {base} | "
              f"pool = {len(A_sparse)} operadores")
        print(f"E0 (exacta) = {E0:.10f} | degeneración = {prob['ground_degeneracy']}")
        print(f"E inicial <psi_0|H_f|psi_0> = {prob['E_inicial']:.10f}")
        print("-" * 78)

    params = []
    spectra = []
    A_ansatz = []

    ansatz_indices = []
    ansatz_labels = []
    ansatz_orders = []

    # k = 0: el estado de referencia, sin ningún parámetro todavía
    recycled_energy = [prob["E_inicial"]]
    cold_energy = [prob["E_inicial"]]
    random_runs = [[]]

    max_grad_trace = []
    norm_grad_trace = []
    selected_grad = []
    stop_reason = "max_iteration_reached"

    # --------------------------------------------------------------
    # Reanudación desde checkpoint
    # --------------------------------------------------------------
    # Con n_random grande una corrida toma horas, y perderla entera por un
    # corte es caro. Si el checkpoint existe se restaura el estado completo y
    # el bucle sigue donde quedó.
    #
    # Para que el resultado sea idéntico a una corrida sin interrupciones hay
    # que reponer también el estado del generador aleatorio: se vuelven a
    # sortear (y descartar) exactamente los mismos x0 de las iteraciones ya
    # hechas, en el mismo orden.
    if resume and checkpoint_path is not None:
        ruta_chk = Path(checkpoint_path)
        if not ruta_chk.is_absolute():
            ruta_chk = JSON_DIR / ruta_chk

        if ruta_chk.exists():
            previo = load_bp_result(ruta_chk)
            hechas = int(previo["num_ansatz_ops"])

            coherente = (
                previo.get("base", "angular") == base
                and int(previo["l"]) == int(l)
                and int(previo["n_random"]) == int(n_random)
                and int(previo["seed"]) == int(seed)
                and len(previo["random_runs"]) == hechas + 1
                and len(previo["recycled_energy"]) == hechas + 1
            )

            if not coherente:
                raise ValueError(
                    f"El checkpoint {ruta_chk.name} no corresponde a esta "
                    f"configuración (base/l/n_random/seed) o está truncado. "
                    f"Borralo o usá otro nombre de salida."
                )

            if hechas >= max_iteration:
                if show:
                    print(f"El checkpoint ya tiene {hechas} operadores; nada que hacer.")
                return previo

            params = [float(p) for p in previo["params"]]
            ansatz_indices = [int(i) for i in previo["ansatz_op_indices"]]
            ansatz_labels = list(previo["ansatz_op_labels"])
            ansatz_orders = [int(o) for o in previo["ansatz_op_orders"]]

            A_ansatz = [A_sparse[i] for i in ansatz_indices]
            spectra = [operator_spectrum(A_sparse[i]) for i in ansatz_indices]

            recycled_energy = [float(e) for e in previo["recycled_energy"]]
            cold_energy = [float(e) for e in previo["cold_energy"]]
            random_runs = previo["random_runs"]

            max_grad_trace = [float(g) for g in previo["max_grad_trace"]]
            norm_grad_trace = [float(g) for g in previo["norm_grad_trace"]]
            selected_grad = [float(g) for g in previo["selected_gradient_trace"]]

            # reponer el estado del RNG descartando los sorteos ya usados
            for k_prev in range(1, hechas + 1):
                for _ in range(n_random):
                    rng.uniform(theta_range[0], theta_range[1], size=k_prev)

            if show:
                print(f"Reanudando desde {ruta_chk.name}: {hechas} operadores ya hechos, "
                      f"siguen hasta {max_iteration}.")
                print("-" * 78)

    def _construir(stop, fgn, fmg):
        """
        Arma el dict de resultado con el estado actual del barrido. Se usa
        tanto para los checkpoints de cada iteración como para el retorno
        final, así que un checkpoint tiene exactamente el mismo esquema que un
        resultado completo y se puede graficar tal cual.
        """
        runtime = float(time.time() - t_start)
        return {
            "n": int(n),
            "edges": [list(e) for e in edges],
            "num_edges": int(len(edges)),
            "l": int(l),
            "epsilon": float(epsilon),
            "max_iteration": int(max_iteration),
            "n_random": int(n_random),
            "theta_range": [float(theta_range[0]), float(theta_range[1])],
            "seed": int(seed),
            "store_history": bool(store_history),
            "n_jobs": int(workers),
            "base": str(base),

            "ground_energy": float(E0),
            "first_excited_energy": prob["first_excited_energy"],
            "spectral_gap": prob["spectral_gap"],
            "ground_degeneracy": int(prob["ground_degeneracy"]),
            "initial_problem_energy": float(prob["E_inicial"]),

            "pool_size": int(len(A_sparse)),
            "pool_labels": pool["labels"],
            "pool_orders": [int(o) for o in pool["orders"]],

            "num_ansatz_ops": int(len(params)),
            "stop_reason": stop,
            "params": [float(p) for p in params],
            "ansatz_op_indices": [int(i) for i in ansatz_indices],
            "ansatz_op_labels": list(ansatz_labels),
            "ansatz_op_orders": [int(o) for o in ansatz_orders],

            "recycled_energy": [float(e) for e in recycled_energy],
            "cold_energy": [float(e) for e in cold_energy],
            "random_runs": random_runs,

            "max_grad_trace": [float(g) for g in max_grad_trace],
            "norm_grad_trace": [float(g) for g in norm_grad_trace],
            "selected_gradient_trace": [float(g) for g in selected_grad],
            "final_gradient_norm": float(fgn),
            "final_max_gradient": float(fmg),

            "runtime_seconds": runtime,
            "runtime_min": runtime / 60.0,
            "ansatz_convention": (
                "psi(theta) = exp(-i theta_m A_m) ... exp(-i theta_1 A_1) psi_0"
            ),
        }

    while len(params) < max_iteration:
        psi = build_state(params, spectra, psi0) if params else psi0

        gradients = np.array(
            [abs(complex(np.vdot(psi, C @ psi))) for C in comms],
            dtype=float,
        )

        max_grad = float(np.max(gradients))
        norm = float(np.linalg.norm(gradients))

        max_grad_trace.append(max_grad)
        norm_grad_trace.append(norm)

        if norm < epsilon:
            stop_reason = "gradient_norm_below_epsilon"
            break

        idx = int(np.argmax(gradients))

        ansatz_indices.append(idx)
        ansatz_labels.append(pool["labels"][idx])
        ansatz_orders.append(pool["orders"][idx])
        selected_grad.append(float(gradients[idx]))

        A_ansatz.append(A_sparse[idx])
        spectra.append(operator_spectrum(A_sparse[idx]))

        k = len(A_ansatz)  # número de parámetros del ansatz actual

        # ---- 1. warm start reciclado: define la trayectoria real de ADAPT ----
        params, E_rec, _, info_rec = optimizar(
            params + [0.0], spectra, A_ansatz, psi0, Hf_mat, maxiter=maxiter
        )
        recycled_energy.append(E_rec)

        # ---- 2. cold start theta = 0 (análogo de la curva HF) ----
        _, E_cold, hist_cold, _ = optimizar(
            np.zeros(k), spectra, A_ansatz, psi0, Hf_mat, maxiter=maxiter
        )
        cold_energy.append(E_cold)

        # ---- 3. reinicios aleatorios uniformes ----
        # Los x0 se sortean acá, en orden, para que el resultado no dependa de
        # cómo el pool de hilos reparta el trabajo.
        x0_list = [
            rng.uniform(theta_range[0], theta_range[1], size=k)
            for _ in range(n_random)
        ]

        def _un_reinicio(x0):
            return optimizar(x0, spectra, A_ansatz, psi0, Hf_mat, maxiter=maxiter)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                salidas = list(ex.map(_un_reinicio, x0_list))
        else:
            salidas = [_un_reinicio(x0) for x0 in x0_list]

        runs_k = []
        for r, (x0, (_, E_rand, hist_rand, info_rand)) in enumerate(zip(x0_list, salidas)):
            run = {
                "restart": int(r),
                "x0": [float(v) for v in x0],
                "final_energy": float(E_rand),
                "nit": info_rand["nit"],
                "nfev": info_rand["nfev"],
            }
            if store_history:
                run["history"] = hist_rand
            runs_k.append(run)

        random_runs.append(runs_k)

        if show:
            cabecera = (
                f"k={k:2d} | reciclada={abs(E_rec - E0):.3e} | "
                f"fría={abs(E_cold - E0):.3e} | "
            )
            if runs_k:
                finales = np.abs(
                    np.array([run["final_energy"] for run in runs_k], float) - E0
                )
                cuerpo = (
                    f"aleatorias ({n_random}): mejor={finales.min():.3e} "
                    f"mediana={np.median(finales):.3e} peor={finales.max():.3e} | "
                    f"únicas={len(np.unique(np.round(finales, 8)))} | "
                )
            else:
                # n_random=0: sólo se calcula la curva del algoritmo
                cuerpo = "sin instancias aleatorias | "

            print(cabecera + cuerpo + f"t={(time.time()-t_start)/60:.1f}min")

        # Checkpoint: se guarda al terminar cada iteración, así una corrida
        # interrumpida conserva todo lo calculado hasta ahí.
        if checkpoint_path is not None:
            save_bp_result(
                _construir("en_progreso", norm_grad_trace[-1], max_grad_trace[-1]),
                checkpoint_path,
            )

    # gradiente final, ya con el ansatz completo
    psi_final = build_state(params, spectra, psi0) if params else psi0
    final_gradients = np.array(
        [abs(complex(np.vdot(psi_final, C @ psi_final))) for C in comms],
        dtype=float,
    )
    final_gradient_norm = float(np.linalg.norm(final_gradients))
    final_max_gradient = float(np.max(final_gradients))

    runtime = float(time.time() - t_start)

    if show:
        print("-" * 78)
        print(f"Operadores en el ansatz: {len(params)} | razón de parada: {stop_reason}")
        print(f"Error final (reciclada): {abs(recycled_energy[-1] - E0):.6e}")
        print(f"Tiempo total: {runtime / 60:.2f} min")

    resultado = _construir(stop_reason, final_gradient_norm, final_max_gradient)

    if checkpoint_path is not None:
        save_bp_result(resultado, checkpoint_path)

    return resultado


# ============================================================
# 4. Persistencia
# ============================================================

def save_bp_result(result, path):
    path = Path(path)
    if not path.is_absolute():
        path = JSON_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(result), f, indent=2, ensure_ascii=False)

    return path


def load_bp_result(path):
    path = Path(path)
    if not path.is_absolute():
        path = JSON_DIR / path

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 3b. Control positivo: ansatz hardware-efficient para qutrits
# ============================================================

def gell_mann():
    """
    Las 8 matrices de Gell-Mann: base de su(3), o sea de las rotaciones
    generales de un qutrit. Normalizadas como Tr(l_a l_b) = 2 delta_ab.
    """
    L = [
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, -2]],   # se normaliza abajo
    ]
    mats = [np.array(m, dtype=complex) for m in L]
    mats[7] = mats[7] / np.sqrt(3.0)
    return mats


def _embeber(op_local, n, sitio):
    """Mete una matriz 3x3 en el sitio dado de un registro de n qutrits."""
    ops = [np.eye(3, dtype=complex)] * n
    ops[sitio - 1] = op_local
    out = ops[0]
    for m in ops[1:]:
        out = np.kron(out, m)
    return out


def hea_generadores(n, depth, seed=0, acople="anillo"):
    """
    Ansatz hardware-efficient para qutrits, como CONTROL POSITIVO del
    diagnóstico de barren plateaus.

    Es el análogo qudit del circuito de McClean et al. (2018): capas de
    rotaciones locales aleatorias seguidas de entrelazamiento. Con
    profundidad creciente el ensemble se acerca a un 2-design, que es
    exactamente el régimen donde la teoría predice barren plateaus.

    Referencia para el caso qudit: Friedrich, de Souza Farias & Maziero,
    "Barren plateaus are amplified by the dimension of qudits"
    (arXiv:2405.08190), que reporta que subir la dimensión del qudit
    *empeora* el problema.

    Estructura de cada capa:
      - en cada sitio, una rotación exp(-i theta l_a) con l_a una matriz de
        Gell-Mann elegida al azar (rotación general de qutrit);
      - acoples de dos sitios exp(-i theta Jz_i Jz_j) sobre el anillo.

    Todo el circuito es un producto de exponenciales de un solo generador,
    así que sirve tal cual con `energy_and_grad`, `optimizar`, etc.

    Retorna (generadores_sparse, etiquetas).
    """
    rng = np.random.default_rng(seed)
    gm = gell_mann()

    if acople == "anillo":
        pares = [(i, i % n + 1) for i in range(1, n + 1)] if n > 2 else [(1, 2)]
    elif acople == "cadena":
        pares = [(i, i + 1) for i in range(1, n)]
    else:
        raise ValueError("acople debe ser 'anillo' o 'cadena'")

    Jz = np.array(qt_jz_local(), dtype=complex)

    gens, labels = [], []

    for capa in range(depth):
        for sitio in range(1, n + 1):
            a = int(rng.integers(0, 8))
            gens.append(sps.csr_matrix(_embeber(gm[a], n, sitio)))
            labels.append(f"capa{capa}:l{a+1}({sitio})")

        for (i, j) in pares:
            M = _embeber(Jz, n, i) @ _embeber(Jz, n, j)
            gens.append(sps.csr_matrix(M))
            labels.append(f"capa{capa}:Jz({i})Jz({j})")

    return gens, labels


def qt_jz_local():
    """J_z de spin-1 como matriz 3x3 (diag(1,0,-1))."""
    return np.diag([1.0, 0.0, -1.0]).astype(complex)


def varianza_gradiente(generadores, psi0, Hf_mat, n_muestras=200, seed=0,
                       theta_range=(-np.pi, np.pi), spectra=None):
    """
    Var_theta[ dE/dtheta_j ] sobre puntos de parámetros uniformemente
    aleatorios: el estimador estándar de barren plateau.

    Un barren plateau se manifiesta como esta varianza decayendo
    exponencialmente con n (o, a n fijo, hasta el valor de 2-design al crecer
    la profundidad).
    """
    rng = np.random.default_rng(seed)

    if spectra is None:
        spectra = [operator_spectrum(A) for A in generadores]

    k = len(generadores)
    todas = []

    for _ in range(n_muestras):
        theta = rng.uniform(theta_range[0], theta_range[1], size=k)
        _, g = energy_and_grad(theta, spectra, generadores, psi0, Hf_mat)
        todas.append(g)

    todas = np.asarray(todas)          # (n_muestras, k)

    return {
        "var_global": float(np.var(todas)),
        "var_por_parametro": np.var(todas, axis=0),
        "var_mediana": float(np.median(np.var(todas, axis=0))),
        "abs_medio": float(np.mean(np.abs(todas))),
        "k": int(k),
        "n_muestras": int(n_muestras),
    }


def test_2design(generadores, psi0, n_muestras=400, seed=0,
                 theta_range=(-np.pi, np.pi), spectra=None):
    """
    Cuán cerca está el ensemble de circuitos de ser un 2-design.

    Para estados Haar-aleatorios en dimensión D, la fidelidad
    F = |<psi_0|psi(theta)>|^2 sigue la distribución de Porter-Thomas, con

        E[F]   = 1/D
        Var[F] = (D-1) / (D^2 (D+1))

    Comparar los momentos empíricos contra esos valores da una medida
    operativa de qué tan aleatorizado está el circuito. Es lo que justifica
    llamar "control positivo" al HEA: si sus momentos coinciden con Haar,
    está en el régimen donde la teoría predice barren plateaus.
    """
    rng = np.random.default_rng(seed)

    if spectra is None:
        spectra = [operator_spectrum(A) for A in generadores]

    D = len(psi0)
    k = len(generadores)

    fids = []
    for _ in range(n_muestras):
        theta = rng.uniform(theta_range[0], theta_range[1], size=k)
        psi = build_state(theta, spectra, psi0)
        fids.append(float(np.abs(np.vdot(psi0, psi)) ** 2))

    fids = np.asarray(fids)

    media_haar = 1.0 / D
    var_haar = (D - 1) / (D ** 2 * (D + 1))

    return {
        "D": int(D),
        "media": float(fids.mean()),
        "media_haar": float(media_haar),
        "razon_media": float(fids.mean() / media_haar),
        "var": float(fids.var()),
        "var_haar": float(var_haar),
        "razon_var": float(fids.var() / var_haar),
        "fidelidades": fids,
    }


# ============================================================
# 4. Álgebra de Lie dinámica (DLA)
# ============================================================

def dla_dimension(generadores, tol=1e-9, max_dim=None, verbose=False):
    """
    Dimensión del álgebra de Lie dinámica generada por un conjunto de
    operadores hermíticos.

    Definición
    ----------
    Dados G_1, ..., G_M hermíticos, el DLA es la subálgebra de Lie real

        g = < i G_1, ..., i G_M >_Lie
          = span_R( {iG_j} U {[iG_a,iG_b]} U {[iG_a,[iG_b,iG_c]]} U ... )

    cerrada bajo conmutación. Es g subconjunto de u(D); si los G_j son de
    traza nula, g subconjunto de su(D), con dim su(3^n) = 9^n - 1.

    Por qué el cálculo es limpio
    ----------------------------
    1. Si A, B son antihermíticas, [A,B] también lo es: la clausura nunca se
       sale del espacio de matrices antihermíticas.
    2. Restringido a antihermíticas, el producto de Hilbert-Schmidt
       <A,B> = Tr(A^dag B) = -Tr(AB) es REAL, porque <A,B> = <B,A> = <A,B>*.
       Así que las antihermíticas forman un espacio vectorial REAL de
       dimensión D^2 con producto interno real, y Gram-Schmidt sobre R
       funciona sin más.

    El DLA depende sólo del CONJUNTO de generadores distintos: ni el orden ni
    las repeticiones importan (ADAPT repite operadores).

    Algoritmo
    ---------
    Clausura de Lie por Gram-Schmidt incremental: se ortonormaliza la base,
    se conmuta cada elemento nuevo contra todos los anteriores, y cada
    residuo con norma mayor que `tol` entra a la base. Termina cuando no
    entra nada nuevo.

    Parámetros
    ----------
    generadores : lista de matrices hermíticas (densas o sparse)
    max_dim : corta el cálculo si la dimensión lo supera. Útil porque si el
              álgebra es exponencial la base no cabe en memoria; en ese caso
              se retorna saturado=False y la dimensión es una cota inferior.

    Retorna
    -------
    dict con 'dim', 'dim_su', 'saturado' (True si cerró de verdad),
    'es_completa' (si alcanzó su(D)) y 'base'.
    """
    mats = []
    for G in generadores:
        if hasattr(G, "full"):          # qutip.Qobj
            A = np.asarray(G.full(), dtype=complex)
        elif sps.issparse(G):
            A = np.asarray(G.todense(), dtype=complex)
        else:
            A = np.asarray(G, dtype=complex)
        mats.append(1j * A)                      # hermítica -> antihermítica

    D = mats[0].shape[0]
    dim_u = D * D          # dim_R u(D): cota dura de la clausura
    dim_su = D * D - 1     # dim_R su(D)

    # Los operadores del pool NO son de traza nula (p.ej. Tr(Jz^2) != 0), así
    # que el álgebra puede contener la identidad y llegar hasta u(D).
    cap = dim_u if max_dim is None else min(int(max_dim), dim_u)

    # La base se guarda como un array 2D (m x D^2) con filas ortonormales
    # respecto del producto real. Proyectar es entonces un solo BLAS en vez de
    # un loop de Python sobre la base, que es lo que dominaba el costo.
    cap_alloc = min(cap, 512)
    B = np.zeros((cap_alloc, D * D), dtype=complex)
    m = 0

    def _agregar(M):
        """Ortogonaliza M contra la base; si sobra algo, lo agrega."""
        nonlocal B, m
        v = M.ravel().astype(complex, copy=True)

        # dos pasadas de Gram-Schmidt modificado: la segunda recupera la
        # ortogonalidad que la primera pierde por cancelación numérica
        for _ in range(2):
            if m:
                c = np.real(B[:m].conj() @ v)
                v -= c @ B[:m]

        nrm = float(np.linalg.norm(v))
        if nrm <= tol:
            return False

        if m == B.shape[0]:
            B = np.vstack([B, np.zeros_like(B)])

        B[m] = v / nrm
        m += 1
        return True

    def _fila(i):
        return B[i].reshape(D, D)

    # 1. los generadores
    pendientes = []
    for M in mats:
        if _agregar(M):
            pendientes.append(m - 1)

    # 2. clausura: conmutar cada elemento nuevo contra todos los anteriores
    saturado = True
    while pendientes:
        i = pendientes.pop(0)
        Ai = _fila(i)

        for j in range(m):
            Aj = _fila(j)
            C = Ai @ Aj - Aj @ Ai

            if np.linalg.norm(C) <= tol:
                continue

            if _agregar(C):
                pendientes.append(m - 1)

                if verbose and m % 100 == 0:
                    print(f"    dim parcial = {m}", flush=True)

                if m >= cap:
                    # Sólo es "no saturado" si cortamos por debajo del máximo
                    # posible; si llegamos a u(D) el álgebra sí cerró.
                    saturado = cap >= dim_u
                    pendientes.clear()
                    break

    return {
        "dim": int(m),
        "dim_u": int(dim_u),
        "dim_su": int(dim_su),
        "D": int(D),
        "saturado": bool(saturado),
        "es_completa": bool(saturado and m >= dim_su),
        "fraccion_u": m / dim_u,
        "base": B[:m],
    }


def dla_del_ansatz(result, tol=1e-9, max_dim=None, verbose=False):
    """
    DLA de los operadores que ADAPT efectivamente seleccionó en una corrida.

    Toma `ansatz_op_indices` del resultado, se queda con los índices
    DISTINTOS (el DLA no depende de repeticiones ni del orden), reconstruye
    esos operadores del pool y calcula la clausura de Lie.

    Ésta es el álgebra del circuito que se generó de verdad, que es distinta
    de la del pool completo: g_ansatz es subconjunto de g_pool.
    """
    n = result["n"]
    edges = [tuple(e) for e in result["edges"]]

    prob = prepare_problem(n, edges)
    pool = obtener_pool(n, edges, result["l"], base=result.get("base", "angular"))

    indices = sorted(set(int(i) for i in result["ansatz_op_indices"]))
    generadores = [pool["ops"][i] for i in indices]

    salida = dla_dimension(generadores, tol=tol, max_dim=max_dim, verbose=verbose)
    salida["num_generadores"] = len(indices)
    salida["indices"] = indices
    salida["labels"] = [pool["labels"][i] for i in indices]
    salida["n"] = int(n)

    return salida


def dla_del_pool(n, edges, l, tol=1e-9, max_dim=None, verbose=False):
    """
    DLA del pool contradiabático completo: cota superior de todo lo que ADAPT
    podría alcanzar con ese pool, independiente de la corrida.
    """
    prob = prepare_problem(n, edges)
    pool = build_cd_pool(n, edges, l)

    salida = dla_dimension(pool["ops"], tol=tol, max_dim=max_dim, verbose=verbose)
    salida["num_generadores"] = len(pool["ops"])
    salida["n"] = int(n)
    salida["l"] = int(l)

    return salida


# ============================================================
# 4a. Matriz de información cuántica de Fisher (QFIM)
# ============================================================

def qfim(params, spectra, A_ansatz, psi0):
    """
    Matriz de información cuántica de Fisher del ansatz, en el punto `params`.

        F_ij = 4 Re[ <d_i psi|d_j psi> - <d_i psi|psi><psi|d_j psi> ]

    OJO con qué objeto es esto: F es de tamaño k x k, con k el número de
    **parámetros variacionales** — no de dimensión 3^n, y no tiene que ver con
    la matriz del Hamiltoniano. Describe la geometría de la variedad que barre
    el ansatz: cuánto se mueve el estado cuando se mueven los parámetros.

    Su relevancia para barren plateaus:

    - Si los autovalores de F colapsan exponencialmente al crecer n, el estado
      deja de moverse al variar los parámetros y no hay señal que seguir.
    - El **rango** de F cuenta las direcciones independientes que el ansatz
      puede explorar. Satura cuando agregar parámetros deja de aportar
      direcciones nuevas (sobreparametrización), y esa saturación ocurre a la
      dimensión del álgebra de Lie dinámica.

    Las derivadas |d_j psi> = U_k...U_{j+1} (-i A_j) |phi_j> son las mismas
    piezas que usa el gradiente analítico.
    """
    k = len(params)
    dim = len(psi0)

    phis = [psi0]
    vec = psi0
    for theta, spec in zip(params, spectra):
        vec = apply_exp(theta, spec, vec, -1.0)
        phis.append(vec)

    psi = phis[-1]

    D = np.empty((dim, k), dtype=complex)
    for j in range(k):
        w = -1j * (A_ansatz[j] @ phis[j + 1])
        for m in range(j + 1, k):
            w = apply_exp(params[m], spectra[m], w, -1.0)
        D[:, j] = w

    b = D.conj().T @ psi                      # b_i = <d_i psi | psi>
    F = 4.0 * np.real(D.conj().T @ D - np.outer(b, b.conj()))

    return 0.5 * (F + F.T)                    # simetriza ruido numérico


def qfim_stats(F, rtol=1e-10):
    """Autovalores, rango efectivo y escala de una QFIM."""
    evals = np.linalg.eigvalsh(F)
    evals = np.clip(evals, 0.0, None)          # F es semidefinida positiva
    vmax = float(evals.max()) if len(evals) else 0.0
    rango = int(np.sum(evals > rtol * max(vmax, 1e-300)))

    return {
        "eigenvalues": evals,
        "rank": rango,
        "dim": int(F.shape[0]),
        "max": vmax,
        "mean": float(evals.mean()) if len(evals) else 0.0,
        "traza": float(np.trace(F)),
    }


def qfim_scan_aleatorio(n, edges, l, k, n_muestras=20, seed=0,
                        theta_range=(-np.pi, np.pi), indices=None):
    """
    Estadística de la QFIM en puntos de parámetros aleatorios, para un ansatz
    de k operadores del pool.

    `indices` fija qué operadores del pool forman el ansatz; si es None se
    eligen aleatoriamente. Para comparar contra ADAPT conviene pasarle los
    `ansatz_op_indices` de una corrida.

    Retorna medias sobre las muestras del rango, la traza y los autovalores.
    """
    rng = np.random.default_rng(seed)

    prob = prepare_problem(n, edges)
    pool = build_cd_pool(n, edges, l)
    A_sparse, _ = pool_to_sparse(pool["ops"], prob["Hf_sparse"])
    pool["ops"] = None

    if indices is None:
        indices = list(rng.integers(0, len(A_sparse), size=k))
    indices = list(indices)[:k]

    A_ansatz = [A_sparse[i] for i in indices]
    spectra = [operator_spectrum(A_sparse[i]) for i in indices]

    rangos, trazas, maximos, espectros = [], [], [], []

    for _ in range(n_muestras):
        theta = rng.uniform(theta_range[0], theta_range[1], size=len(indices))
        st = qfim_stats(qfim(theta, spectra, A_ansatz, prob["psi0"]))
        rangos.append(st["rank"])
        trazas.append(st["traza"])
        maximos.append(st["max"])
        espectros.append(st["eigenvalues"])

    return {
        "n": int(n),
        "l": int(l),
        "k": int(len(indices)),
        "indices": [int(i) for i in indices],
        "rank_medio": float(np.mean(rangos)),
        "rank_max": int(np.max(rangos)),
        "traza_media": float(np.mean(trazas)),
        "autovalor_max_medio": float(np.mean(maximos)),
        "espectro_medio": np.mean(np.array(espectros), axis=0),
    }


# ============================================================
# 4b. Decodificar la solución Max-3-Cut del estado final
# ============================================================

# Colores de las tres clases (paleta Okabe-Ito, segura para daltonismo)
COLORES_CLASE = ["#0072B2", "#D55E00", "#009E73"]


def reconstruir_estado_final(result):
    """
    Reconstruye |psi(theta_opt)> a partir de lo guardado en el resultado:
    regenera el pool, toma los operadores en el orden que eligió ADAPT y
    aplica los parámetros optimizados.
    """
    n = result["n"]
    edges = [tuple(e) for e in result["edges"]]

    prob = prepare_problem(n, edges)
    pool = obtener_pool(n, edges, result["l"], base=result.get("base", "angular"))
    A_sparse, _ = pool_to_sparse(pool["ops"], prob["Hf_sparse"])
    pool["ops"] = None

    spectra = [operator_spectrum(A_sparse[i]) for i in result["ansatz_op_indices"]]
    psi = build_state(result["params"], spectra, prob["psi0"])

    return psi, prob


def solucion_max3cut(result, psi=None, prob=None):
    """
    Decodifica la coloración Max-3-Cut que representa el estado final.

    El fundamental de H_C es degenerado (todas las coloraciones óptimas y sus
    permutaciones de color), así que el estado convergido es una superposición
    de muchas soluciones equivalentes. La coloración que se reporta es la del
    estado de la base computacional con mayor probabilidad — o sea, el
    resultado más probable de medir el circuito.

    Cada sitio queda en una de tres clases, que corresponden a los autoestados
    de J_z con autovalor +1, 0, -1.

    Con la convención de H_p, cada arista cortada aporta -2 y cada arista
    monocromática aporta 0, así que  |E_0| / 2 = número máximo de aristas
    cortadas. Eso permite verificar la solución decodificada.
    """
    if psi is None or prob is None:
        psi, prob = reconstruir_estado_final(result)

    n = result["n"]
    edges = [tuple(e) for e in result["edges"]]

    probs = np.abs(psi) ** 2
    idx = int(np.argmax(probs))

    # índice -> dígitos en base 3; el sitio 1 es el más significativo
    trits = []
    resto = idx
    for pos in range(n):
        div = 3 ** (n - 1 - pos)
        trits.append(resto // div)
        resto = resto % div

    # sitio (1..n) -> clase (0,1,2)
    coloreo = {sitio: int(trits[sitio - 1]) for sitio in range(1, n + 1)}

    cortadas = [(i, j) for i, j in edges if coloreo[i] != coloreo[j]]
    monocromaticas = [(i, j) for i, j in edges if coloreo[i] == coloreo[j]]

    max_corte = int(round(abs(float(result["ground_energy"])) / 2))

    return {
        "coloreo": coloreo,
        "indice_base": idx,
        "probabilidad": float(probs[idx]),
        "aristas_cortadas": cortadas,
        "aristas_monocromaticas": monocromaticas,
        "num_cortadas": len(cortadas),
        "max_corte": max_corte,
        "es_optima": len(cortadas) == max_corte,
        "energia_del_coloreo": -2.0 * len(cortadas),
    }


def layout_poligono(n):
    """
    Posiciones sobre un polígono regular: el sitio 1 arriba y los siguientes
    en sentido antihorario. Para n = 6 es un hexágono numerado 1..6 desde
    arriba, contra las agujas del reloj.

    Es preferible a `spring_layout` para las figuras del paper: es
    determinista, simétrico, y el mismo grafo se ve siempre igual.
    """
    return {
        j: (
            float(np.cos(np.pi / 2 + 2 * np.pi * (j - 1) / n)),
            float(np.sin(np.pi / 2 + 2 * np.pi * (j - 1) / n)),
        )
        for j in range(1, n + 1)
    }


def layout_por_clases(coloreo, radio_grupo=0.42, separacion=1.0):
    """
    Agrupa los vértices espacialmente según la clase que les asignó la
    solución: tres cúmulos separados, uno por clase.

    Es el layout que hace evidente qué significa una solución de Max-3-Cut.
    Como las clases deben ser conjuntos independientes, una solución óptima se
    ve como tres grupos **sin aristas internas**: todas las aristas van de un
    grupo a otro. Cada arista que quede dentro de un grupo es una arista sin
    cortar.
    """
    por_clase = {}
    for v, c in coloreo.items():
        por_clase.setdefault(c, []).append(v)

    pos = {}
    n_clases = max(3, len(por_clase))

    for c in sorted(por_clase):
        ang = np.pi / 2 + 2 * np.pi * c / n_clases
        cx, cy = separacion * np.cos(ang), separacion * np.sin(ang)

        miembros = sorted(por_clase[c])
        m = len(miembros)

        for t, v in enumerate(miembros):
            if m == 1:
                pos[v] = (float(cx), float(cy))
            else:
                a = 2 * np.pi * t / m + ang
                pos[v] = (
                    float(cx + radio_grupo * np.cos(a)),
                    float(cy + radio_grupo * np.sin(a)),
                )

    return pos


def plot_grafo_solucion(
    result,
    solucion=None,
    ax=None,
    seed=3,
    node_size=420,
    font_size=10,
    titulo=None,
    con_leyenda=False,
    layout="poligono",
):
    """
    Dibuja el grafo del problema con los vértices coloreados por la solución
    Max-3-Cut que encontró el algoritmo.

    Convención de colores:

    - **Vértices**: el color indica a cuál de las 3 clases lo asignó la
      solución. Una arista está *cortada* si sus dos extremos quedaron de
      colores distintos, que es lo que Max-3-Cut quiere maximizar.
    - **Aristas cortadas**: rojo sólido y grueso. Son las que el algoritmo
      logra cortar, o sea lo que cuenta para la función objetivo.
    - **Aristas monocromáticas**: gris punteado y fino. Sus extremos quedaron
      del mismo color, así que no aportan.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 4.0))

    if solucion is None:
        solucion = solucion_max3cut(result)

    edges = [tuple(e) for e in result["edges"]]
    G = nx.Graph()
    G.add_nodes_from(range(1, result["n"] + 1))
    G.add_edges_from(edges)

    if layout == "poligono":
        pos = layout_poligono(result["n"])
    elif layout == "clases":
        pos = layout_por_clases(solucion["coloreo"])
    elif layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    else:
        raise ValueError("layout debe ser 'poligono', 'clases' o 'spring'")

    # monocromáticas primero, para que las cortadas queden encima
    if solucion["aristas_monocromaticas"]:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=solucion["aristas_monocromaticas"],
            edge_color="#B0B0B0", width=1.2, style=(0, (3, 3)),
        )
    nx.draw_networkx_edges(
        G, pos, ax=ax, edgelist=solucion["aristas_cortadas"],
        edge_color="#CC0000", width=2.3,
    )

    colores = [COLORES_CLASE[solucion["coloreo"][v] % 3] for v in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colores, node_size=node_size,
        edgecolors="white", linewidths=1.5,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size,
                            font_color="white", font_weight="bold")

    if titulo is None:
        titulo = (
            f"{solucion['num_cortadas']} de {len(edges)} aristas cortadas"
        )
        if solucion["es_optima"]:
            titulo += "\n(el máximo posible)"
    ax.set_title(titulo, fontsize=9, fontweight="bold", pad=6)

    if con_leyenda:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([], [], color="#CC0000", lw=2.3, label="arista cortada"),
        ]
        if solucion["aristas_monocromaticas"]:
            handles.append(
                Line2D([], [], color="#B0B0B0", lw=1.2, ls=(0, (3, 3)),
                       label="sin cortar")
            )
        handles += [
            Line2D([], [], marker="o", ls="", markersize=7, color=c,
                   label=f"clase {k}")
            for k, c in enumerate(COLORES_CLASE)
        ]
        # debajo del grafo, para no pisar los vértices
        ax.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
            ncol=len(handles), fontsize=7.5, frameon=False, columnspacing=1.0,
            handletextpad=0.35,
        )

    # Se ocultan ejes y spines pero se deja el patch dibujable, para poder
    # darle fondo opaco cuando el grafo va como inset encima de los datos.
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.margins(0.14)

    return ax


# ============================================================
# 5. Forma explícita de las compuertas del ansatz
# ============================================================

def _factores_por_sitio(monomio):
    """
    Agrupa un monomio ((sitio, eje), ...) por sitio, preservando el orden
    canónico. Como el pool usa tuple(sorted(op)), dentro de cada sitio los
    ejes vienen ordenados alfabéticamente (x < y < z).
    """
    por_sitio = {}
    for sitio, eje in monomio:
        por_sitio.setdefault(sitio, []).append(eje)
    return por_sitio


def _potencias(ejes):
    """['z','z','y'] -> [('z',2), ('y',1)]"""
    salida = []
    for eje in ejes:
        if salida and salida[-1][0] == eje:
            salida[-1] = (eje, salida[-1][1] + 1)
        else:
            salida.append((eje, 1))
    return salida


def _render_sitio(sitio, ejes, latex=True):
    partes = []
    for eje, p in _potencias(ejes):
        if latex:
            base = rf"L_{eje}^{{({sitio})}}"
            partes.append(base if p == 1 else rf"\left({base}\right)^{{{p}}}")
        else:
            base = f"L{eje}({sitio})"
            partes.append(base if p == 1 else f"{base}^{p}")
    return (" " if latex else "·").join(partes)


def _render_producto(por_sitio, latex=True, invertir=False):
    trozos = []
    for sitio in sorted(por_sitio):
        ejes = por_sitio[sitio]
        trozos.append(_render_sitio(sitio, ejes[::-1] if invertir else ejes, latex))
    return (r"\," if latex else " · ").join(trozos)


def gate_expression(label, latex=True):
    """
    Traduce una etiqueta del pool a la forma explícita del generador hermítico
    G y de la compuerta exp(-i theta G).

    El pool guarda monomios de momento angular, p.ej. "((1, 'y'), (2, 'z'))",
    pero el operador que entra al ansatz es G = (O + O^dagger)/2 (ver
    `pool_to_qutip`). Hay dos casos:

    - Todos los factores de cada sitio son del mismo eje (o hay uno solo por
      sitio): O ya es hermítico — operadores en sitios distintos conmutan —
      así que G = O. Ejemplo: L_y^(1) L_z^(2).

    - Algún sitio tiene dos ejes distintos: ahí O no es hermítico, porque
      L_a y L_b del mismo sitio no conmutan. G queda simetrizado:
      G = ½{L_a^(s), L_b^(s)} (por el resto de los sitios).

    Retorna dict con 'G', 'gate', 'hermitico_directo' y 'sitios'.
    """
    import ast as _ast

    monomio = _ast.literal_eval(label)
    por_sitio = _factores_por_sitio(monomio)

    # O es hermítico si en cada sitio la lista de ejes es un palíndromo.
    # Como vienen ordenados, eso equivale a que sean todos iguales.
    directo = all(len(set(ejes)) == 1 for ejes in por_sitio.values())

    prod = _render_producto(por_sitio, latex=latex)

    if directo:
        G = prod
    else:
        # caso frecuente: un único sitio con exactamente dos ejes distintos
        conflictivos = [s for s, e in por_sitio.items() if len(set(e)) > 1]
        if len(conflictivos) == 1 and len(por_sitio[conflictivos[0]]) == 2:
            s = conflictivos[0]
            a, b = por_sitio[s]
            resto = {k: v for k, v in por_sitio.items() if k != s}

            if latex:
                anti = rf"\tfrac{{1}}{{2}}\left\{{L_{a}^{{({s})}},\, L_{b}^{{({s})}}\right\}}"
            else:
                anti = f"½{{L{a}({s}), L{b}({s})}}"

            if resto:
                G = anti + (r"\," if latex else " · ") + _render_producto(resto, latex=latex)
            else:
                G = anti
        else:
            rev = _render_producto(por_sitio, latex=latex, invertir=True)
            G = (rf"\tfrac{{1}}{{2}}\left({prod} + {rev}\right)" if latex
                 else f"½({prod} + {rev})")

    gate = rf"\exp\left(-i\,\theta\,{G}\right)" if latex else f"exp(-i·θ·[{G}])"

    return {
        "G": G,
        "gate": gate,
        "hermitico_directo": bool(directo),
        "sitios": sorted(por_sitio),
        "label": label,
    }


def ansatz_gates(result, latex=True):
    """
    Devuelve el ansatz final como lista ordenada de compuertas,

        |psi> = exp(-i theta_m G_m) ... exp(-i theta_1 G_1) |psi_0>,

    con la forma explícita de cada generador G_k en términos de los operadores
    de momento angular L_x, L_y, L_z actuando sobre cada sitio.
    """
    salida = []

    for paso, (idx, label, theta) in enumerate(
        zip(result["ansatz_op_indices"], result["ansatz_op_labels"], result["params"]),
        start=1,
    ):
        expr = gate_expression(label, latex=latex)
        salida.append({
            "step": int(paso),
            "operator_index": int(idx),
            "label": label,
            "theta": float(theta),
            "G": expr["G"],
            "gate": expr["gate"],
            "sitios": expr["sitios"],
            "hermitico_directo": expr["hermitico_directo"],
        })

    return salida


def _etiqueta_corta_cd(label):
    """
    De la etiqueta del pool a {sitio: string corto} para el diagrama.
    Refleja la simetrización: dos ejes distintos en el mismo sitio dan un
    anticonmutador, no un producto.
    """
    import ast as _ast

    por_sitio = _factores_por_sitio(_ast.literal_eval(label))
    salida = {}

    for sitio, ejes in por_sitio.items():
        if len(set(ejes)) == 1:
            eje = ejes[0]
            p = len(ejes)
            # para spin-1: L^3 = L, L^4 = L^2  (autovalores -1,0,1)
            if p >= 3:
                p = 1 if p % 2 else 2
            salida[sitio] = f"L{eje}" if p == 1 else f"L{eje}^{p}"
        elif len(ejes) == 2:
            a, b = ejes
            salida[sitio] = "{L%s,L%s}" % (a, b)
        else:
            salida[sitio] = "sym(" + "".join(ejes) + ")"

    return salida


def _etiqueta_corta_hea(label):
    """De 'capa0:l3(2)' o 'capa0:Jz(1)Jz(2)' a {sitio: string}."""
    import re

    cuerpo = label.split(":", 1)[1]

    if cuerpo.startswith("Jz"):
        sitios = [int(s) for s in re.findall(r"Jz\((\d+)\)", cuerpo)]
        return {s: "Jz" for s in sitios}

    m = re.match(r"l(\d+)\((\d+)\)", cuerpo)
    return {int(m.group(2)): f"λ{m.group(1)}"}


def circuito_ascii(pasos, n, thetas=None, max_pasos=None, ancho=None):
    """
    Diagrama ASCII del circuito: una fila por qutrit, una columna por
    compuerta, en orden de aplicación (izquierda a derecha).

    `pasos` es una lista de dicts {sitio: etiqueta_local}. Las compuertas de
    varios sitios se dibujan conectadas con '|' entre las filas que tocan.
    """
    if max_pasos is not None:
        pasos = pasos[:max_pasos]
        if thetas is not None:
            thetas = thetas[:max_pasos]

    if ancho is None:
        ancho = max(6, max((len(v) for p in pasos for v in p.values()), default=6) + 2)

    filas = {s: [] for s in range(1, n + 1)}
    cabecera = []

    for idx, paso in enumerate(pasos):
        involucrados = sorted(paso)
        lo, hi = (min(involucrados), max(involucrados)) if involucrados else (0, -1)

        cabecera.append(f"θ{idx+1}".center(ancho))

        for s in range(1, n + 1):
            if s in paso:
                celda = f"[{paso[s]}]".center(ancho, "─")
            elif lo < s < hi:
                celda = "│".center(ancho, "─")     # cruza una compuerta multi-sitio
            else:
                celda = "─" * ancho
            filas[s].append(celda)

    lineas = ["      " + "".join(cabecera)]
    for s in range(1, n + 1):
        lineas.append(f"q{s}: ──" + "".join(filas[s]) + "──")

    if thetas is not None:
        vals = "  ".join(f"θ{i+1}={t:+.4f}" for i, t in enumerate(thetas))
        lineas.append("")
        lineas.append("  " + vals)

    return "\n".join(lineas)


def circuito_ansatz_cd(result, max_pasos=None, con_thetas=False):
    """Diagrama ASCII del ansatz que generó CD-ADAPT-VQE."""
    pasos = [_etiqueta_corta_cd(lbl) for lbl in result["ansatz_op_labels"]]
    thetas = result["params"] if con_thetas else None
    return circuito_ascii(pasos, result["n"], thetas=thetas, max_pasos=max_pasos)


def circuito_hea(n, depth, seed=0, acople="anillo", max_pasos=None):
    """Diagrama ASCII del ansatz hardware-efficient (control positivo)."""
    _, labels = hea_generadores(n, depth, seed=seed, acople=acople)
    pasos = [_etiqueta_corta_hea(lbl) for lbl in labels]
    return circuito_ascii(pasos, n, max_pasos=max_pasos)


def ansatz_latex(result, max_pasos=None):
    """
    Ansatz completo como una ecuación LaTeX lista para pegar en el paper.
    """
    gates = ansatz_gates(result, latex=True)
    if max_pasos is not None:
        gates = gates[:max_pasos]

    factores = [rf"e^{{-i\theta_{{{g['step']}}} G_{{{g['step']}}}}}" for g in gates]
    producto = r"\,".join(reversed(factores))

    lineas = [rf"|\psi\rangle = {producto}\,|\psi_0\rangle", r"\\[0.6em]"]
    for g in gates:
        lineas.append(
            rf"G_{{{g['step']}}} &= {g['G']}, \qquad \theta_{{{g['step']}}} = {g['theta']:.6f} \\"
        )

    return "\n".join(lineas)


def verificar_gate_expressions(result, tol=1e-10):
    """
    Chequea que la expresión simbólica renderizada corresponda al operador
    numérico real del pool: reconstruye G = (O + O^dagger)/2 con qutip y lo
    compara contra el operador que efectivamente usó el ansatz.

    Retorna (n_verificados, error_maximo).
    """
    import ast as _ast

    n = result["n"]
    edges = [tuple(e) for e in result["edges"]]
    pool = build_cd_pool(n, edges, result["l"])

    err_max = 0.0
    contados = 0

    for idx, label in zip(result["ansatz_op_indices"], result["ansatz_op_labels"]):
        if pool["labels"][idx] != label:
            raise ValueError(
                f"Desajuste de etiqueta en el índice {idx}: "
                f"{pool['labels'][idx]!r} != {label!r}"
            )

        A_pool = pool["ops"][idx]

        # reconstrucción independiente a partir de la etiqueta
        monomio = _ast.literal_eval(label)
        O = monomial_to_qutip(n, monomio)
        G = (O + O.dag()) / 2

        err_max = max(err_max, float(np.max(np.abs(A_pool.full() - G.full()))))
        contados += 1

    return contados, err_max

def bp_error_cloud(result, floor=1e-14, mode="final", relativo=False):
    """
    Extrae la nube de errores de los reinicios aleatorios.

    mode="final" (por defecto)
        Un punto por instancia: el óptimo al que convergió. Es lo que grafica
        la figura del paper, y es lo informativo — cada dash es un mínimo
        (local o global) donde efectivamente terminó una optimización, así que
        la columna muestra la estructura de mínimos del paisaje.

    mode="history"
        Todos los valores que visitó la función de coste durante la
        optimización. Muestra el recorrido, no la estructura de mínimos.
        Requiere haber corrido con `store_history=True`.

    Retorna (k, err) planos.
    """
    if mode not in ("final", "history"):
        raise ValueError("mode debe ser 'final' o 'history'")

    E0 = float(result["ground_energy"])
    esc = abs(E0) if relativo else 1.0

    ks = []
    errs = []

    for k, runs_k in enumerate(result["random_runs"]):
        for run in runs_k:
            if mode == "final":
                ks.append(k)
                errs.append(abs(float(run["final_energy"]) - E0) / esc)
            else:
                if "history" not in run:
                    raise KeyError(
                        "Esta corrida se guardó sin historia de optimización "
                        "(store_history=False). Usá mode='final'."
                    )
                for E in run["history"]:
                    ks.append(k)
                    errs.append(abs(float(E) - E0) / esc)

    ks = np.asarray(ks, dtype=float)
    errs = np.clip(np.asarray(errs, dtype=float), floor, None)

    return ks, errs


def bp_error_curves(result, floor=1e-14, relativo=False):
    """Curvas de error de las estrategias reciclada y fría, y el mejor/peor aleatorio."""
    E0 = float(result["ground_energy"])
    esc = abs(E0) if relativo else 1.0

    recycled = np.clip(np.abs(np.asarray(result["recycled_energy"], float) - E0) / esc, floor, None)
    cold = np.clip(np.abs(np.asarray(result["cold_energy"], float) - E0) / esc, floor, None)

    best = []
    median = []
    worst = []
    for runs_k in result["random_runs"]:
        if not runs_k:
            best.append(np.nan)
            median.append(np.nan)
            worst.append(np.nan)
            continue
        finales = np.abs(np.array([r["final_energy"] for r in runs_k], float) - E0) / esc
        best.append(max(float(np.min(finales)), floor))
        median.append(max(float(np.median(finales)), floor))
        worst.append(max(float(np.max(finales)), floor))

    return {
        "k": np.arange(len(recycled)),
        "recycled": recycled,
        "cold": cold,
        "best_random": np.asarray(best),
        "median_random": np.asarray(median),
        "worst_random": np.asarray(worst),
    }


def plot_bp_landscape(
    result,
    ax=None,
    cmap="turbo",
    floor=1e-14,
    linthresh=1e-13,
    dash_size=110,
    dash_lw=1.4,
    dash_alpha=0.75,
    show_cold=True,
    title=None,
    colorbar=False,
    mode="final",
    jitter=0.13,
    seed=0,
    inset_grafo=True,
    inset_loc=(0.60, 0.58, 0.38, 0.38),
    inset_seed=3,
    relativo=False,
):
    """
    Reproduce, para Qudit-ADAPT-VQE sobre Max-3-Cut, la figura que muestra que
    ADAPT es insensible a paisajes rugosos y barren plateaus.

    - Cada dash horizontal es **el óptimo final** de una de las `n_random`
      instancias inicializadas con parámetros uniformemente aleatorios, para un
      ansatz de k parámetros (`mode="final"`). Es decir, cada dash es un mínimo
      del paisaje donde efectivamente terminó una optimización, así que la
      columna muestra la estructura de mínimos locales que ve el ansatz de ese
      tamaño. El color va en gradiente según la magnitud del error (redundante
      con la posición vertical: sirve para leer la densidad).
    - `jitter` desplaza horizontalmente los dashes dentro de su columna. En
      Max-3-Cut el espectro es discreto y muy degenerado, así que muchas
      instancias caen en **exactamente** el mismo mínimo; sin jitter esos
      dashes se superponen y una columna con 40 instancias en un mínimo se ve
      igual que una con 1. Poner `jitter=0` para alinearlos.
    - La línea verde es ADAPT tal como se ejecuta de verdad: parámetros
      reciclados de la iteración anterior más un 0.0 para el parámetro nuevo.
    - La línea roja reinicializa todos los parámetros a 0 en cada iteración
      (el estado vuelve al de referencia): el análogo qudit de la curva "HF".

    Retorna el eje de matplotlib.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.colors import LogNorm

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 5.4))

    ks, errs = bp_error_cloud(result, floor=floor, mode=mode, relativo=relativo)
    curvas = bp_error_curves(result, floor=floor, relativo=relativo)

    # --- nube de reinicios aleatorios -----------------------------------
    if len(errs) > 0:
        vmin = max(float(np.min(errs)), floor)
        vmax = float(np.max(errs))
        if vmax <= vmin:
            vmax = vmin * 10

        ks_plot = ks
        if jitter:
            despl = np.random.default_rng(seed).uniform(-jitter, jitter, size=len(ks))
            ks_plot = ks + despl

        sc = ax.scatter(
            ks_plot,
            errs,
            c=errs,
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            marker="_",
            s=dash_size,
            linewidths=dash_lw,
            alpha=dash_alpha,
            zorder=2,
            rasterized=True,
        )

        if colorbar:
            cb = ax.figure.colorbar(sc, ax=ax, pad=0.02)
            cb.set_label(r"$|E-E_0|$ visitado")

    # --- curvas de referencia -------------------------------------------
    k = curvas["k"]

    # La curva fría suele coincidir exactamente con la reciclada mientras el
    # ansatz es chico. Se dibuja más gruesa y punteada, por debajo, para que se
    # vea asomar por los costados donde las dos se superponen.
    if show_cold:
        ax.plot(
            k, curvas["cold"],
            color="#D62728", lw=3.2, ls=(0, (4, 2.5)),
            label=r"Reinicio frío ($\theta=0$)", zorder=3,
        )

    ax.plot(
        k, curvas["recycled"],
        color="#2CA02C", lw=1.8, marker="o", ms=3.2,
        label="ADAPT (parámetros reciclados)", zorder=4,
    )

    # --- ejes -------------------------------------------------------------
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlabel("Número de parámetros")
    ax.set_ylabel(r"$\epsilon_{\rm rel} = |E-E_0|/|E_0|$" if relativo
                  else r"Error respecto a la energía exacta  $|E-E_0|$")

    ax.set_xlim(-0.6, float(k[-1]) + 0.6)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if title is None:
        title = (
            f"Qudit-ADAPT-VQE — Max-3-Cut, $n={result['n']}$ qutrits, "
            f"{result['num_edges']} aristas, $\\ell={result['l']}$"
        )
    ax.set_title(title)

    ax.legend(loc="lower left", framealpha=0.92)
    ax.grid(True, which="major", alpha=0.35)

    # --- inset: el grafo con su solución ---------------------------------
    if inset_grafo:
        try:
            sol = solucion_max3cut(result)
            ax_in = ax.inset_axes(inset_loc, zorder=6)
            plot_grafo_solucion(result, solucion=sol, ax=ax_in, seed=inset_seed,
                                node_size=250, font_size=8)
            # fondo opaco: el inset va encima de la nube de dashes
            ax_in.patch.set_visible(True)
            ax_in.patch.set_facecolor("white")
            ax_in.patch.set_alpha(0.93)
            ax_in.patch.set_edgecolor("#CCCCCC")
            ax_in.patch.set_linewidth(0.8)
        except Exception as exc:  # networkx ausente, o resultado sin params
            print(f"[aviso] no se pudo dibujar el grafo en el inset: {exc}")

    return ax


def gradient_stats_at_random_inits(result):
    """
    Diagnóstico de barren plateau propiamente tal.

    Un barren plateau NO es que el gradiente baje a medida que el algoritmo
    converge — eso es simplemente estar cerca del mínimo. Un barren plateau es
    que el paisaje sea exponencialmente plano **en puntos de parámetros
    aleatorios**, de modo que un optimizador que arranca sin información no
    tenga señal que seguir. La firma es que Var(∂E/∂θ_j) decaiga
    exponencialmente al crecer el circuito.

    Esta función reevalúa el gradiente exacto en los MISMOS puntos iniciales
    aleatorios θ⁰ que se guardaron en la corrida, ansatz por ansatz. No
    requiere volver a optimizar: se reconstruyen los operadores del ansatz a
    partir de `ansatz_op_indices` y se evalúa ∇E(θ⁰).

    Retorna un dict con, por cada tamaño de ansatz k, la varianza y la
    magnitud media/máxima de las derivadas parciales en los puntos aleatorios.
    """
    n = result["n"]
    edges = [tuple(e) for e in result["edges"]]
    l = result["l"]

    prob = prepare_problem(n, edges)
    pool = obtener_pool(n, edges, l, base=result.get("base", "angular"))
    A_sparse, _ = pool_to_sparse(pool["ops"], prob["Hf_sparse"])
    pool["ops"] = None

    A_ansatz = []
    spectra = []

    ks = []
    var_grad = []
    mean_abs_grad = []
    max_abs_grad = []

    for k, runs_k in enumerate(result["random_runs"]):
        if k == 0:
            continue

        # incorporar el operador que ADAPT eligió en este paso.
        # Se hace antes de mirar runs_k para que el ansatz nunca se desincronice
        # de `ansatz_op_indices` si alguna iteración viniera sin reinicios.
        idx = result["ansatz_op_indices"][k - 1]
        A_ansatz.append(A_sparse[idx])
        spectra.append(operator_spectrum(A_sparse[idx]))

        if not runs_k:
            continue

        todas = []
        for run in runs_k:
            x0 = np.asarray(run["x0"], dtype=float)
            _, g = energy_and_grad(x0, spectra, A_ansatz, prob["psi0"], prob["Hf_mat"])
            todas.append(g)

        todas = np.concatenate(todas)

        ks.append(k)
        var_grad.append(float(np.var(todas)))
        mean_abs_grad.append(float(np.mean(np.abs(todas))))
        max_abs_grad.append(float(np.max(np.abs(todas))))

    return {
        "k": np.asarray(ks, dtype=int),
        "var": np.asarray(var_grad, dtype=float),
        "mean_abs": np.asarray(mean_abs_grad, dtype=float),
        "max_abs": np.asarray(max_abs_grad, dtype=float),
    }


def figura_bp_con_grafo(
    result,
    figsize=(10.6, 5.4),
    width_ratios=(3.1, 1.0),
    cmap="turbo",
    mode="final",
    con_leyenda_colores=True,
    **kwargs,
):
    """
    Arma la figura completa: el paisaje de optimización a la izquierda y, al
    lado, el grafo del problema coloreado con la solución Max-3-Cut que
    encontró el algoritmo.

    Es la forma recomendada de generar la figura para el paper: el grafo no
    tapa los datos y se lee a tamaño de publicación.

    Los kwargs extra van a `plot_bp_landscape`.
    """
    import matplotlib.pyplot as plt

    fig, (ax_main, ax_g) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": list(width_ratios)},
    )

    plot_bp_landscape(result, ax=ax_main, cmap=cmap, mode=mode,
                      inset_grafo=False, **kwargs)

    sol = solucion_max3cut(result)
    plot_grafo_solucion(result, solucion=sol, ax=ax_g,
                        con_leyenda=con_leyenda_colores)

    ax_g.set_aspect("equal")
    fig.tight_layout()

    return fig, (ax_main, ax_g), sol


def costo_compuertas(result):
    """
    Costo acumulado del ansatz en compuertas entrelazantes de dos qutrits.

    Un operador del pool que actúa no trivialmente sobre w sitios se
    implementa como exp(-i theta G) con la construcción estándar de escalera:
    2(w-1) compuertas de dos qutrits más una rotación local. Un operador de un
    solo sitio (w=1) no necesita entrelazamiento y cuesta 0.

    Por qué importa: comparar dos pools por número de parámetros es injusto,
    porque un pool con operadores de mayor peso baja más el error por
    parámetro pero cada compuerta cuesta más. El costo en compuertas es la
    comparación honesta.

    Retorna el costo acumulado, alineado con las curvas de energía (empieza en
    0 para el ansatz vacío).
    """
    pesos = [peso_operador(lb) for lb in result["ansatz_op_labels"]]
    costos = [2 * (w - 1) for w in pesos]
    return np.concatenate([[0.0], np.cumsum(costos)])


def razon_aproximacion(result, relativo_curva=None):
    """
    Razón de aproximación tal como la define el paper (Ec. 35):

        r = |E_final / E_0|

    Es la métrica de la Tabla I. Se retorna la curva completa (un valor por
    tamaño de ansatz) usando la trayectoria de parámetros reciclados, que es
    la del algoritmo real.

    Relación con el error relativo: r = 1 - eps_rel cuando E_final y E_0 tienen
    el mismo signo, que es el caso acá porque ambos son negativos.
    """
    E0 = float(result["ground_energy"])
    E = np.asarray(result["recycled_energy"], dtype=float)
    return np.abs(E / E0)


def resumen_tabla_i(result):
    """
    Fila comparable con la Tabla I del paper: razón de aproximación final y
    número de parámetros del ansatz.
    """
    r = razon_aproximacion(result)
    E0 = float(result["ground_energy"])
    err_rel = abs(float(result["recycled_energy"][-1]) - E0) / abs(E0)

    return {
        "base": result.get("base", "angular"),
        "l": int(result["l"]),
        "params": int(result["num_ansatz_ops"]),
        "r": float(r[-1]),
        "eps_rel": float(err_rel),
        "stop_reason": result["stop_reason"],
    }


def plot_replica_fig1(
    resultados,
    ax=None,
    colores=None,
    marcadores=None,
    floor=1e-12,
    titulo=None,
):
    """
    Réplica de las curvas de Qudit-ADAPT de la Fig. 1 del paper: error
    relativo `eps_rel = |E - E0| / |E0|` (Ec. 33) contra número de parámetros
    variacionales.

    `resultados` es un dict {etiqueta: resultado}, típicamente
    {"Qudit-ADAPT (l=1)": res_l1, "Qudit-ADAPT (l=2)": res_l2}.

    Se usan los colores y marcadores del paper: rojo con círculos para l=1,
    azul con cruces para l=2.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 5.2))

    if colores is None:
        colores = ["#D62728", "#1F3FD6", "#2CA02C"]
    if marcadores is None:
        marcadores = ["o", "x", "s"]

    for i, (etiqueta, res) in enumerate(resultados.items()):
        curvas = bp_error_curves(res, floor=floor, relativo=True)
        ax.plot(
            curvas["k"], curvas["recycled"],
            color=colores[i % len(colores)],
            marker=marcadores[i % len(marcadores)],
            ms=4.5, lw=1.4, ls="--", label=etiqueta,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Number of variational parameters")
    ax.set_ylabel(r"$\epsilon_{\rm rel}$")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.95)
    if titulo:
        ax.set_title(titulo)

    return ax


# ============================================================
# Conteo de compuertas en el conjunto nativo de iones atrapados
# ============================================================
#
# Modelo de compilación siguiendo el procesador universal de qudits de
# Ringbauer et al., Nat. Phys. 18, 1053 (2022) [arXiv:2109.06903], que es la
# referencia de hardware del paper. El conjunto nativo es
#
#   R^(i,j)(theta, phi) = exp(-i theta sigma_phi^(i,j) / 2)          (1 qudit)
#   MS^(i,j)(theta, phi) = exp(-i theta/4 (sigma_phi (x) 1 + 1 (x) sigma_phi)^2)
#
# donde (i,j) selecciona un PAR DE NIVELES. Una unitaria arbitraria de un
# qudit se descompone en O(d^2) rotaciones de dos niveles vía Givens; para
# d = 3 hacen falta a lo más 3 (los pares 01, 02, 12).
#
# Para implementar exp(-i theta G) con G = (x)_s A_s un producto sobre w
# sitios se usa la construcción estándar de escalera:
#
#   1. diagonalizar cada factor local:  A_s = U_s D_s U_s^dag
#   2. aplicar la fase diagonal, entrelazando con 2(w-1) compuertas MS
#   3. deshacer las U_s
#
# Lo que se cuenta como EXACTO: el número de rotaciones de dos niveles que
# necesita cada factor local, calculado de la matriz (cuántos pares de niveles
# conecta). Lo que es MODELO: las 2(w-1) MS de la escalera, extrapoladas de la
# construcción de qubits y de la compuerta Cex de ese trabajo, que se
# descompone en 2 MS con independencia de d y fija así el caso w = 2.

_PARES_NIVELES = [(0, 1), (0, 2), (1, 2)]

# Ángulo genérico para compilar exp(-i theta G) de un generador de un solo
# sitio. El número de pulsos no depende de theta salvo en un conjunto de medida
# nula (theta = 0, pi/2, ...), así que basta un valor cualquiera fuera de él.
_THETA_GENERICO = 0.37

# La escalera de MS deja la fase parametrizada sobre un único sitio, como una
# unitaria diagonal de un qutrit. Se cobra el peor caso del Algoritmo 1: las
# dos fases relativas independientes, a tres pulsos cada una. Es el punto donde
# el conteo sigue siendo MODELO y no algoritmo publicado.
_COSTE_FASE_DIAGONAL = 6


def _givens_local(A, tol=1e-10):
    """
    Rotaciones de dos niveles necesarias para diagonalizar una matriz local
    hermítica de 3x3.

    Es donde se separan las dos bases:
      - lambda_3, lambda_8 son diagonales            -> 0
      - lambda_1, lambda_2 conectan sólo (0,1)       -> 1  (ya es nativa)
      - cualquier otra                               -> 3

    El último caso es la cota de Givens, d(d-1)/2 = 3 para d = 3, y no el
    número de pares que conecta la matriz: al anular un par la rotación
    repuebla en general el tercero, así que conectar dos pares no se
    diagonaliza con dos rotaciones. Contar pares da una cota inferior, no el
    valor exacto; sólo coincide cuando la matriz es diagonal o vive dentro de
    un único par de niveles, que son justamente los casos de Gell-Mann.
    """
    A = np.asarray(A, dtype=complex)
    pares = sum(1 for (i, j) in _PARES_NIVELES if abs(A[i, j]) > tol)
    return pares if pares <= 1 else 3


def _factores_locales(label, base):
    """
    Factores locales de un operador del pool a partir de su etiqueta.

    Retorna (dict {sitio: matriz local}, es_producto).

    `es_producto` es False cuando la hermitización (O + O^dag)/2 NO factoriza:
    eso pasa si dos o más sitios tienen factor local no hermítico, porque
    entonces G es una SUMA de dos productos que no conmutan y la escalera
    simple ya no sirve (haría falta trotterizar).
    """
    import ast as _ast
    from funciones.utilidades import Jx1, Jy1, Jz1
    from funciones.utilidades_gellmann import GELLMANN

    partes = _ast.literal_eval(label)
    porsitio = {}
    for p in partes:
        porsitio.setdefault(p[0], []).append(p[1])

    locales = {}
    n_no_herm = 0

    for sitio, factores in porsitio.items():
        if base == "gellmann":
            M = GELLMANN[int(factores[0])]
        elif base == "heisenberg":
            from funciones.utilidades_heisenberg import hw_matriz
            a, b = int(factores[0][0]), int(factores[0][1])
            M = hw_matriz(a, b)
        else:
            tabla = {"x": Jx1.full(), "y": Jy1.full(), "z": Jz1.full()}
            M = np.eye(3, dtype=complex)
            for eje in factores:
                M = M @ tabla[eje]

        if np.max(np.abs(M - M.conj().T)) > 1e-10:
            n_no_herm += 1
            M = (M + M.conj().T) / 2      # la parte hermítica local

        locales[sitio] = M

    return locales, (n_no_herm <= 1)


def conteo_compuertas(label, base="angular"):
    """
    Compuertas nativas para implementar exp(-i theta G) de un operador del pool.

    Retorna dict con:
      'r_dos_niveles' : rotaciones de un qudit de dos niveles
      'ms'            : compuertas Mølmer-Sørensen
      'peso'          : sitios en que actúa
      'es_producto'   : si la escalera simple aplica (ver _factores_locales)
    """
    from funciones.utilidades_ringbauer import coste, coste_diagonalizacion

    locales, es_producto = _factores_locales(label, base)
    w = len(locales)

    if w == 1:
        # Un generador de un solo sitio no necesita escalera ni conjugación:
        # exp(-i theta G) YA ES una unitaria de un qutrit y se compila directo.
        M = next(iter(locales.values()))
        r = coste(sla.expm(-1j * _THETA_GENERICO * M), adyacentes=True)
    else:
        # Diagonalizar cada factor local y deshacerlo (de ahí el factor dos),
        # más la fase diagonal que la escalera de MS deja sobre un solo sitio.
        r = 2 * sum(coste_diagonalizacion(M, adyacentes=True)
                    for M in locales.values())
        r += _COSTE_FASE_DIAGONAL

    # escalera de entrelazamiento
    ms = 2 * (w - 1) if w > 1 else 0

    return {
        "r_dos_niveles": int(r),
        "ms": int(ms),
        "peso": int(w),
        "es_producto": bool(es_producto),
    }


def conteo_compuertas_ansatz(result, acumulado=True):
    """
    Conteo de compuertas del ansatz completo, iteración por iteración.

    Retorna dict de arrays alineados con las curvas de energía (empiezan en 0
    para el ansatz vacío): 'r_dos_niveles', 'ms', 'total', y 'no_producto'
    (cuántos generadores requieren trotterización).
    """
    base = result.get("base", "angular")

    r_list, ms_list, np_list = [0], [0], [0]
    for label in result["ansatz_op_labels"]:
        c = conteo_compuertas(label, base=base)
        r_list.append(c["r_dos_niveles"])
        ms_list.append(c["ms"])
        np_list.append(0 if c["es_producto"] else 1)

    r = np.asarray(r_list, dtype=float)
    ms = np.asarray(ms_list, dtype=float)
    nop = np.asarray(np_list, dtype=float)

    if acumulado:
        r, ms, nop = np.cumsum(r), np.cumsum(ms), np.cumsum(nop)

    return {"r_dos_niveles": r, "ms": ms, "total": r + ms, "no_producto": nop}


def plot_conteo_compuertas(
    resultados,
    axes=None,
    floor=1e-16,
    colores=None,
    relativo=True,
    umbral=None,
):
    """
    Error contra compuertas nativas del procesador de iones atrapados,
    separando rotaciones de un qudit, compuertas MS, y el total.

    `resultados` es un dict {etiqueta: resultado}. Tres paneles:
      (a) rotaciones de dos niveles R^(i,j)
      (b) compuertas Mølmer-Sørensen
      (c) total

    Si `umbral` es un valor de error, se marca con una línea horizontal y se
    anota cuántas compuertas necesita cada base para alcanzarlo, que es la
    comparación justa entre pools que convergen a ritmos distintos.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    axes = list(np.atleast_1d(axes))

    if colores is None:
        colores = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    claves = [("r_dos_niveles", r"Rotaciones $R^{(i,j)}$ (acumuladas)", "(a) rotaciones de un qudit"),
              ("ms", r"Compuertas MS (acumuladas)", "(b) compuertas Mølmer-Sørensen"),
              ("total", "Compuertas nativas totales", "(c) total")]

    for idx, (etiqueta, res) in enumerate(resultados.items()):
        color = colores[idx % len(colores)]
        E0 = float(res["ground_energy"])
        esc = abs(E0) if relativo else 1.0
        err = np.clip(np.abs(np.asarray(res["recycled_energy"], float) - E0) / esc,
                      floor, None)
        cuentas = conteo_compuertas_ansatz(res)

        for ax, (clave, _, _) in zip(axes, claves):
            x = cuentas[clave]
            m = min(len(err), len(x))
            ax.plot(x[:m], err[:m], color=color, lw=2.0, marker="o", ms=3.4,
                    label=etiqueta, zorder=4)

            if umbral is not None:
                bajo = np.where(err[:m] < umbral)[0]
                if len(bajo):
                    k = int(bajo[0])
                    ax.plot([x[k]], [err[k]], marker="*", ms=15, color=color,
                            markeredgecolor="white", markeredgewidth=0.8, zorder=6)

    for ax, (_, xlabel, titulo) in zip(axes, claves):
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\epsilon_{\rm rel}$" if relativo else r"$|E-E_0|$")
        ax.set_title(titulo)
        ax.grid(True, which="major", alpha=0.35)
        if umbral is not None:
            ax.axhline(umbral, color="#888888", ls=":", lw=1.2, zorder=1)
        ax.legend(loc="lower left", framealpha=0.92)

    return axes


def costo_mediciones(result):
    """
    Gradientes acumulados que hay que medir a lo largo de la corrida.

    En cada iteración ADAPT evalúa el gradiente de TODOS los operadores del
    pool para elegir el siguiente, así que el costo es |pool| por iteración.
    En hardware éste —y no el número de compuertas— suele ser el costo
    dominante del algoritmo.

    Es la métrica donde un pool más chico puede compensar una convergencia más
    lenta: lo que importa es |pool| x n_iteraciones para llegar a una precisión
    dada, no cuál converge en menos pasos.
    """
    n = int(result["num_ansatz_ops"])
    return np.arange(n + 1, dtype=float) * float(result["pool_size"])


def plot_comparacion_pools(
    resultados,
    axes=None,
    floor=1e-14,
    linthresh=1e-13,
    colores=None,
    mostrar_nube=True,
    incluir_mediciones=True,
):
    """
    Compara varios pools sobre el MISMO problema.

    `resultados` es un dict {etiqueta: resultado}. Todos deben corresponder al
    mismo grafo y Hamiltoniano (se verifica que E0 coincida).

    Tres paneles:
      (a) error vs número de parámetros — la vista habitual.
      (b) error vs costo en compuertas entrelazantes — la comparación justa
          entre pools con operadores de distinto peso.
      (c) error vs gradientes medidos (|pool| x iteración) — el costo que
          domina en hardware, donde un pool más chico puede compensar una
          convergencia más lenta.

    Si `mostrar_nube`, agrega en gris los óptimos finales de las instancias
    aleatorias de cada pool, para ver si alguno es más propenso a mínimos
    locales.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    n_paneles = 3 if incluir_mediciones else 2
    if axes is None:
        _, axes = plt.subplots(1, n_paneles, figsize=(6.2 * n_paneles, 5.0))
    axes = list(np.atleast_1d(axes))
    ax_p, ax_c = axes[0], axes[1]
    ax_m = axes[2] if incluir_mediciones and len(axes) > 2 else None

    if colores is None:
        colores = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    E0s = {k: float(v["ground_energy"]) for k, v in resultados.items()}
    if max(E0s.values()) - min(E0s.values()) > 1e-8:
        raise ValueError(f"Los resultados no son del mismo problema: E0 = {E0s}")

    for idx, (etiqueta, res) in enumerate(resultados.items()):
        color = colores[idx % len(colores)]
        E0 = float(res["ground_energy"])

        err = np.clip(np.abs(np.asarray(res["recycled_energy"], float) - E0), floor, None)
        k = np.arange(len(err))
        costo = costo_compuertas(res)
        m = min(len(err), len(costo))

        etiq = f"{etiqueta} (pool {res['pool_size']})"

        ax_p.plot(k, err, color=color, lw=2.0, marker="o", ms=3.4, label=etiq, zorder=4)
        ax_c.plot(costo[:m], err[:m], color=color, lw=2.0, marker="o", ms=3.4,
                  label=etiq, zorder=4)

        if ax_m is not None:
            med = costo_mediciones(res)
            mm = min(len(err), len(med))
            ax_m.plot(med[:mm], err[:mm], color=color, lw=2.0, marker="o", ms=3.4,
                      label=etiq, zorder=4)

        if mostrar_nube:
            ks, errs = bp_error_cloud(res, floor=floor, mode="final")
            if len(errs):
                ax_p.scatter(ks, errs, marker="_", s=55, linewidths=1.0,
                             color=color, alpha=0.28, zorder=2)

    paneles = [(ax_p, "Número de parámetros", "(a) por parámetro"),
               (ax_c, "Compuertas de dos qutrits (acumuladas)",
                "(b) por costo en compuertas")]
    if ax_m is not None:
        paneles.append((ax_m, "Gradientes medidos (acumulados)",
                        "(c) por costo de medición"))

    for ax, xlabel, titulo in paneles:
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$|E-E_0|$")
        ax.grid(True, which="major", alpha=0.35)
        ax.legend(loc="lower left", framealpha=0.92)
        ax.set_title(titulo)

    ax_p.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if ax_m is not None:
        ax_m.set_xscale("log")

    return axes


def plot_bp_gradients(result, stats=None, axes=None, title=None):
    """
    Dos paneles de gradientes:

    (a) Gradientes de selección de ADAPT a lo largo de su trayectoria. Bajan
        porque el algoritmo converge — es una traza de convergencia, no una
        prueba de ausencia de barren plateau.

    (b) Estadística del gradiente en los puntos iniciales aleatorios, en
        función del tamaño del ansatz. Éste sí es el diagnóstico de barren
        plateau: si el paisaje se aplanara exponencialmente con la
        profundidad, Var(∂E/∂θ) caería en línea recta en escala log.

    `stats` es la salida de `gradient_stats_at_random_inits`; si es None se
    calcula (implica reconstruir el pool, unos segundos).
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax_a, ax_b = axes

    # --- (a) gradientes de selección a lo largo de ADAPT ------------------
    max_grad = np.asarray(result["max_grad_trace"], float)
    norm_grad = np.asarray(result["norm_grad_trace"], float)
    k_a = np.arange(len(max_grad))

    ax_a.plot(k_a, norm_grad, color="#0072B2", lw=2.0, marker="o", ms=3.5,
              label=r"$\|g\|_2$ sobre el pool")
    ax_a.plot(k_a, max_grad, color="#D55E00", lw=2.0, marker="s", ms=3.5,
              label=r"$\max_j |g_j|$ (operador elegido)")
    ax_a.axhline(result["epsilon"], color="#888888", ls="--", lw=1.2,
                 label=rf"$\epsilon = {result['epsilon']:g}$")

    ax_a.set_yscale("log")
    ax_a.set_xlabel("Iteración de ADAPT")
    ax_a.set_ylabel("Magnitud del gradiente")
    ax_a.set_title("(a) Gradientes de selección\n(traza de convergencia)")
    ax_a.legend(loc="best", framealpha=0.92)
    ax_a.grid(True, which="major", alpha=0.35)

    # --- (b) planitud del paisaje en puntos aleatorios ---------------------
    if stats is None:
        stats = gradient_stats_at_random_inits(result)

    ax_b.plot(stats["k"], stats["var"], color="#8E44AD", lw=2.0, marker="o", ms=3.5,
              label=r"Var$(\partial E/\partial\theta_j)$")
    ax_b.plot(stats["k"], stats["mean_abs"], color="#16A085", lw=2.0, marker="s", ms=3.5,
              label=r"$\langle|\partial E/\partial\theta_j|\rangle$")

    ax_b.set_yscale("log")
    ax_b.set_xlabel("Número de parámetros del ansatz")
    ax_b.set_ylabel("Gradiente en $\\theta^0$ aleatorio")
    ax_b.set_title("(b) Planitud del paisaje en puntos aleatorios\n(diagnóstico de barren plateau)")
    ax_b.legend(loc="best", framealpha=0.92)
    ax_b.grid(True, which="major", alpha=0.35)

    if title:
        ax_a.figure.suptitle(title)

    return axes
