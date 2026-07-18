from collections import defaultdict
from functools import lru_cache
import sympy as sp
import numpy as np
import qutip as qt

import numpy as np
import qutip as qt
from scipy.optimize import minimize

import ast
import csv
import json
import math
import random
import time

from itertools import combinations


ZERO = sp.Integer(0)
I = sp.I

def X_local():
    """
    Operador local qutrit:
    X = (Lz)^2 + sqrt(2) Lx
      = [[1,1,0],
         [1,0,1],
         [0,1,1]]
    """
    return qt.Qobj(np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=complex))

def X_site(n, j):
    """
    Inserta el operador X en el sitio j de un sistema de n qutrits.
    j va de 1 a n.
    """
    ops = []
    for site in range(1, n + 1):
        if site == j:
            ops.append(X_local())
        else:
            ops.append(qt.qeye(3))
    return qt.tensor(ops)

def Hi_qutip(n, omega0=1):
    """
    H_i = -omega0 * sum_{j=1}^n X_j
    """
    H = 0
    for j in range(1, n + 1):
        H += -omega0 * X_site(n, j)
    return H

def Hp_qutip(n, edges):
    """
    H_p = sum_{(i,j) in edges} [
        Jz_i Jz_j
        - 2 (Jz_i^2 + Jz_j^2)
        + 3 Jz_i^2 Jz_j^2
    ]
    """
    H = 0
    for i, j in edges:
        Jzi = Jz_site(n, i)
        Jzj = Jz_site(n, j)

        H += Jzi * Jzj
        H += -2 * (Jzi * Jzi + Jzj * Jzj)
        H += 3 * (Jzi * Jzi * Jzj * Jzj)

    return H

def J(site, axis):
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis debe ser 'x', 'y' o 'z'")
    return (site, axis)


def canonicalize_monomial(monomial):
    return tuple(sorted(monomial, key=lambda op: op[0]))


def clean_expr(expr):
    """
    Limpieza liviana:
    - canonicaliza monomios
    - suma términos repetidos
    - elimina coeficientes cero
    No usa simplify para ganar velocidad.
    """
    out = defaultdict(lambda: ZERO)

    for mono, coeff in expr.items():
        if coeff == 0:
            continue
        mono = canonicalize_monomial(tuple(mono))
        out[mono] += coeff

    return {mono: coeff for mono, coeff in out.items() if coeff != 0}


def expr_from_monomial(monomial, coeff=1):
    monomial = canonicalize_monomial(tuple(monomial))
    coeff = sp.sympify(coeff)

    if coeff == 0:
        return {}

    return {monomial: coeff}


def add_expr(expr1, expr2):
    out = defaultdict(lambda: ZERO)

    for mono, coeff in expr1.items():
        if coeff != 0:
            out[mono] += coeff

    for mono, coeff in expr2.items():
        if coeff != 0:
            out[mono] += coeff

    return clean_expr(out)


def scale_expr(c, expr):
    c = sp.sympify(c)

    if c == 0 or not expr:
        return {}

    out = {mono: c * coeff for mono, coeff in expr.items() if coeff != 0}
    return clean_expr(out)


@lru_cache(maxsize=None)
def _comm_local_local_cached(op1, op2):
    i, a = op1
    j, b = op2

    if i != j:
        return None

    if a == b:
        return None

    table = {
        ('x', 'y'): ((i, 'z'),  I),
        ('y', 'z'): ((i, 'x'),  I),
        ('z', 'x'): ((i, 'y'),  I),

        ('y', 'x'): ((i, 'z'), -I),
        ('z', 'y'): ((i, 'x'), -I),
        ('x', 'z'): ((i, 'y'), -I),
    }

    return table[(a, b)]


def sum_expr(expr_list):
    out = defaultdict(lambda: ZERO)

    for expr in expr_list:
        for mono, coeff in expr.items():
            if coeff != 0:
                out[mono] += coeff

    return clean_expr(out)


def comm_expr_expr(expr1, expr2):
    out = defaultdict(lambda: ZERO)

    for m1, c1 in expr1.items():
        if c1 == 0:
            continue
        for m2, c2 in expr2.items():
            if c2 == 0:
                continue

            term = comm_monomial_monomial(m1, m2)

            if not term:
                continue

            factor = c1 * c2
            for mono, coeff in term.items():
                out[mono] += factor * coeff

    return clean_expr(out)


def left_multiply_monomial(op_or_mono, expr):
    if isinstance(op_or_mono, tuple) and len(op_or_mono) == 2 and isinstance(op_or_mono[1], str):
        op_or_mono = (op_or_mono,)

    op_or_mono = tuple(op_or_mono)

    if not expr:
        return {}

    out = defaultdict(lambda: ZERO)
    for mono, coeff in expr.items():
        if coeff != 0:
            new_mono = canonicalize_monomial(op_or_mono + mono)
            out[new_mono] += coeff

    return clean_expr(out)


def right_multiply_monomial(expr, op_or_mono):
    if isinstance(op_or_mono, tuple) and len(op_or_mono) == 2 and isinstance(op_or_mono[1], str):
        op_or_mono = (op_or_mono,)

    op_or_mono = tuple(op_or_mono)

    if not expr:
        return {}

    out = defaultdict(lambda: ZERO)
    for mono, coeff in expr.items():
        if coeff != 0:
            new_mono = canonicalize_monomial(mono + op_or_mono)
            out[new_mono] += coeff

    return clean_expr(out)


@lru_cache(maxsize=None)
def _comm_local_monomial_cached(local_op, monomial):
    monomial = tuple(monomial)

    if len(monomial) == 0:
        return ()

    out = defaultdict(lambda: ZERO)

    for k, factor in enumerate(monomial):
        cached = _comm_local_local_cached(local_op, factor)

        if cached is None:
            continue

        op_out, coeff = cached
        left_part = monomial[:k]
        right_part = monomial[k+1:]

        new_mono = canonicalize_monomial(left_part + (op_out,) + right_part)
        out[new_mono] += coeff

    return tuple((mono, coeff) for mono, coeff in out.items() if coeff != 0)


def comm_local_monomial(local_op, monomial):
    data = _comm_local_monomial_cached(local_op, tuple(monomial))
    return dict(data)


def comm_monomial_monomial(m1, m2):
    m1 = tuple(m1)
    m2 = tuple(m2)

    if len(m1) == 0 or len(m2) == 0:
        return {}

    if len(m1) == 1:
        return comm_local_monomial(m1[0], m2)

    a1 = (m1[0],)
    rest = m1[1:]

    term1_base = comm_monomial_monomial(rest, m2)
    term2_base = comm_local_monomial(m1[0], m2)

    term1 = left_multiply_monomial(a1, term1_base) if term1_base else {}
    term2 = right_multiply_monomial(term2_base, rest) if term2_base else {}

    return add_expr(term1, term2)


def nested_commutators(H, dH, order=3):
    """
    Calcula los conmutadores anidados:

        O1 = [H, dH]
        O2 = [H, O1]
        O3 = [H, O2]
        ...

    Parámetros
    ----------
    H : dict
        Expresión que representa el Hamiltoniano principal.
    dH : dict
        Expresión que representa la derivada del Hamiltoniano.
    order : int
        Orden máximo de conmutación anidada.

    Retorna
    -------
    dict
        Diccionario:
            1 -> O1
            2 -> O2
            3 -> O3
            ...
    """
    results = {}
    current = dH

    for k in range(1, order + 1):
        current = comm_expr_expr(H, current)
        results[k] = current

    return results


Jx1 = qt.jmat(1, 'x')
Jy1 = qt.jmat(1, 'y')
Jz1 = qt.jmat(1, 'z')
I3  = qt.qeye(3)

def local_op(n, site, op):
    ops = []
    for k in range(1, n + 1):
        ops.append(op if k == site else I3)
    return qt.tensor(ops)

def Jx_site(n, j):
    return local_op(n, j, Jx1)

def Jy_site(n, j):
    return local_op(n, j, Jy1)

def Jz_site(n, j):
    return local_op(n, j, Jz1)



def local_symbol_to_qutip(n, symbol):
    """
    symbol tiene la forma (site, axis), por ejemplo:
    (1, 'y'), (2, 'z')
    """
    site, axis = symbol

    if axis == 'x':
        return Jx_site(n, site)
    elif axis == 'y':
        return Jy_site(n, site)
    elif axis == 'z':
        return Jz_site(n, site)
    else:
        raise ValueError(f"Eje no reconocido: {axis}")
    

def monomial_to_qutip(n, monomial):
    """
    monomial es una tupla de símbolos locales, por ejemplo:
    ((1,'y'), (2,'z'))
    """
    op = None

    for symbol in monomial:
        op_local = local_symbol_to_qutip(n, symbol)
        op = op_local if op is None else op * op_local

    # por si alguna vez aparece el monomio vacío
    if op is None:
        op = qt.tensor([I3 for _ in range(n)])

    return op


def expr_dict_to_qutip(n, expr_dict, lam_value=None, lam_symbol=None):
    """
    expr_dict = {monomial: coeff}
    Si coeff depende de lambda, se sustituye lambda -> lam_value.
    """
    H = 0

    for monomial, coeff in expr_dict.items():
        op = monomial_to_qutip(n, monomial)

        coeff_eval = coeff
        if lam_symbol is not None and lam_value is not None:
            coeff_eval = sp.sympify(coeff).subs(lam_symbol, lam_value)

        coeff_num = complex(sp.N(coeff_eval))
        H += coeff_num * op

    return H


def Hi(n, omega0=1):
    """
    Hamiltoniano inicial simbólico:
        H_i = -omega0 * sum_{j=1}^n X_j

    con
        X_j = J_{z,j}^2 + sqrt(2) J_{x,j}

    Esto coincide con la definición numérica usada en Hi_qutip,
    donde X_local = [[1,1,0],
                     [1,0,1],
                     [0,1,1]]
    """
    terms = []

    for j in range(1, n + 1):
        # -omega0 * sqrt(2) * Jx_j
        terms.append(
            expr_from_monomial(
                (J(j, "x"),),
                -omega0 * sp.sqrt(2)
            )
        )

        # -omega0 * Jz_j^2
        terms.append(
            expr_from_monomial(
                (J(j, "z"), J(j, "z")),
                -omega0
            )
        )

    return sum_expr(terms)

def Hp(edges):
    """
    Hamiltoniano problema:
        H_p = sum_{(i,j) in edges} [
            Jz_i Jz_j
            - 2(Jz_i^2 + Jz_j^2)
            + 3 Jz_i^2 Jz_j^2
        ]
    """
    terms = []

    for i, j in edges:
        # Jz_i Jz_j
        terms.append(expr_from_monomial((J(i, "z"), J(j, "z")), 1))

        # -2 Jz_i^2
        terms.append(expr_from_monomial((J(i, "z"), J(i, "z")), -2))

        # -2 Jz_j^2
        terms.append(expr_from_monomial((J(j, "z"), J(j, "z")), -2))

        # +3 Jz_i^2 Jz_j^2
        terms.append(
            expr_from_monomial(
                (J(i, "z"), J(i, "z"), J(j, "z"), J(j, "z")),
                3
            )
        )

    return sum_expr(terms)

def Had(n, edges, lam):
    """
    Hamiltoniano adiabático lineal:
        H_ad(lam) = (1-lam) H_i + lam H_p
    """
    hi = Hi(n)
    hp = Hp(edges)

    return add_expr(
        scale_expr(1 - lam, hi),
        scale_expr(lam, hp)
    )
def dHad_dlam(n, edges):
    """
    Derivada respecto de lambda del Hamiltoniano adiabático lineal:
        dH_ad/dlambda = -H_i + H_p
    """
    hi = Hi(n)
    hp = Hp(edges)

    return add_expr(
        scale_expr(-1, hi),
        hp
    )

def canonical_op(op):
    return tuple(sorted(op))

def pool_to_qutip(operator_pool, n_sites):
    pool_qutip = []

    for op in operator_pool:
        O = expr_dict_to_qutip(n_sites, {op: 1.0})
        O_H = (O + O.dag()) / 2
        pool_qutip.append(O_H)

    return pool_qutip




from pathlib import Path
import os
import ast
import copy
import csv
import json
import math
import random
import time
from itertools import combinations

import numpy as np
import qutip as qt
import sympy as sp
from scipy.optimize import minimize


# =========================
# RUTAS DEL PROYECTO
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTADOS_DIR = PROJECT_ROOT / "resultados"
CSV_DIR = RESULTADOS_DIR / "csv"
JSON_DIR = RESULTADOS_DIR / "json"
DATOS_DIR = PROJECT_ROOT / "datos"   # opcional, para guardar archivos de grafos

# Crear carpetas automáticamente si no existen
CSV_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)
DATOS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# FUNCIONES AUXILIARES
# =========================

def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, complex):
        return {"real": float(np.real(obj)), "imag": float(np.imag(obj))}
    else:
        return obj


# =========================
# GENERAR GRAFOS
# =========================

def generar_m_grafos(n, m, min_edges=None, max_edges=None, seed=None, filename=None):
    if seed is not None:
        random.seed(seed)

    total_max_edges = n * (n - 1) // 2

    if min_edges is None:
        min_edges = n
    if max_edges is None:
        max_edges = total_max_edges

    if min_edges > max_edges:
        raise ValueError("min_edges no puede ser mayor que max_edges")

    if max_edges > total_max_edges:
        raise ValueError("max_edges excede el máximo posible")

    todas_las_aristas = list(combinations(range(1, n + 1), 2))

    max_grafos = sum(math.comb(total_max_edges, k) for k in range(min_edges, max_edges + 1))

    if m > max_grafos:
        raise ValueError(f"Máximo número de grafos distintos: {max_grafos}")

    grafos = set()

    while len(grafos) < m:
        k = random.randint(min_edges, max_edges)
        aristas = tuple(sorted(random.sample(todas_las_aristas, k)))
        grafos.add(aristas)

    grafos = [list(g) for g in grafos]

    if filename is not None:
        file_path = Path(filename)

        # Si solo te pasan un nombre como "grafos_n4.txt",
        # lo guardamos automáticamente en /datos
        if not file_path.is_absolute():
            file_path = DATOS_DIR / file_path

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            for g in grafos:
                linea = ", ".join(str(e) for e in g)
                f.write(linea + "\n")

    return grafos


# =========================
# ALGORITMO PRINCIPAL
# =========================

def precompute_operator_spectra(operators):
    spectra = []
    for op in operators:
        evals, evecs = op.eigenstates()
        V = qt.Qobj(np.column_stack([vec.full() for vec in evecs]), dims=op.dims)
        Vdag = V.dag()
        spectra.append((np.array(evals, dtype=float), V, Vdag))
    return spectra


def apply_exp_to_state_from_spectrum(theta, spectrum, state):
    evals, V, Vdag = spectrum

    coeffs = Vdag * state
    phased = coeffs.full().flatten() * np.exp(-1j * float(theta) * evals)
    phased_qobj = qt.Qobj(phased.reshape((-1, 1)), dims=state.dims)

    return V * phased_qobj


def build_ansatz_fast(params, spectra, initial_state):
    state = initial_state
    for theta, spec in zip(params, spectra):
        state = apply_exp_to_state_from_spectrum(theta, spec, state)
    return state


# =========================
# VERSIONES NUMPY PARA ACELERAR EL OPTIMIZADOR
# =========================

def precompute_operator_spectra_numpy(operators):
    spectra = []
    for op in operators:
        evals, evecs = op.eigenstates()
        V = np.column_stack([vec.full().ravel() for vec in evecs])
        Vdag = V.conj().T
        spectra.append((np.array(evals, dtype=float), V, Vdag))
    return spectra


def apply_exp_to_state_from_spectrum_numpy(theta, spectrum, state_vec):
    evals, V, Vdag = spectrum
    coeffs = Vdag @ state_vec
    coeffs = coeffs * np.exp(-1j * float(theta) * evals)
    return V @ coeffs


def build_ansatz_fast_numpy(params, spectra, initial_state_vec):
    state_vec = initial_state_vec
    for theta, spec in zip(params, spectra):
        state_vec = apply_exp_to_state_from_spectrum_numpy(theta, spec, state_vec)
    return state_vec


def cost_function_fast_numpy(params, spectra, initial_state_vec, Hf_mat):
    psi = build_ansatz_fast_numpy(params, spectra, initial_state_vec)
    return float(np.real(np.vdot(psi, Hf_mat @ psi)))


# =========================
# ALGORITMO PRINCIPAL MEJORADO
# =========================

def cd_adapt_vqe_algorithm(n, edges, l, epsilon=1e-2, max_iteration=30, show=False):
    start_time = time.time()

    lam = sp.symbols("lam", real=True)

    Hf = Hp_qutip(n, edges)
    Hi = Hi_qutip(n, 1)

    evals_i, evecs_i = Hi.eigenstates()
    evals_i = np.array(evals_i, dtype=float)
    initial_ground_energy = float(evals_i[0])
    psi_0 = evecs_i[0]

    evals_f, evecs_f = Hf.eigenstates()
    evals_f = np.array(evals_f, dtype=float)

    E0_f = float(evals_f[0])

    first_excited_energy = float(evals_f[1]) if len(evals_f) > 1 else None
    spectral_gap = float(evals_f[1] - evals_f[0]) if len(evals_f) > 1 else None

    tol = 1e-10
    ground_indices = [i for i, ev in enumerate(evals_f) if abs(ev - E0_f) < tol]
    ground_degeneracy = len(ground_indices)

    if show:
        print("Ground energy of Hp =", E0_f)

    H = Had(n, edges, lam)
    dH = dHad_dlam(n, edges)

    results = nested_commutators(H, dH, order=3)

    O1 = results[1]
    O3 = results[3]

    operator_pool_unique_1 = list({canonical_op(op) for op in O1.keys()})
    operator_pool_unique_3 = list({canonical_op(op) for op in O3.keys()})

    operator_pool_1 = pool_to_qutip(operator_pool_unique_1, n)
    operator_pool_3 = pool_to_qutip(operator_pool_unique_3, n)

    if l == 1:
        operator_pool = operator_pool_1
        operator_pool_labels = [str(op) for op in operator_pool_unique_1]
    elif l == 2:
        operator_pool = operator_pool_1 + operator_pool_3
        operator_pool_labels = (
            [str(op) for op in operator_pool_unique_1] +
            [str(op) for op in operator_pool_unique_3]
        )
    else:
        raise ValueError("solo se admiten l=1 o l=2")

    if show:
        print("Cantidad de operadores en el pool: ", len(operator_pool))
        print("........................Comenzando algoritmo........................")

    iteration = 0

    ansatz_ops = []
    ansatz_op_labels = []
    ansatz_op_indices = []

    ansatz_spectra_qutip = []
    ansatz_spectra_numpy = []

    params = []

    grad_trace = []
    norm_grad_trace = []
    energy_trace = [float(np.real(qt.expect(Hf, psi_0)))]

    commutators = [Hf * A - A * Hf for A in operator_pool]

    initial_state_vec = psi_0.full().ravel()
    Hf_mat = Hf.full()

    while True:
        psi = build_ansatz_fast(params, ansatz_spectra_qutip, psi_0) if params else psi_0

        gradients = []
        for comm in commutators:
            grad = psi.dag() @ comm @ psi
            gradients.append(abs(complex(grad)))

        gradients = np.array(gradients, dtype=float)

        max_grad = float(np.max(gradients)) if len(gradients) > 0 else 0.0
        norm = float(np.linalg.norm(gradients)) if len(gradients) > 0 else 0.0

        grad_trace.append(max_grad)
        norm_grad_trace.append(norm)

        if norm < epsilon or len(ansatz_ops) >= max_iteration:
            break

        max_index = int(np.argmax(gradients))

        new_op = operator_pool[max_index]
        ansatz_ops.append(new_op)
        ansatz_op_indices.append(max_index)
        ansatz_op_labels.append(operator_pool_labels[max_index])

        spec_qutip = precompute_operator_spectra([new_op])[0]
        ansatz_spectra_qutip.append(spec_qutip)

        spec_numpy = precompute_operator_spectra_numpy([new_op])[0]
        ansatz_spectra_numpy.append(spec_numpy)

        init_theta = params + [0.0]

        result = minimize(
            cost_function_fast_numpy,
            init_theta,
            args=(ansatz_spectra_numpy, initial_state_vec, Hf_mat),
            method="BFGS",
            options={"maxiter": 1000}
        )

        params = list(map(float, result.x))
        energy_trace.append(float(np.real(result.fun)))

        if show:
            print(f"Iteración {iteration+1} | Energía: {result.fun:.10f} | Grad norm: {norm:.5e}")

        iteration += 1

    if len(params) > 0:
        result = minimize(
            cost_function_fast_numpy,
            params,
            args=(ansatz_spectra_numpy, initial_state_vec, Hf_mat),
            method="BFGS",
            options={"maxiter": 100}
        )

        params = list(map(float, result.x))
        optimizer_success = bool(result.success)
        optimizer_status = int(result.status)
        optimizer_message = str(result.message)
        optimizer_nfev = int(getattr(result, "nfev", -1))
        optimizer_njev = int(getattr(result, "njev", -1))
        optimizer_nit = int(getattr(result, "nit", -1))
        optimizer_fun = float(np.real(result.fun))
    else:
        optimizer_success = True
        optimizer_status = 0
        optimizer_message = "No optimization needed"
        optimizer_nfev = 0
        optimizer_njev = 0
        optimizer_nit = 0
        optimizer_fun = float(np.real(qt.expect(Hf, psi_0)))

    psi_final = build_ansatz_fast(params, ansatz_spectra_qutip, psi_0) if params else psi_0

    final_gradients = []
    for comm in commutators:
        grad = psi_final.dag() @ comm @ psi_final
        final_gradients.append(abs(complex(grad)))

    final_gradients = np.array(final_gradients, dtype=float)

    final_max_gradient = float(np.max(final_gradients)) if len(final_gradients) > 0 else 0.0
    final_gradient_norm = float(np.linalg.norm(final_gradients)) if len(final_gradients) > 0 else 0.0

    grad_trace.append(final_max_gradient)
    norm_grad_trace.append(final_gradient_norm)
    energy_trace.append(optimizer_fun)

    final_energy = float(np.real(qt.expect(Hf, psi_final)))
    difference_ground = float(np.abs(final_energy - E0_f))

    runtime_seconds = float(time.time() - start_time)
    runtime_min = runtime_seconds / 60.0

    if show:
        print("........................Fin del algoritmo........................")
        print(f"Número total de operadores en el ansatz: {len(ansatz_ops)}")
        print(f"Energía final: {final_energy}")
        print(f"Diferencia con el Groundstate: {difference_ground}")
        print(f"Gradiente final: {final_gradient_norm}")
        print(f"Tiempo de ejecución: {runtime_min:.4f} min")

    return {
        "n": n,
        "edges": edges,
        "num_edges": len(edges),
        "l": l,
        "epsilon": float(epsilon),
        "max_iteration": int(max_iteration),

        "initial_energy": initial_ground_energy,

        "ground_energy": E0_f,
        "first_excited_energy": first_excited_energy,
        "spectral_gap": spectral_gap,
        "ground_degeneracy": ground_degeneracy,

        "pool_size": len(operator_pool),
        "pool_labels": operator_pool_labels,

        "final_energy": final_energy,
        "difference_ground": difference_ground,

        "iterations": iteration,
        "num_ansatz_ops": len(ansatz_ops),
        "final_gradient_norm": final_gradient_norm,
        "final_max_gradient": final_max_gradient,

        "params": params,
        "ansatz_op_indices": ansatz_op_indices,
        "ansatz_op_labels": ansatz_op_labels,

        "energy_trace": energy_trace,
        "grad_trace": grad_trace,
        "norm_grad_trace": norm_grad_trace,

        "optimizer_success": optimizer_success,
        "optimizer_status": optimizer_status,
        "optimizer_message": optimizer_message,
        "optimizer_nfev": optimizer_nfev,
        "optimizer_njev": optimizer_njev,
        "optimizer_nit": optimizer_nit,

        "runtime_seconds": runtime_seconds,
        "runtime_min": runtime_min
    }
# =========================
# EJECUTAR MUCHOS GRAFOS
# =========================

def ejecutar_grafos(
    input_file,
    output_csv,
    output_json,
    n,
    l,
    epsilon=1e-2,
    max_iteration=30,
    show=False,
    start_idx=None,
    end_idx=None
):
    input_path = Path(input_file)

    # Si no es absoluta, asumimos que está dentro de /datos
    if not input_path.is_absolute():
        input_path = DATOS_DIR / input_path

    output_csv_path = Path(output_csv)
    if not output_csv_path.is_absolute():
        output_csv_path = CSV_DIR / output_csv_path

    output_json_path = Path(output_json)
    if not output_json_path.is_absolute():
        output_json_path = JSON_DIR / output_json_path

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        lineas = f.readlines()

    # =========================
    # NUEVO: seleccionar rango
    # =========================
    if start_idx is None:
        start_idx = 1
    if end_idx is None:
        end_idx = len(lineas)

    lineas = lineas[start_idx - 1:end_idx]

    csv_fieldnames = [
        "grafo_id",
        "num_edges",
        "ground_energy",
        "iteraciones",
        "energia_final",
        "diferencia_ground",
        "gradiente_final",
        "runtime_min",
        "initial_energy"
    ]

    json_results = []

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()

        # =========================
        # IMPORTANTE: cambiar start
        # =========================
        for i, linea in enumerate(lineas, start=start_idx):
            linea = linea.strip()

            if not linea:
                continue

            try:
                edges = ast.literal_eval("[" + linea + "]")

                result = cd_adapt_vqe_algorithm(
                    n=n,
                    edges=edges,
                    l=l,
                    epsilon=epsilon,
                    max_iteration=max_iteration,
                    show=show
                )

                result_complete = {"grafo_id": i, **result}
                json_results.append(to_jsonable(result_complete))

                row_csv = {
                    "grafo_id": i,
                    "num_edges": result["num_edges"],
                    "ground_energy": result["ground_energy"],
                    "iteraciones": result["iterations"],
                    "energia_final": result["final_energy"],
                    "diferencia_ground": result["difference_ground"],
                    "gradiente_final": result["final_gradient_norm"],
                    "runtime_min": result["runtime_min"],
                    "initial_energy": result["initial_energy"]
                }

                writer.writerow(row_csv)

            except Exception as e:
                print(f"Error en grafo {i}: {e}")

                json_results.append({
                    "grafo_id": i,
                    "error": str(e)
                })

                writer.writerow({
                    "grafo_id": i,
                    "num_edges": "ERROR",
                    "ground_energy": "ERROR",
                    "iteraciones": "ERROR",
                    "energia_final": "ERROR",
                    "diferencia_ground": "ERROR",
                    "gradiente_final": "ERROR",
                    "runtime_min": "ERROR",
                    "initial_energy": "ERROR"
                })

    with open(output_json_path, "w", encoding="utf-8") as jf:
        json.dump(json_results, jf, indent=4, ensure_ascii=False)

    print(f"CSV guardado en: {output_csv_path}")
    print(f"JSON guardado en: {output_json_path}")
    print(f"Procesados grafos desde {start_idx} hasta {end_idx}")

def cd_adapt_vqe_algorithm_profundo(
    n,
    edges,
    l,
    epsilon=1e-2,
    max_iteration=30,
    show=False,
    top_k_gradients=10
):
    """
    Versión profunda del algoritmo CD-ADAPT-VQE.

    Guarda información adicional para:
    - reconstruir el ansatz final,
    - conocer el orden exacto de operadores seleccionados,
    - guardar parámetros optimizados,
    - analizar trazas de energía y gradientes,
    - reconstruir posteriormente un circuito cuántico a partir del ansatz.

    Importante:
    El orden del pool se fija usando sorted(..., key=str), para que sea determinista.
    Esto es clave para reconstruir el ansatz después.
    """

    start_time = time.time()

    lam = sp.symbols("lam", real=True)

    # ============================================================
    # Hamiltonianos
    # ============================================================

    Hf = Hp_qutip(n, edges)
    Hi = Hi_qutip(n, 1)

    # Ground state de Hi
    evals_i, evecs_i = Hi.eigenstates()
    evals_i = np.array(evals_i, dtype=float)

    initial_ground_energy = float(evals_i[0])
    psi_0 = evecs_i[0]

    # Espectro de Hf
    evals_f, evecs_f = Hf.eigenstates()
    evals_f = np.array(evals_f, dtype=float)

    E0_f = float(evals_f[0])

    first_excited_energy = float(evals_f[1]) if len(evals_f) > 1 else None
    spectral_gap = float(evals_f[1] - evals_f[0]) if len(evals_f) > 1 else None

    tol = 1e-10
    ground_indices = [i for i, ev in enumerate(evals_f) if abs(ev - E0_f) < tol]
    ground_degeneracy = len(ground_indices)

    # Energía inicial medida con Hf
    # Esta es la energía inicial real de la optimización.
    initial_problem_energy = float(np.real(qt.expect(Hf, psi_0)))

    if show:
        print("Ground energy of Hi =", initial_ground_energy)
        print("Initial problem energy <psi_0|Hf|psi_0> =", initial_problem_energy)
        print("Ground energy of Hp =", E0_f)
        print("Spectral gap =", spectral_gap)
        print("Ground degeneracy =", ground_degeneracy)

    # ============================================================
    # Construcción del pool CD
    # ============================================================

    H = Had(n, edges, lam)
    dH = dHad_dlam(n, edges)

    results = nested_commutators(H, dH, order=3)

    O1 = results[1]
    O3 = results[3]

    # IMPORTANTE:
    # En la función original se usaba list({ ... }).
    # Eso NO asegura un orden determinista.
    # Para reconstruir el ansatz después, conviene ordenar por string.
    operator_pool_unique_1 = sorted(
        {canonical_op(op) for op in O1.keys()},
        key=str
    )

    operator_pool_unique_3 = sorted(
        {canonical_op(op) for op in O3.keys()},
        key=str
    )

    operator_pool_1 = pool_to_qutip(operator_pool_unique_1, n)
    operator_pool_3 = pool_to_qutip(operator_pool_unique_3, n)

    operator_pool_labels_1 = [str(op) for op in operator_pool_unique_1]
    operator_pool_labels_3 = [str(op) for op in operator_pool_unique_3]

    if l == 1:
        operator_pool = operator_pool_1
        operator_pool_labels = operator_pool_labels_1
        operator_pool_orders = [1 for _ in operator_pool_1]

    elif l == 2:
        operator_pool = operator_pool_1 + operator_pool_3
        operator_pool_labels = operator_pool_labels_1 + operator_pool_labels_3
        operator_pool_orders = (
            [1 for _ in operator_pool_1] +
            [3 for _ in operator_pool_3]
        )

    else:
        raise ValueError("Solo se admiten l=1 o l=2.")

    pool_size = len(operator_pool)

    if show:
        print("Cantidad de operadores en el pool:", pool_size)
        print("........................Comenzando algoritmo profundo........................")

    # ============================================================
    # Inicialización ADAPT
    # ============================================================

    iteration = 0
    stop_reason = None

    ansatz_ops = []
    ansatz_op_labels = []
    ansatz_op_indices = []
    ansatz_op_orders = []

    ansatz_spectra_qutip = []
    ansatz_spectra_numpy = []

    params = []

    grad_trace = []
    norm_grad_trace = []
    energy_trace = [initial_problem_energy]

    selected_gradient_trace = []
    top_gradients_trace = []
    optimizer_trace = []

    commutators = [Hf * A - A * Hf for A in operator_pool]

    initial_state_vec = psi_0.full().ravel()
    Hf_mat = Hf.full()

    # ============================================================
    # Loop principal ADAPT-VQE
    # ============================================================

    while True:

        psi = build_ansatz_fast(params, ansatz_spectra_qutip, psi_0) if params else psi_0

        # --------------------------------------------------------
        # Cálculo de gradientes
        # --------------------------------------------------------

        gradients = []

        for comm in commutators:
            grad = psi.dag() @ comm @ psi
            gradients.append(abs(complex(grad)))

        gradients = np.array(gradients, dtype=float)

        max_grad = float(np.max(gradients)) if len(gradients) > 0 else 0.0
        norm = float(np.linalg.norm(gradients)) if len(gradients) > 0 else 0.0

        grad_trace.append(max_grad)
        norm_grad_trace.append(norm)

        # Guardar top-k gradientes de esta iteración
        if len(gradients) > 0:
            top_indices = np.argsort(gradients)[::-1][:top_k_gradients]

            top_gradients = []
            for rank, idx in enumerate(top_indices, start=1):
                top_gradients.append({
                    "rank": int(rank),
                    "operator_index": int(idx),
                    "operator_label": str(operator_pool_labels[idx]),
                    "operator_order": int(operator_pool_orders[idx]),
                    "gradient": float(gradients[idx])
                })

            top_gradients_trace.append({
                "iteration": int(iteration),
                "top_gradients": top_gradients
            })

        # --------------------------------------------------------
        # Criterio de parada
        # --------------------------------------------------------

        if norm < epsilon:
            stop_reason = "gradient_norm_below_epsilon"
            break

        if len(ansatz_ops) >= max_iteration:
            stop_reason = "max_iteration_reached"
            break

        # --------------------------------------------------------
        # Selección del operador con mayor gradiente
        # --------------------------------------------------------

        max_index = int(np.argmax(gradients))

        new_op = operator_pool[max_index]

        ansatz_ops.append(new_op)
        ansatz_op_indices.append(max_index)
        ansatz_op_labels.append(operator_pool_labels[max_index])
        ansatz_op_orders.append(operator_pool_orders[max_index])

        selected_gradient_trace.append({
            "iteration": int(iteration + 1),
            "selected_operator_index": int(max_index),
            "selected_operator_label": str(operator_pool_labels[max_index]),
            "selected_operator_order": int(operator_pool_orders[max_index]),
            "selected_gradient": float(gradients[max_index]),
            "max_gradient": float(max_grad),
            "gradient_norm": float(norm)
        })

        # --------------------------------------------------------
        # Precomputar espectro del nuevo operador
        # --------------------------------------------------------

        spec_qutip = precompute_operator_spectra([new_op])[0]
        ansatz_spectra_qutip.append(spec_qutip)

        spec_numpy = precompute_operator_spectra_numpy([new_op])[0]
        ansatz_spectra_numpy.append(spec_numpy)

        # --------------------------------------------------------
        # Optimización variacional
        # --------------------------------------------------------

        init_theta = params + [0.0]

        result = minimize(
            cost_function_fast_numpy,
            init_theta,
            args=(ansatz_spectra_numpy, initial_state_vec, Hf_mat),
            method="BFGS",
            options={"maxiter": 1000}
        )

        params = list(map(float, result.x))

        energy_after_optimization = float(np.real(result.fun))
        energy_trace.append(energy_after_optimization)

        optimizer_trace.append({
            "iteration": int(iteration + 1),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "fun": energy_after_optimization,
            "nfev": int(getattr(result, "nfev", -1)),
            "njev": int(getattr(result, "njev", -1)),
            "nit": int(getattr(result, "nit", -1)),
            "params": [float(p) for p in params]
        })

        if show:
            print(
                f"Iteración {iteration + 1} | "
                f"Energía: {energy_after_optimization:.10f} | "
                f"Grad norm: {norm:.5e} | "
                f"Operador: {max_index}"
            )

        iteration += 1

    # ============================================================
    # Reoptimización final
    # ============================================================

    if len(params) > 0:
        result = minimize(
            cost_function_fast_numpy,
            params,
            args=(ansatz_spectra_numpy, initial_state_vec, Hf_mat),
            method="BFGS",
            options={"maxiter": 100}
        )

        params = list(map(float, result.x))

        optimizer_success = bool(result.success)
        optimizer_status = int(result.status)
        optimizer_message = str(result.message)
        optimizer_nfev = int(getattr(result, "nfev", -1))
        optimizer_njev = int(getattr(result, "njev", -1))
        optimizer_nit = int(getattr(result, "nit", -1))
        optimizer_fun = float(np.real(result.fun))

    else:
        optimizer_success = True
        optimizer_status = 0
        optimizer_message = "No optimization needed"
        optimizer_nfev = 0
        optimizer_njev = 0
        optimizer_nit = 0
        optimizer_fun = initial_problem_energy

    # ============================================================
    # Estado final y gradientes finales
    # ============================================================

    psi_final = build_ansatz_fast(params, ansatz_spectra_qutip, psi_0) if params else psi_0

    final_gradients = []

    for comm in commutators:
        grad = psi_final.dag() @ comm @ psi_final
        final_gradients.append(abs(complex(grad)))

    final_gradients = np.array(final_gradients, dtype=float)

    final_max_gradient = float(np.max(final_gradients)) if len(final_gradients) > 0 else 0.0
    final_gradient_norm = float(np.linalg.norm(final_gradients)) if len(final_gradients) > 0 else 0.0

    grad_trace.append(final_max_gradient)
    norm_grad_trace.append(final_gradient_norm)

    # Evitamos duplicar innecesariamente si la última energía ya coincide.
    energy_trace.append(optimizer_fun)

    final_energy = float(np.real(qt.expect(Hf, psi_final)))
    difference_ground = float(np.abs(final_energy - E0_f))

    runtime_seconds = float(time.time() - start_time)
    runtime_min = runtime_seconds / 60.0

    converged = bool(final_gradient_norm < epsilon)

    # ============================================================
    # Construcción explícita del ansatz final
    # ============================================================

    final_ansatz = []

    for step, (idx, label, order, theta) in enumerate(
        zip(ansatz_op_indices, ansatz_op_labels, ansatz_op_orders, params),
        start=1
    ):
        final_ansatz.append({
            "step": int(step),
            "operator_index": int(idx),
            "operator_label": str(label),
            "operator_order": int(order),
            "theta": float(theta)
        })

    # Esta forma es útil si después quieres reconstruir como circuito:
    # U(theta) = exp(-i theta_m A_m) ... exp(-i theta_1 A_1)
    circuit_reconstruction_steps = []

    for item in final_ansatz:
        circuit_reconstruction_steps.append({
            "step": item["step"],
            "apply_unitary": f"exp(-i * theta_{item['step']} * A_{item['operator_index']})",
            "operator_index": item["operator_index"],
            "operator_label": item["operator_label"],
            "theta": item["theta"]
        })

    if show:
        print("........................Fin del algoritmo profundo........................")
        print(f"Número total de operadores en el ansatz: {len(ansatz_ops)}")
        print(f"Energía final: {final_energy}")
        print(f"Diferencia con el Groundstate: {difference_ground}")
        print(f"Gradiente final: {final_gradient_norm}")
        print(f"Razón de parada: {stop_reason}")
        print(f"Tiempo de ejecución: {runtime_min:.4f} min")

    # ============================================================
    # Return completo
    # ============================================================

    return {
        "n": int(n),
        "edges": edges,
        "num_edges": int(len(edges)),
        "l": int(l),
        "epsilon": float(epsilon),
        "max_iteration": int(max_iteration),

        # --------------------------------------------------------
        # Energías importantes
        # --------------------------------------------------------
        "initial_energy": float(initial_ground_energy),
        "initial_problem_energy": float(initial_problem_energy),

        "ground_energy": float(E0_f),
        "first_excited_energy": first_excited_energy,
        "spectral_gap": spectral_gap,
        "ground_degeneracy": int(ground_degeneracy),

        "final_energy": float(final_energy),
        "difference_ground": float(difference_ground),

        # --------------------------------------------------------
        # Información del pool
        # --------------------------------------------------------
        "pool_size": int(pool_size),
        "pool_labels": operator_pool_labels,
        "pool_orders": [int(o) for o in operator_pool_orders],

        "pool_info": {
            "pool_construction": "nested_commutators(Had, dHad_dlam, order=3)",
            "l": int(l),
            "included_orders": [1] if l == 1 else [1, 3],
            "ordering_rule": "sorted unique canonical operators by str(op)",
            "important_note": (
                "El orden del pool se fijó con sorted(..., key=str). "
                "Para reconstruir el ansatz, se debe usar exactamente la misma regla."
            )
        },

        # --------------------------------------------------------
        # Información ADAPT
        # --------------------------------------------------------
        "iterations": int(iteration),
        "num_ansatz_ops": int(len(ansatz_ops)),
        "converged": bool(converged),
        "stop_reason": str(stop_reason),

        "final_gradient_norm": float(final_gradient_norm),
        "final_max_gradient": float(final_max_gradient),

        # --------------------------------------------------------
        # Parámetros y operadores seleccionados
        # --------------------------------------------------------
        "params": [float(p) for p in params],
        "ansatz_op_indices": [int(i) for i in ansatz_op_indices],
        "ansatz_op_labels": [str(label) for label in ansatz_op_labels],
        "ansatz_op_orders": [int(o) for o in ansatz_op_orders],

        # --------------------------------------------------------
        # Ansatz final explícito
        # --------------------------------------------------------
        "final_ansatz": final_ansatz,

        "ansatz_convention": (
            "psi(theta) = exp(-i theta_m A_m) ... exp(-i theta_2 A_2) "
            "exp(-i theta_1 A_1) psi_0"
        ),

        "circuit_reconstruction_steps": circuit_reconstruction_steps,

        "reconstruction_info": {
            "selected_operator_indices": [int(i) for i in ansatz_op_indices],
            "selected_operator_labels": [str(label) for label in ansatz_op_labels],
            "selected_operator_orders": [int(o) for o in ansatz_op_orders],
            "optimized_parameters": [float(p) for p in params],
            "pool_size": int(pool_size),
            "pool_labels": operator_pool_labels,
            "pool_orders": [int(o) for o in operator_pool_orders],
            "initial_state": "ground state of Hi_qutip(n, 1)",
            "problem_hamiltonian": "Hp_qutip(n, edges)",
            "note": (
                "Para reconstruir el ansatz, regenerar el mismo pool usando el mismo "
                "n, edges, l y la misma regla de ordenamiento. Luego tomar los operadores "
                "en el orden indicado por selected_operator_indices y aplicar los parámetros "
                "optimized_parameters."
            )
        },

        # --------------------------------------------------------
        # Trazas
        # --------------------------------------------------------
        "energy_trace": [float(e) for e in energy_trace],
        "grad_trace": [float(g) for g in grad_trace],
        "norm_grad_trace": [float(g) for g in norm_grad_trace],

        "selected_gradient_trace": selected_gradient_trace,
        "top_gradients_trace": top_gradients_trace,
        "optimizer_trace": optimizer_trace,

        # --------------------------------------------------------
        # Optimizador final
        # --------------------------------------------------------
        "optimizer_success": bool(optimizer_success),
        "optimizer_status": int(optimizer_status),
        "optimizer_message": str(optimizer_message),
        "optimizer_nfev": int(optimizer_nfev),
        "optimizer_njev": int(optimizer_njev),
        "optimizer_nit": int(optimizer_nit),

        # --------------------------------------------------------
        # Tiempo
        # --------------------------------------------------------
        "runtime_seconds": float(runtime_seconds),
        "runtime_min": float(runtime_min)
    }


# ============================================================
# 10. Estadísticas de convergencia sobre un ensemble de grafos
# (usada en cuadernillos/grafos_aleatorios_n6.ipynb y max3cut_n5_adapt.ipynb)
# ============================================================

def compute_error_stats(json_paths):
    """
    Calcula media/mediana/IQR de |E_k - E0| a través de todas las
    trazas de energía de uno o varios JSON completos de CD-ADAPT-VQE,
    alineadas por índice de iteración (las trazas más cortas se
    extienden repitiendo su último valor).
    """
    if isinstance(json_paths, str):
        json_paths = [json_paths]

    resultados_totales = []

    for json_path in json_paths:
        with open(json_path, "r") as f:
            resultados = json.load(f)
        resultados_totales.extend(resultados)

    curvas_error = []

    for item in resultados_totales:
        if "energy_trace" not in item or "ground_energy" not in item:
            continue

        trace = np.array(item["energy_trace"], dtype=float)
        E0 = float(item["ground_energy"])

        error_trace = np.abs(trace - E0)
        curvas_error.append(error_trace)

    if len(curvas_error) == 0:
        raise ValueError("No se encontraron curvas válidas en los archivos JSON.")

    max_len = max(len(c) for c in curvas_error)

    curvas_padded = []
    for c in curvas_error:
        if len(c) < max_len:
            c = np.pad(c, (0, max_len - len(c)), mode="edge")
        curvas_padded.append(c)

    curvas_padded = np.array(curvas_padded)

    return {
        "mean": curvas_padded.mean(axis=0),
        "median": np.median(curvas_padded, axis=0),
        "q25": np.percentile(curvas_padded, 25, axis=0),
        "q75": np.percentile(curvas_padded, 75, axis=0),
        "len": max_len
    }


# ============================================================
# 11. Presupuesto común: truncar y guardar resultados de CD-ADAPT-VQE
# (usadas en la carga de datos "presupuesto común" de los cuadernillos
# grafos_aleatorios_n6.ipynb y max3cut_n5_adapt.ipynb)
# ============================================================

def get_num_ops(result):
    """
    Número de operadores efectivamente usados en el ansatz.
    """
    if "num_ansatz_ops" in result:
        return int(result["num_ansatz_ops"])

    if "iterations" in result:
        return int(result["iterations"])

    raise KeyError("No se encontró 'num_ansatz_ops' ni 'iterations'.")


def truncate_trace_like_normal(trace, r_common):
    """
    En los JSON normales de cd_adapt_vqe_algorithm_profundo:
        iterations = r
        len(energy_trace) = r + 2

    Además, el último valor suele ser una repetición de la energía final.

    Entonces, para cortar en r_common operadores:
        trace_controlada = trace[:r_common + 1] + [trace[r_common]]
    """
    if not isinstance(trace, list):
        return trace

    if len(trace) == 0:
        return trace

    if r_common >= len(trace):
        raise ValueError(
            f"No se puede cortar en r={r_common}; len(trace)={len(trace)}"
        )

    return trace[:r_common + 1] + [trace[r_common]]


def truncate_ansatz_list(value, r_common):
    """
    params, ansatz_op_indices y ansatz_op_labels deben quedar con largo r_common.
    """
    if isinstance(value, list):
        return value[:r_common]

    return value


def truncate_result_to_budget(result, r_common):
    """
    Corta un resultado de l=1 o l=2 hasta r_common operadores,
    manteniendo la misma estructura del JSON normal.
    """
    controlled = copy.deepcopy(result)

    energy_trace = controlled["energy_trace"]
    grad_trace = controlled["grad_trace"]
    norm_grad_trace = controlled["norm_grad_trace"]

    if r_common >= len(energy_trace):
        raise ValueError(
            f"Grafo {controlled.get('grafo_id')}: "
            f"r_common={r_common}, len(energy_trace)={len(energy_trace)}"
        )

    if r_common >= len(grad_trace):
        raise ValueError(
            f"Grafo {controlled.get('grafo_id')}: "
            f"r_common={r_common}, len(grad_trace)={len(grad_trace)}"
        )

    if r_common >= len(norm_grad_trace):
        raise ValueError(
            f"Grafo {controlled.get('grafo_id')}: "
            f"r_common={r_common}, len(norm_grad_trace)={len(norm_grad_trace)}"
        )

    final_energy = float(energy_trace[r_common])
    ground_energy = float(controlled["ground_energy"])
    difference_ground = abs(final_energy - ground_energy)

    final_max_gradient = float(grad_trace[r_common])
    final_gradient_norm = float(norm_grad_trace[r_common])

    controlled["max_iteration"] = int(r_common)
    controlled["final_energy"] = final_energy
    controlled["difference_ground"] = difference_ground
    controlled["iterations"] = int(r_common)
    controlled["num_ansatz_ops"] = int(r_common)
    controlled["final_gradient_norm"] = final_gradient_norm
    controlled["final_max_gradient"] = final_max_gradient

    # Cortar trazas manteniendo la misma forma normal: longitud = iterations + 2
    controlled["energy_trace"] = truncate_trace_like_normal(
        controlled["energy_trace"], r_common
    )
    controlled["grad_trace"] = truncate_trace_like_normal(
        controlled["grad_trace"], r_common
    )
    controlled["norm_grad_trace"] = truncate_trace_like_normal(
        controlled["norm_grad_trace"], r_common
    )

    # Cortar listas del ansatz: longitud = iterations
    controlled["params"] = truncate_ansatz_list(
        controlled.get("params", []), r_common
    )
    controlled["ansatz_op_indices"] = truncate_ansatz_list(
        controlled.get("ansatz_op_indices", []), r_common
    )
    controlled["ansatz_op_labels"] = truncate_ansatz_list(
        controlled.get("ansatz_op_labels", []), r_common
    )

    # Nota: runtime y datos del optimizador quedan como los originales.
    # No se pueden reconstruir exactamente sin volver a correr.
    return controlled


def save_json(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def save_csv(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    csv_fieldnames = [
        "grafo_id",
        "num_edges",
        "ground_energy",
        "iteraciones",
        "energia_final",
        "diferencia_ground",
        "gradiente_final",
        "runtime_min",
        "initial_energy"
    ]

    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()

        for result in results:
            if "error" in result:
                writer.writerow({
                    "grafo_id": result.get("grafo_id", "ERROR"),
                    "num_edges": "ERROR",
                    "ground_energy": "ERROR",
                    "iteraciones": "ERROR",
                    "energia_final": "ERROR",
                    "diferencia_ground": "ERROR",
                    "gradiente_final": "ERROR",
                    "runtime_min": "ERROR",
                    "initial_energy": "ERROR"
                })
                continue

            writer.writerow({
                "grafo_id": result["grafo_id"],
                "num_edges": result["num_edges"],
                "ground_energy": result["ground_energy"],
                "iteraciones": result["iterations"],
                "energia_final": result["final_energy"],
                "diferencia_ground": result["difference_ground"],
                "gradiente_final": result["final_gradient_norm"],
                "runtime_min": result["runtime_min"],
                "initial_energy": result["initial_energy"]
            })