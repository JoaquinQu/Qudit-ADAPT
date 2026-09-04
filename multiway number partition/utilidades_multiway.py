"""
Motor de CD-ADAPT-VQE y QAOA para multiway number partitioning (k=3) con
qutrits. Sigue la formulacion de formulacion_qubo.tex (misma carpeta):
operador digito d_j = J_{z,j} + I (autovalores {0,1,2}, ya definido en
funciones/utilidades_factorizacion.py), proyectores de etiqueta Pi_i(d_j)
via interpolacion de Lagrange, Sigma_i = sum_j a_j Pi_i(d_j), y
H_p = sum_i (Sigma_i - mu*I)^2 con mu = (1/3) sum_j a_j.

Todo lo que no es especifico de este problema (Hi_qutip, el motor de
conmutadores anidados, el bucle ADAPT generico, el optimizador QAOA) se
importa sin cambios desde funciones/. Vive en esta carpeta (no en
funciones/) porque todavia es exploratorio -- ver la carpeta
"multiway number partition/".
"""

import json
import random
import sys
import time
from pathlib import Path

import cma
import numpy as np
import qutip as qt
import sympy as sp
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funciones.utilidades import (
    Hi_qutip,
    Jz_site,
    expr_from_monomial,
    add_expr,
    scale_expr,
    sum_expr,
    canonical_op,
    pool_to_qutip,
    nested_commutators,
    Hi as Hi_symbolic,
    precompute_operator_spectra,
    build_ansatz_fast,
    precompute_operator_spectra_numpy,
    cost_function_fast_numpy,
    to_jsonable,
)
from funciones.utilidades_factorizacion import (
    digit_operator,
    digit_expr,
    mult_expr_expr,
    initial_state,
    cd_adapt_vqe_factorizacion as cd_adapt_vqe_multiway,
)
from funciones.utilidades_QAOA import (
    Hm_qaoa_jx,
    Hm_qaoa_x_custom,
    optimize_qaoa_for_p,
    qaoa_energy_numpy,
    qaoa_relative_error,
    expand_params_previous_p,
)

# cd_adapt_vqe_multiway = cd_adapt_vqe_factorizacion (reimportado con otro
# nombre): esa funcion ya es 100% generica -- recibe Hf, psi_0, pool y
# labels ya armados, no tiene nada especifico de factorizacion adentro
# (a diferencia de cd_adapt_vqe_algorithm, que construye Hf desde
# (n, edges) y esta hardcodeada a Max-3-Cut). No hace falta forkearla de
# nuevo, alcanza con reusarla bajo un nombre mas claro para este notebook.


# ============================================================
# 1. Proyectores de etiqueta Pi_i(d), numericos y simbolicos
# ============================================================

def proyector_etiqueta(n, site, i):
    """
    Pi_i(d_site), operador numerico (qutip), ver formulacion_qubo.tex S3.2:
        Pi_0(d) = (1/2)(d-I)(d-2I)
        Pi_1(d) = -d(d-2I)
        Pi_2(d) = (1/2) d(d-I)
    Autovalores {0,1} -- proyector ortogonal sobre "el elemento site tiene
    etiqueta i".
    """
    if i not in (0, 1, 2):
        raise ValueError("i debe ser 0, 1 o 2")

    d = digit_operator(n, site)
    I_n = qt.tensor([qt.qeye(3) for _ in range(n)])

    if i == 0:
        return 0.5 * (d - I_n) * (d - 2 * I_n)
    elif i == 1:
        return -d * (d - 2 * I_n)
    else:
        return 0.5 * d * (d - I_n)


def proyector_etiqueta_expr(site, i):
    """Version simbolica (expr dict) de proyector_etiqueta, para nested_commutators."""
    if i not in (0, 1, 2):
        raise ValueError("i debe ser 0, 1 o 2")

    d = digit_expr(site)
    I_e = expr_from_monomial((), 1)

    if i == 0:
        d_menos_I = add_expr(d, scale_expr(-1, I_e))
        d_menos_2I = add_expr(d, scale_expr(-2, I_e))
        return scale_expr(sp.Rational(1, 2), mult_expr_expr(d_menos_I, d_menos_2I))
    elif i == 1:
        d_menos_2I = add_expr(d, scale_expr(-2, I_e))
        return scale_expr(-1, mult_expr_expr(d, d_menos_2I))
    else:
        d_menos_I = add_expr(d, scale_expr(-1, I_e))
        return scale_expr(sp.Rational(1, 2), mult_expr_expr(d, d_menos_I))


# ============================================================
# 2. Sigma_i = sum_j a_j Pi_i(d_j), numerico y simbolico
# ============================================================

def Sigma_operator(n, a, i):
    assert len(a) == n, "a debe tener n elementos"
    H = 0
    for j, aj in enumerate(a, start=1):
        H += aj * proyector_etiqueta(n, j, i)
    return H


def Sigma_expr(a, i):
    terms = [
        scale_expr(aj, proyector_etiqueta_expr(j, i))
        for j, aj in enumerate(a, start=1)
    ]
    return sum_expr(terms)


# ============================================================
# 3. H_p = sum_i (Sigma_i - mu*I)^2, numerico y simbolico
# ============================================================

def Hp_multiway(n, a):
    """
    H_p para multiway number partitioning, k=3, ver formulacion_qubo.tex
    S3.3. mu = (1/3) sum(a) es el promedio ideal, fijo (no depende de z).
    """
    assert len(a) == n, "a debe tener n elementos"
    mu = sum(a) / 3.0
    I_n = qt.tensor([qt.qeye(3) for _ in range(n)])

    H = 0
    for i in range(3):
        Sigma_i = Sigma_operator(n, a, i)
        dev = Sigma_i - mu * I_n
        H += dev * dev
    return H


def Hp_multiway_symbolic(a):
    mu = sp.Rational(sum(a), 3)
    I_e = expr_from_monomial((), 1)

    terms = []
    for i in range(3):
        Sigma_i = Sigma_expr(a, i)
        dev = add_expr(Sigma_i, scale_expr(-mu, I_e))
        terms.append(mult_expr_expr(dev, dev))
    return sum_expr(terms)


# ============================================================
# 4. H_ad(lambda), dH_ad/dlambda y pool CD (nested_commutators)
# ============================================================

def Had_multiway(n, a, lam):
    hi = Hi_symbolic(n)
    hp = Hp_multiway_symbolic(a)
    return add_expr(scale_expr(1 - lam, hi), scale_expr(lam, hp))


def dHad_dlam_multiway(n, a):
    hi = Hi_symbolic(n)
    hp = Hp_multiway_symbolic(a)
    return add_expr(scale_expr(-1, hi), hp)


def build_pool_multiway(n, a, l=1, show=False):
    """
    Pool CD = union de O1 (y O3 si l=2) de nested_commutators(Had, dHad_dlam).
    Igual patron que cd_adapt_vqe_algorithm/build_pool de factorizacion,
    pero usando el motor simbolico sympy de utilidades.py directamente
    (no el motor nativo rapido de factorizacion) -- para n<=5 alcanza sin
    necesidad de esa optimizacion extra.
    """
    lam = sp.symbols("lam", real=True)
    H = Had_multiway(n, a, lam)
    dH = dHad_dlam_multiway(n, a)

    t0 = time.time()
    results = nested_commutators(H, dH, order=3 if l == 2 else 1)
    if show:
        print(f"nested_commutators (orden {'3' if l==2 else '1'}): {time.time()-t0:.2f}s")

    O1 = results[1]
    pool_1 = list({canonical_op(op) for op in O1.keys()})
    labels_1 = [str(op) for op in pool_1]
    qutip_pool_1 = pool_to_qutip(pool_1, n)

    if l == 1:
        return qutip_pool_1, labels_1

    if l == 2:
        O3 = results[3]
        pool_3 = list({canonical_op(op) for op in O3.keys()})
        labels_3 = [str(op) for op in pool_3]
        qutip_pool_3 = pool_to_qutip(pool_3, n)
        return qutip_pool_1 + qutip_pool_3, labels_1 + labels_3

    raise ValueError("l debe ser 1 o 2")


# ============================================================
# 5. QAOA: fork minimo de preparar_qaoa_para_grafo/scan_qaoa_p que
# recibe Hp_multiway ya armado en vez de construirlo desde edges.
# El resto del motor (optimize_qaoa_for_p, Hm_qaoa_jx/custom,
# qaoa_energy_numpy adentro de optimize_qaoa_for_p) se reutiliza sin
# ningun cambio.
# ============================================================

def preparar_qaoa_multiway(n, Hc, mixer="jx"):
    if mixer == "jx":
        Hm = Hm_qaoa_jx(n)
    elif mixer == "custom":
        Hm = Hm_qaoa_x_custom(n)
    else:
        raise ValueError("mixer debe ser 'jx' o 'custom'.")

    if not Hc.isherm:
        raise ValueError("Hc no es hermitico.")
    if not Hm.isherm:
        raise ValueError("Hm no es hermitico.")

    evals_c = Hc.eigenenergies()
    E0 = float(np.min(evals_c))

    Hi_uniform = Hi_qutip(n, omega0=1)
    _, evecs_i = Hi_uniform.eigenstates()
    psi0 = evecs_i[0]

    initial_mixer_energy = float(np.real(qt.expect(Hm, psi0)))
    initial_problem_energy = float(np.real(qt.expect(Hc, psi0)))

    spec_Hc = precompute_operator_spectra_numpy([Hc])[0]
    spec_Hm = precompute_operator_spectra_numpy([Hm])[0]

    return {
        "n": int(n),
        "mixer": mixer,
        "Hc": Hc,
        "Hm": Hm,
        "ground_energy": E0,
        "initial_mixer_energy": initial_mixer_energy,
        "initial_problem_energy": initial_problem_energy,
        "psi0": psi0,
        "Hc_mat": Hc.full(),
        "psi0_vec": psi0.full().ravel(),
        "spec_Hc": spec_Hc,
        "spec_Hm": spec_Hm,
    }


def optimize_qaoa_for_p_cma(
    p,
    Hc_mat,
    spec_Hc,
    spec_Hm,
    psi0_vec,
    ground_energy,
    best_params_previous=None,
    num_restarts=15,
    maxiter=500,
    seed=123,
    bounds_scale=np.pi,
    use_warmstart=True,
    show=True,
):
    """
    Igual estructura de entrada/salida que optimize_qaoa_for_p
    (funciones/utilidades_QAOA.py), pero con CMA-ES en vez de L-BFGS-B --
    el optimizador que Deller et al. recomiendan para estos paisajes
    multimodales: "the CMA-ES is a population-based global optimizer
    capable of dealing with this cost function landscape and finds lower
    cost minima more reliably" (comparado con L-BFGS, que "may end up in
    a local minimum with a high probability").

    maxiter aca es maxfevals por restart (limite de evaluaciones de
    energia dentro de una corrida de CMA-ES), no "iteraciones" de scipy.
    """
    rng = np.random.default_rng(seed)
    num_params = 2 * p

    initial_points = []

    warm_start = None
    if use_warmstart:
        warm_start = expand_params_previous_p(best_params_previous, p)
    if warm_start is not None:
        initial_points.append(("warm", warm_start))

    for r in range(num_restarts):
        x0 = rng.uniform(low=-bounds_scale, high=bounds_scale, size=num_params)
        initial_points.append((f"random_{r + 1}", x0))

    def objective(x):
        return qaoa_energy_numpy(x, Hc_mat, spec_Hc, spec_Hm, psi0_vec, p)

    best_energy = np.inf
    best_params = None
    best_init_name = None
    total_evals = 0

    all_runs = []

    for init_name, x0 in initial_points:
        sigma0 = bounds_scale / 2.0

        es = cma.CMAEvolutionStrategy(
            list(x0), sigma0,
            {
                "bounds": [-bounds_scale, bounds_scale],
                "maxfevals": maxiter,
                "seed": int(rng.integers(1, 2 ** 31 - 1)),
                "verbose": -9,
            },
        )
        es.optimize(objective)

        energy = float(es.result.fbest)
        params = np.array(es.result.xbest, dtype=float)
        n_evals = int(es.result.evaluations)
        total_evals += n_evals

        rel_error = qaoa_relative_error(energy, ground_energy)

        run_data = {
            "init_name": init_name,
            "energy": energy,
            "relative_error": float(rel_error),
            "success": True,
            "status": 0,
            "message": "CMA-ES",
            "nfev": n_evals,
            "nit": int(es.result.iterations),
            "params": params,
        }
        all_runs.append(run_data)

        if energy < best_energy:
            best_energy = energy
            best_params = params
            best_init_name = init_name

        if show:
            print(
                f"p={p:02d} | init={init_name:>9s} | "
                f"E={energy:.10f} | err_rel={rel_error:.6e} | evals={n_evals}"
            )

    best_relative_error = qaoa_relative_error(best_energy, ground_energy)

    return {
        "p": int(p),
        "num_params": int(num_params),

        "best_energy": float(best_energy),
        "best_relative_error": float(best_relative_error),
        "best_params": best_params,

        "best_init_name": str(best_init_name),

        "optimizer_success": True,
        "optimizer_status": 0,
        "optimizer_message": "CMA-ES",
        "optimizer_nfev": int(total_evals),
        "optimizer_nit": -1,

        "num_restarts": int(num_restarts),
        "maxiter": int(maxiter),
        "method": "CMA-ES",
        "bounds_scale": float(bounds_scale),
        "used_warmstart": bool(warm_start is not None),

        "all_runs": all_runs,
    }


def scan_qaoa_p_multiway(
    n,
    Hc,
    p_max=8,
    mixer="jx",
    num_restarts=15,
    maxiter=300,
    seed=123,
    bounds_scale=np.pi,
    method="L-BFGS-B",
    use_warmstart=True,
    show=True,
):
    data = preparar_qaoa_multiway(n, Hc, mixer=mixer)

    Hc_mat = data["Hc_mat"]
    spec_Hc = data["spec_Hc"]
    spec_Hm = data["spec_Hm"]
    psi0_vec = data["psi0_vec"]
    ground_energy = data["ground_energy"]

    results = []
    best_params_previous = None

    for p in range(1, p_max + 1):
        if show:
            print(f"\nQAOA multiway | p={p} | num_params={2*p} | mixer={mixer}")

        t0 = time.time()

        if method == "CMA-ES":
            result_p = optimize_qaoa_for_p_cma(
                p=p,
                Hc_mat=Hc_mat,
                spec_Hc=spec_Hc,
                spec_Hm=spec_Hm,
                psi0_vec=psi0_vec,
                ground_energy=ground_energy,
                best_params_previous=best_params_previous,
                num_restarts=num_restarts,
                maxiter=maxiter,
                seed=seed + p,
                bounds_scale=bounds_scale,
                use_warmstart=use_warmstart,
                show=show,
            )
        else:
            result_p = optimize_qaoa_for_p(
                p=p,
                Hc_mat=Hc_mat,
                spec_Hc=spec_Hc,
                spec_Hm=spec_Hm,
                psi0_vec=psi0_vec,
                ground_energy=ground_energy,
                best_params_previous=best_params_previous,
                num_restarts=num_restarts,
                maxiter=maxiter,
                seed=seed + p,
                bounds_scale=bounds_scale,
                method=method,
                use_warmstart=use_warmstart,
                show=show,
            )

        runtime_min = (time.time() - t0) / 60.0

        result_p.update({
            "n": int(n),
            "mixer": str(mixer),
            "ground_energy": float(ground_energy),
            "initial_problem_energy": float(data["initial_problem_energy"]),
            "initial_mixer_energy": float(data["initial_mixer_energy"]),
            "absolute_error": float(abs(result_p["best_energy"] - ground_energy)),
            "relative_error": float(result_p["best_relative_error"]),
            "runtime_min": float(runtime_min),
        })

        results.append(result_p)
        best_params_previous = result_p["best_params"]

    return results


# ============================================================
# 6. Decodificacion: leer una particion desde un estado computacional
# ============================================================

def decode_partition(digits, a):
    """digits[j] in {0,1,2} es la parte del elemento j (a[j]). Retorna
    (Sigma_0, Sigma_1, Sigma_2)."""
    sumas = [0.0, 0.0, 0.0]
    for aj, zj in zip(a, digits):
        sumas[zj] += aj
    return tuple(sumas)


def imbalance(sumas):
    return max(sumas) - min(sumas)


def top_partitions(psi_final, n, a, top_k=5):
    """
    Mide psi_final en la base computacional y devuelve las top_k
    asignaciones mas probables, decodificadas a sumas por parte.
    Misma convencion de indice->digito que top_candidates en
    utilidades_factorizacion.py (digito = 2 - i).
    """
    probs = np.abs(psi_final.full().ravel()) ** 2
    top_idx = np.argsort(probs)[::-1][:top_k]

    out = []
    for idx in top_idx:
        digits = tuple(int(2 - i) for i in np.unravel_index(idx, [3] * n))
        sumas = decode_partition(digits, a)
        out.append({
            "digits": digits,
            "sumas": sumas,
            "imbalance": imbalance(sumas),
            "prob": float(probs[idx]),
        })
    return out


# ============================================================
# 7. Generacion de instancias de prueba
# ============================================================
#
# Resuelve (con un parametro, no hardcodeado) el punto abierto de
# formulacion_qubo.tex S6 "Generacion de instancias": por defecto los
# numeros se muestrean genericos, SIN forzar que exista una particion
# perfecta (H_p=0). Esto no complica nada rio abajo -- E0 siempre se
# obtiene por diagonalizacion exacta (barata para n=5,6: dim = 3**n =
# 243, 729), exista o no una particion balanceada real. Con
# forzar_divisible_3=True se puede pedir la version mas facil (existencia
# garantizada de particion perfecta), como en el toy de
# comparacion_multiway.ipynb.

def generar_instancias_multiway(
    n,
    num_instancias,
    valor_min=1,
    valor_max=15,
    forzar_divisible_3=False,
    seed=None,
    filename=None,
):
    """
    Genera num_instancias conjuntos de n enteros positivos (uno por
    qutrit), cada uno una instancia de multiway number partitioning
    (k=3) de n qutrits.

    forzar_divisible_3=True: resamplea el ultimo numero de cada instancia
    hasta que sum(a) sea multiplo de 3 -- condicion necesaria (no
    suficiente) para que exista una particion perfecta en 3 partes de
    igual suma. False (default): instancias genericas, sin garantia de
    particion perfecta.

    Si filename no es None, ademas guarda las instancias (una por linea,
    enteros separados por coma) en esa ruta -- mismo formato que
    datos/grafos_n6.txt en la raiz del proyecto (una instancia por
    linea), pero para listas de enteros en vez de aristas.
    """
    rng = random.Random(seed)
    intentos_max_por_instancia = 1000

    instancias = []
    for _ in range(num_instancias):
        a = [rng.randint(valor_min, valor_max) for _ in range(n)]

        if forzar_divisible_3:
            intentos = 0
            while sum(a) % 3 != 0 and intentos < intentos_max_por_instancia:
                a[-1] = rng.randint(valor_min, valor_max)
                intentos += 1
            if sum(a) % 3 != 0:
                raise RuntimeError(
                    f"No se logro sum(a) % 3 == 0 en {intentos_max_por_instancia} intentos"
                )

        instancias.append(a)

    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for a in instancias:
                f.write(", ".join(str(x) for x in a) + "\n")

    return instancias


def _generar_instancia_balanceada(n, valor_min, valor_max, rng, intentos_max=2000):
    """
    Construye una instancia de n enteros con particion perfecta GARANTIZADA:
    a diferencia de forzar_divisible_3=True (necesario pero no suficiente,
    ver nota de generar_instancias_multiway), esta arma la instancia AL
    REVES -- reparte n en 3 partes de tamanio >=1, genera cada parte para
    que sume exactamente el mismo S, y devuelve los n numeros mezclados.
    Por construccion existe z con Sigma_0(z)=Sigma_1(z)=Sigma_2(z)=S, o sea
    H_p(z)=0 (ver formulacion_qubo.tex S3.3).
    """
    for _ in range(intentos_max):
        tamanios = [1, 1, 1]
        for _ in range(n - 3):
            tamanios[rng.randrange(3)] += 1

        hay_singleton = any(k == 1 for k in tamanios)
        k_max = max(tamanios)
        lo = valor_min * k_max
        hi = valor_max if hay_singleton else valor_max * min(tamanios)
        if lo > hi:
            continue
        S = rng.randint(lo, hi)

        partes, ok = [], True
        for k in tamanios:
            if k == 1:
                if not (valor_min <= S <= valor_max):
                    ok = False
                    break
                partes.append([S])
                continue

            parte = None
            for _ in range(200):
                valores = [rng.randint(valor_min, valor_max) for _ in range(k - 1)]
                ultimo = S - sum(valores)
                if valor_min <= ultimo <= valor_max:
                    parte = valores + [ultimo]
                    break
            if parte is None:
                ok = False
                break
            partes.append(parte)

        if ok:
            a = [x for parte in partes for x in parte]
            rng.shuffle(a)
            return a

    raise RuntimeError(f"No se pudo generar instancia balanceada para n={n} tras {intentos_max} intentos")


def generar_instancias_multiway_balanceadas(
    n,
    num_instancias,
    valor_min=1,
    valor_max=15,
    seed=None,
    filename=None,
):
    """
    Como generar_instancias_multiway pero con particion perfecta
    garantizada por construccion (ver _generar_instancia_balanceada) --
    no solo sum(a) multiplo de 3, que es necesario pero no suficiente.
    Mismo patron de a=[1,2,3,4,5] del toy de comparacion_multiway.ipynb,
    generalizado y aleatorizado.
    """
    rng = random.Random(seed)
    instancias = [
        _generar_instancia_balanceada(n, valor_min, valor_max, rng)
        for _ in range(num_instancias)
    ]

    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for a in instancias:
                f.write(", ".join(str(x) for x in a) + "\n")

    return instancias


def leer_instancias_multiway(path):
    """
    Lee un archivo .txt donde cada linea es una instancia: enteros
    separados por coma. Mismo patron que leer_grafos en
    funciones/utilidades_QAOA.py, adaptado a listas de enteros.
    """
    path = Path(path)
    instancias = []

    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            a = [int(x.strip()) for x in linea.split(",")]
            instancias.append(a)

    return instancias


# ============================================================
# 8. Cache de resultados a JSON (para notebooks tipo
# comparacion_multiway.ipynb: correr una vez, despues cargar del disco
# controlando todo con un solo booleano en vez de recalcular siempre).
# ============================================================

def guardar_resultado_adapt(res, path):
    """Guarda el dict de cd_adapt_vqe_multiway a JSON. psi_final (Qobj) no es
    serializable por to_jsonable, se guarda aparte como vector complejo + dims."""
    psi = res["psi_final"]
    data = to_jsonable({k: v for k, v in res.items() if k != "psi_final"})
    psi_vec = psi.full().ravel()
    data["psi_final_real"] = psi_vec.real.tolist()
    data["psi_final_imag"] = psi_vec.imag.tolist()
    data["psi_final_dims"] = psi.dims
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def cargar_resultado_adapt(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    psi_vec = np.array(data.pop("psi_final_real")) + 1j * np.array(data.pop("psi_final_imag"))
    dims = data.pop("psi_final_dims")
    data["psi_final"] = qt.Qobj(psi_vec.reshape(-1, 1), dims=dims)
    return data


def guardar_resultado_qaoa(qaoa_results, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(qaoa_results), f, indent=2, ensure_ascii=False)


def cargar_resultado_qaoa(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        item["best_params"] = np.array(item["best_params"])
        for run in item.get("all_runs", []):
            if "params" in run:
                run["params"] = np.array(run["params"])
    return data


def cachear_adapt(path, recompute, fn):
    """Si recompute es False y existe `path`, carga el resultado cacheado.
    Si no, corre `fn()` (sin argumentos, tipicamente una closure que llama a
    cd_adapt_vqe_multiway) y guarda el resultado en `path`."""
    path = Path(path)
    if not recompute and path.exists():
        return cargar_resultado_adapt(path), True
    path.parent.mkdir(parents=True, exist_ok=True)
    res = fn()
    guardar_resultado_adapt(res, path)
    return res, False


def cachear_qaoa(path, recompute, fn):
    """Igual que cachear_adapt, para el resultado de scan_qaoa_p_multiway."""
    path = Path(path)
    if not recompute and path.exists():
        return cargar_resultado_qaoa(path), True
    path.parent.mkdir(parents=True, exist_ok=True)
    res = fn()
    guardar_resultado_qaoa(res, path)
    return res, False
