"""
Qudit QAOA: the fixed-ansatz baseline that Qudit-ADAPT is compared against.

The ansatz alternates p layers of the cost and mixer Hamiltonians,

    |psi(gamma, beta)> = prod_k exp(-i beta_k H_M) exp(-i gamma_k H_C) |phi_g>,

so it carries 2p variational parameters. Both Hamiltonians are reused from
`utilidades.py`, which is what makes the comparison fair: the two algorithms
solve the same problem with the same operators and differ only in how the
ansatz is chosen.

`scan_qaoa_p` sweeps the depth p and, at each one, optimizes from a warm start
expanded from p-1 plus `num_restarts` random initializations, keeping the best.
The paper uses p up to 20 with 25 restarts, and reports the median over
restarts with the interquartile range as a band, because QAOA is far more
sensitive to initialization than the adaptive construction.

The section at the end (`curva_qaoa_con_banda_desde_all_runs` and friends) is
what `cuadernillos/comparacion_QAOA.ipynb` uses to draw Figs. 1-3.
"""

import ast
import ast
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import qutip as qt
from scipy.optimize import minimize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from functions.utilidades import (
    Hp_qutip,
    Jx_site,
    Hi_qutip,
    precompute_operator_spectra_numpy,
    apply_exp_to_state_from_spectrum_numpy,
    to_jsonable,
)


# ============================================================
# 1. Reading graphs
# ============================================================

def leer_grafos(path):
    """
    Reads a .txt file where each line represents a graph.

    Expected format per line:
        (1, 2), (1, 3), (2, 3)

    Returns
    -------
    grafos : list[list[tuple[int, int]]]
        List of graphs. Each graph is a list of edges.
    """
    path = Path(path)
    grafos = []

    with open(path, "r", encoding="utf-8") as f:
        lineas = f.readlines()

    for linea in lineas:
        linea = linea.strip()

        if not linea:
            continue

        edges = ast.literal_eval("[" + linea + "]")
        grafos.append(edges)

    return grafos


# ============================================================
# 2. QAOA/QOAO Hamiltonians
# ============================================================

def Hm_qaoa_jx(n):
    """
    QAOA/QOAO mixer for qutrits using angular momentum operators:

        H_M = sum_j Jx_j

    This is the mixer closest to the QAOA-for-qudits paper.
    """
    Hm = 0

    for j in range(1, n + 1):
        Hm += Jx_site(n, j)

    return Hm


def Hm_qaoa_x_custom(n):
    """
    Alternative mixer using the X_local operator from the original implementation.

    Since Hi_qutip(n, 1) = - sum_j X_j,
    then:

        H_M = sum_j X_j = -Hi_qutip(n, 1)

    This version is useful for comparing against the mixer already used in CD-ADAPT.
    """
    return -Hi_qutip(n, omega0=1)


def preparar_qaoa_para_grafo(n, edges, mixer="jx"):
    """
    Builds all the objects needed to run QAOA/QOAO on a given graph.

    Parameters
    ----------
    n : int
        Number of nodes/qutrits.
    edges : list[tuple[int, int]]
        List of edges with nodes numbered from 1.
    mixer : str
        "jx"      -> H_M = sum_j Jx_j
        "custom"  -> H_M = sum_j X_j, using Hi_qutip

    Returns
    -------
    data : dict
        Dictionary with Hc, Hm, E0, psi0, spectra, and matrices.

    Notes
    -----
    The initial state psi0 is always the uniform superposition
    (ground state of -sum_j X_j, the same Hi_qutip used in
    CD-ADAPT-VQE), regardless of the mixer chosen for the QAOA layers.
    """
    Hc = Hp_qutip(n, edges)

    if mixer == "jx":
        Hm = Hm_qaoa_jx(n)
    elif mixer == "custom":
        Hm = Hm_qaoa_x_custom(n)
    else:
        raise ValueError("mixer must be 'jx' or 'custom'.")

    if not Hc.isherm:
        raise ValueError("Hc is not Hermitian.")

    if not Hm.isherm:
        raise ValueError("Hm is not Hermitian.")

    evals_c = Hc.eigenenergies()
    E0 = float(np.min(evals_c))

    Hi_uniform = Hi_qutip(n, omega0=1)
    _, evecs_i = Hi_uniform.eigenstates()

    psi0 = evecs_i[0]
    initial_mixer_energy = float(np.real(qt.expect(Hm, psi0)))
    initial_problem_energy = float(np.real(qt.expect(Hc, psi0)))

    spec_Hc = precompute_operator_spectra_numpy([Hc])[0]
    spec_Hm = precompute_operator_spectra_numpy([Hm])[0]

    Hc_mat = Hc.full()
    psi0_vec = psi0.full().ravel()

    return {
        "n": int(n),
        "edges": edges,
        "num_edges": int(len(edges)),
        "mixer": mixer,

        "Hc": Hc,
        "Hm": Hm,

        "ground_energy": float(E0),
        "initial_mixer_energy": float(initial_mixer_energy),
        "initial_problem_energy": float(initial_problem_energy),

        "psi0": psi0,
        "Hc_mat": Hc_mat,
        "psi0_vec": psi0_vec,

        "spec_Hc": spec_Hc,
        "spec_Hm": spec_Hm,
    }


# ============================================================
# 3. QAOA/QOAO state and energy
# ============================================================

def qaoa_qudit_state_numpy(params, spec_Hc, spec_Hm, psi0_vec, p):
    """
    Builds the QAOA/QOAO state for qutrits.

    Convention:

        params = [gamma_1, ..., gamma_p, beta_1, ..., beta_p]

    and

        |psi> = exp(-i beta_p H_M) exp(-i gamma_p H_C)
                ...
                exp(-i beta_1 H_M) exp(-i gamma_1 H_C) |psi0>
    """
    gammas = params[:p]
    betas = params[p:]

    psi = psi0_vec.copy()

    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]

        psi = apply_exp_to_state_from_spectrum_numpy(
            gamma,
            spec_Hc,
            psi
        )

        psi = apply_exp_to_state_from_spectrum_numpy(
            beta,
            spec_Hm,
            psi
        )

    return psi


def qaoa_energy_numpy(params, Hc_mat, spec_Hc, spec_Hm, psi0_vec, p):
    """
    Computes the variational energy:

        E(params) = <psi(params)|Hc|psi(params)>
    """
    psi = qaoa_qudit_state_numpy(
        params=params,
        spec_Hc=spec_Hc,
        spec_Hm=spec_Hm,
        psi0_vec=psi0_vec,
        p=p
    )

    energy = np.vdot(psi, Hc_mat @ psi).real

    return float(energy)


def qaoa_relative_error(energy, ground_energy):
    """
    Relative error with respect to the exact ground energy.
    """
    if abs(ground_energy) < 1e-12:
        return float(abs(energy - ground_energy))

    return float(abs(energy - ground_energy) / abs(ground_energy))


# ============================================================
# 4. Warm-start between depths
# ============================================================

def expand_params_previous_p(best_params_previous, p):
    """
    Expands parameters from p-1 layers to p layers.

    Input p-1:
        [gamma_1, ..., gamma_{p-1}, beta_1, ..., beta_{p-1}]

    Output p:
        [gamma_1, ..., gamma_{p-1}, 0,
         beta_1, ..., beta_{p-1}, 0]
    """
    if best_params_previous is None:
        return None

    p_old = p - 1

    gammas_old = np.array(best_params_previous[:p_old], dtype=float)
    betas_old = np.array(best_params_previous[p_old:], dtype=float)

    gammas_new = np.concatenate([gammas_old, [0.0]])
    betas_new = np.concatenate([betas_old, [0.0]])

    return np.concatenate([gammas_new, betas_new])


# ============================================================
# 5. Optimization for fixed p
# ============================================================

def optimize_qaoa_for_p(
    p,
    Hc_mat,
    spec_Hc,
    spec_Hm,
    psi0_vec,
    ground_energy,
    best_params_previous=None,
    num_restarts=25,
    maxiter=500,
    seed=123,
    bounds_scale=np.pi,
    method="L-BFGS-B",
    use_warmstart=True,
    show=True,
):
    """
    Optimizes QAOA/QOAO for a depth p.

    Uses:
    - Warm-start from p-1, if it exists.
    - num_restarts random initial points.

    Parameters
    ----------
    p : int
        Number of QAOA layers.
    num_restarts : int
        Number of random restarts.
    maxiter : int
        Maximum number of iterations of the local optimizer.
    use_warmstart : bool
        If True, adds the best result from p-1 as the first initial point.

    Returns
    -------
    result_dict : dict
        Best result found for that p.
    """
    rng = np.random.default_rng(seed)

    num_params = 2 * p
    bounds = [(-bounds_scale, bounds_scale)] * num_params

    initial_points = []

    warm_start = None
    if use_warmstart:
        warm_start = expand_params_previous_p(best_params_previous, p)

    if warm_start is not None:
        initial_points.append(("warm", warm_start))

    for r in range(num_restarts):
        x0 = rng.uniform(
            low=-bounds_scale,
            high=bounds_scale,
            size=num_params
        )
        initial_points.append((f"random_{r + 1}", x0))

    best_energy = np.inf
    best_params = None
    best_result = None
    best_init_name = None

    all_runs = []

    for init_name, x0 in initial_points:

        result = minimize(
            qaoa_energy_numpy,
            x0,
            args=(Hc_mat, spec_Hc, spec_Hm, psi0_vec, p),
            method=method,
            bounds=bounds if method in ["L-BFGS-B", "SLSQP"] else None,
            options={
                "maxiter": maxiter,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxls": 50,
            } if method == "L-BFGS-B" else {
                "maxiter": maxiter,
            }
        )

        energy = float(result.fun)
        rel_error = qaoa_relative_error(energy, ground_energy)

        run_data = {
            "init_name": init_name,
            "energy": float(energy),
            "relative_error": float(rel_error),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(getattr(result, "nfev", -1)),
            "nit": int(getattr(result, "nit", -1)),
            "params": np.array(result.x, dtype=float),
        }

        all_runs.append(run_data)

        if energy < best_energy:
            best_energy = energy
            best_params = np.array(result.x, dtype=float)
            best_result = result
            best_init_name = init_name

        if show:
            print(
                f"p={p:02d} | init={init_name:>9s} | "
                f"E={energy:.10f} | "
                f"err_rel={rel_error:.6e} | "
                f"success={result.success}"
            )

    best_relative_error = qaoa_relative_error(best_energy, ground_energy)

    return {
        "p": int(p),
        "num_params": int(num_params),

        "best_energy": float(best_energy),
        "best_relative_error": float(best_relative_error),
        "best_params": best_params,

        "best_init_name": str(best_init_name),

        "optimizer_success": bool(best_result.success),
        "optimizer_status": int(best_result.status),
        "optimizer_message": str(best_result.message),
        "optimizer_nfev": int(getattr(best_result, "nfev", -1)),
        "optimizer_nit": int(getattr(best_result, "nit", -1)),

        "num_restarts": int(num_restarts),
        "maxiter": int(maxiter),
        "method": str(method),
        "bounds_scale": float(bounds_scale),
        "used_warmstart": bool(warm_start is not None),

        "all_runs": all_runs,
    }


# ============================================================
# 6. Sweep over p for a graph
# ============================================================

def scan_qaoa_p(
    n,
    edges,
    p_max=15,
    mixer="jx",
    num_restarts=25,
    maxiter=500,
    seed=123,
    bounds_scale=np.pi,
    method="L-BFGS-B",
    use_warmstart=True,
    show=True,
):
    """
    Runs QAOA/QOAO for p = 1, ..., p_max on a graph.

    Returns
    -------
    results : list[dict]
        List of results per depth.
    """
    data = preparar_qaoa_para_grafo(
        n=n,
        edges=edges,
        mixer=mixer
    )

    Hc_mat = data["Hc_mat"]
    spec_Hc = data["spec_Hc"]
    spec_Hm = data["spec_Hm"]
    psi0_vec = data["psi0_vec"]
    ground_energy = data["ground_energy"]

    results = []
    best_params_previous = None

    for p in range(1, p_max + 1):

        if show:
            print("\n" + "=" * 80)
            print(
                f"QAOA/QOAO | p = {p} | "
                f"num_params = {2 * p} | mixer = {mixer}"
            )
            print("=" * 80)

        start_time = time.time()

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

        runtime_min = (time.time() - start_time) / 60.0

        result_p.update({
            "n": int(n),
            "edges": edges,
            "num_edges": int(len(edges)),
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

        if show:
            print("\nBest result:")
            print(f"p = {p}")
            print(f"num_params = {2 * p}")
            print(f"E_best = {result_p['best_energy']}")
            print(f"E0 = {ground_energy}")
            print(f"relative_error = {result_p['relative_error']}")
            print(f"runtime_min = {runtime_min:.4f}")

    return results


# ============================================================
# 7. Run multiple graphs
# ============================================================

def ejecutar_qaoa_grafos(
    grafos,
    n,
    p_max=15,
    mixer="jx",
    num_restarts=25,
    maxiter=500,
    seed=123,
    bounds_scale=np.pi,
    method="L-BFGS-B",
    use_warmstart=True,
    show=True,
):
    """
    Runs QAOA/QOAO for multiple graphs.

    Returns
    -------
    all_results : list[dict]
        List with all the results.
    """
    all_results = []

    for grafo_id, edges in enumerate(grafos, start=1):

        if show:
            print("\n" + "#" * 90)
            print(f"GRAPH {grafo_id} | n = {n} | edges = {edges}")
            print("#" * 90)

        results_grafo = scan_qaoa_p(
            n=n,
            edges=edges,
            p_max=p_max,
            mixer=mixer,
            num_restarts=num_restarts,
            maxiter=maxiter,
            seed=seed + 1000 * grafo_id,
            bounds_scale=bounds_scale,
            method=method,
            use_warmstart=use_warmstart,
            show=show,
        )

        for item in results_grafo:
            item["grafo_id"] = int(grafo_id)

        all_results.extend(results_grafo)

    return all_results


# ============================================================
# 8. Conversion to DataFrame and saving
# ============================================================

def qaoa_results_to_dataframe(results):
    """
    Converts complete QAOA results into a summary DataFrame.
    """
    rows = []

    for r in results:
        rows.append({
            "grafo_id": int(r.get("grafo_id", 1)),
            "n": int(r["n"]),
            "num_edges": int(r["num_edges"]),
            "mixer": str(r["mixer"]),

            "p": int(r["p"]),
            "num_params": int(r["num_params"]),

            "ground_energy": float(r["ground_energy"]),
            "initial_problem_energy": float(r["initial_problem_energy"]),
            "energy_final": float(r["best_energy"]),

            "absolute_error": float(r["absolute_error"]),
            "relative_error": float(r["relative_error"]),

            "runtime_min": float(r["runtime_min"]),

            "num_restarts": int(r["num_restarts"]),
            "maxiter": int(r["maxiter"]),
            "method": str(r["method"]),
            "best_init_name": str(r["best_init_name"]),

            "optimizer_success": bool(r["optimizer_success"]),
            "optimizer_nfev": int(r["optimizer_nfev"]),
            "optimizer_nit": int(r["optimizer_nit"]),
        })

    return pd.DataFrame(rows)


def guardar_resultados_qaoa(
    results,
    csv_path,
    json_path=None,
):
    """
    Saves QAOA results to a summary CSV and, optionally, a complete JSON.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = qaoa_results_to_dataframe(results)
    df.to_csv(csv_path, index=False)

    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                to_jsonable(results),
                f,
                indent=4,
                ensure_ascii=False
            )


# ============================================================
# 9. CD-ADAPT-VQE vs QAOA comparison (used in Max-3-Cut/qaoa_comparison.ipynb)
# ============================================================

# --- General helper functions ---

def cargar_resultados_json(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def error_relativo_desde_energias(energies, ground_energy):
    """
    Computes:
        |E_k - E0| / |E0|
    """
    E0 = float(ground_energy)

    if abs(E0) < 1e-14:
        raise ValueError(
            "The ground energy is too close to 0; it cannot be normalized this way."
        )

    errores = [abs(float(E) - E0) / abs(E0) for E in energies]
    return errores


def cortar_por_max_params(x, y, max_params):
    """
    Truncates a curve using x <= max_params.
    """
    if max_params is None:
        return x, y

    mask = x <= max_params
    return x[mask], y[mask]


def cortar_qaoa_con_banda(x, y_central, y_low, y_high, y_best, max_params):
    """
    Simultaneously truncates the central curve, the band, and QAOA's best curve.
    """
    if max_params is None:
        return x, y_central, y_low, y_high, y_best

    mask = x <= max_params

    return (
        x[mask],
        y_central[mask],
        y_low[mask],
        y_high[mask],
        y_best[mask],
    )


def cargar_imagen_con_fondo_transparente(image_path, threshold=245):
    """
    Loads an image and makes the white or near-white background transparent.

    threshold:
        higher values remove only very light whites.
        lower values remove more of the background.
    """
    image_path = Path(image_path)

    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)

    r, g, b, a = data[..., 0], data[..., 1], data[..., 2], data[..., 3]

    # Detect near-white pixels
    fondo_blanco = (r > threshold) & (g > threshold) & (b > threshold)

    # Make the background transparent
    data[..., 3] = np.where(fondo_blanco, 0, a)

    return data


def agregar_imagen_dentro(ax, image_path, zoom=0.13, xy=(0.21, 0.43)):
    """
    Adds an image inside the plot with a transparent background.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"Warning: image not found: {image_path}")
        return

    img = cargar_imagen_con_fondo_transparente(
        image_path,
        threshold=245
    )

    imagebox = OffsetImage(
        img,
        zoom=zoom
    )

    ab = AnnotationBbox(
        imagebox,
        xy,
        xycoords="axes fraction",
        frameon=False
    )

    ax.add_artist(ab)


# --- Functions for CD-ADAPT-VQE ---

def convertir_adapt_a_diccionario_por_grafo(data):
    """
    Converts the list of ADAPT results into:
        grafo_id -> result
    """
    dic = {}

    for item in data:
        if "error" in item:
            continue

        grafo_id = int(item["grafo_id"])
        dic[grafo_id] = item

    return dic


def curva_adapt(item):
    """
    Builds the ADAPT curve:
        x = number of operators added to the ansatz
        y = relative error
    """
    ground_energy = float(item["ground_energy"])
    energy_trace = list(item["energy_trace"])

    if "num_ansatz_ops" in item:
        q_final = int(item["num_ansatz_ops"])
    elif "iterations" in item:
        q_final = int(item["iterations"])
    elif "iteraciones" in item:
        q_final = int(item["iteraciones"])
    else:
        q_final = len(energy_trace) - 1

    energies = energy_trace[:q_final + 1]

    # Use the actual final energy at the last point if it exists.
    if "final_energy" in item and len(energies) > 0:
        energies[-1] = float(item["final_energy"])
    elif "energia_final" in item and len(energies) > 0:
        energies[-1] = float(item["energia_final"])

    x = np.arange(len(energies))
    y = np.array(error_relativo_desde_energias(energies, ground_energy))

    return x, y


# --- Functions for QAOA ---

def convertir_qaoa_a_diccionario_por_grafo(data):
    """
    Converts the list of QAOA results into:
        grafo_id -> list of results per depth p
    """
    dic = {}

    for item in data:
        if "error" in item:
            continue

        grafo_id = int(item["grafo_id"])

        if grafo_id not in dic:
            dic[grafo_id] = []

        dic[grafo_id].append(item)

    for grafo_id in dic:
        dic[grafo_id] = sorted(
            dic[grafo_id],
            key=lambda item: int(item["p"]) if "p" in item else int(item.get("num_params", 0))
        )

    return dic


def extraer_errores_desde_all_runs(item, incluir_warmstart=False):
    """
    Extracts the relative errors from item["all_runs"].
    """
    if "all_runs" not in item:
        raise KeyError(
            "This QAOA result does not have 'all_runs'. "
            "Make sure to use the complete JSON, not the summary CSV."
        )

    errores = []
    ground_energy = float(item["ground_energy"])

    for run in item["all_runs"]:

        init_name = str(run.get("init_name", ""))

        if not incluir_warmstart and init_name == "warm":
            continue

        if "relative_error" in run:
            errores.append(float(run["relative_error"]))

        elif "energy" in run:
            energy = float(run["energy"])

            if abs(ground_energy) < 1e-14:
                errores.append(abs(energy - ground_energy))
            else:
                errores.append(abs(energy - ground_energy) / abs(ground_energy))

        else:
            raise KeyError(
                "A run inside all_runs has neither 'relative_error' nor 'energy'."
            )

    errores = np.array(errores, dtype=float)

    if len(errores) == 0:
        raise ValueError(
            f"No valid errors found in all_runs for p={item.get('p', 'unknown')}."
        )

    return errores


def curva_qaoa_con_banda_desde_all_runs(
    items_qaoa,
    band_type="iqr",
    incluir_warmstart=False,
):
    """
    Builds the QAOA curve with a statistical band using all_runs.

    x-axis:
        num_params = 2p
    """
    items_qaoa = sorted(
        items_qaoa,
        key=lambda item: int(item["p"]) if "p" in item else int(item.get("num_params", 0))
    )

    xs = []
    central = []
    low = []
    high = []
    best = []

    for item in items_qaoa:

        if "num_params" in item:
            x = int(item["num_params"])
        else:
            x = 2 * int(item["p"])

        errores = extraer_errores_desde_all_runs(
            item,
            incluir_warmstart=incluir_warmstart,
        )

        xs.append(x)

        if band_type == "iqr":
            central.append(np.median(errores))
            low.append(np.percentile(errores, 25))
            high.append(np.percentile(errores, 75))

        elif band_type == "std":
            mean = np.mean(errores)
            std = np.std(errores)

            central.append(mean)
            low.append(max(mean - std, 1e-16))
            high.append(mean + std)

        else:
            raise ValueError("band_type must be 'iqr' or 'std'.")

        best.append(np.min(errores))

    return (
        np.array(xs),
        np.array(central),
        np.array(low),
        np.array(high),
        np.array(best),
    )

    return df