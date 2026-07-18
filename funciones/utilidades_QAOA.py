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

from funciones.utilidades import (
    Hp_qutip,
    Jx_site,
    Hi_qutip,
    precompute_operator_spectra_numpy,
    apply_exp_to_state_from_spectrum_numpy,
    to_jsonable,
)


# ============================================================
# 1. Lectura de grafos
# ============================================================

def leer_grafos(path):
    """
    Lee un archivo .txt donde cada línea representa un grafo.

    Formato esperado por línea:
        (1, 2), (1, 3), (2, 3)

    Retorna
    -------
    grafos : list[list[tuple[int, int]]]
        Lista de grafos. Cada grafo es una lista de aristas.
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
# 2. Hamiltonianos QAOA/QOAO
# ============================================================

def Hm_qaoa_jx(n):
    """
    Mixer QAOA/QOAO para qutrits usando operadores de momento angular:

        H_M = sum_j Jx_j

    Este es el mixer más cercano al paper de QAOA para qudits.
    """
    Hm = 0

    for j in range(1, n + 1):
        Hm += Jx_site(n, j)

    return Hm


def Hm_qaoa_x_custom(n):
    """
    Mixer alternativo usando el operador X_local de tu implementación original.

    Como Hi_qutip(n, 1) = - sum_j X_j,
    entonces:

        H_M = sum_j X_j = -Hi_qutip(n, 1)

    Esta versión sirve para comparar con el mixer que ya usas en CD-ADAPT.
    """
    return -Hi_qutip(n, omega0=1)


def preparar_qaoa_para_grafo(n, edges, mixer="jx"):
    """
    Construye todos los objetos necesarios para ejecutar QAOA/QOAO
    sobre un grafo dado.

    Parámetros
    ----------
    n : int
        Número de nodos/qutrits.
    edges : list[tuple[int, int]]
        Lista de aristas con nodos numerados desde 1.
    mixer : str
        "jx"      -> H_M = sum_j Jx_j
        "custom"  -> H_M = sum_j X_j, usando tu Hi_qutip

    Retorna
    -------
    data : dict
        Diccionario con Hc, Hm, E0, psi0, espectros y matrices.

    Notas
    -----
    El estado inicial psi0 siempre es la superposición uniforme
    (estado fundamental de -sum_j X_j, el mismo Hi_qutip usado en
    CD-ADAPT-VQE), independiente del mixer elegido para las capas QAOA.
    """
    Hc = Hp_qutip(n, edges)

    if mixer == "jx":
        Hm = Hm_qaoa_jx(n)
    elif mixer == "custom":
        Hm = Hm_qaoa_x_custom(n)
    else:
        raise ValueError("mixer debe ser 'jx' o 'custom'.")

    if not Hc.isherm:
        raise ValueError("Hc no es hermítico.")

    if not Hm.isherm:
        raise ValueError("Hm no es hermítico.")

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
# 3. Estado y energía QAOA/QOAO
# ============================================================

def qaoa_qudit_state_numpy(params, spec_Hc, spec_Hm, psi0_vec, p):
    """
    Construye el estado QAOA/QOAO para qutrits.

    Convención:

        params = [gamma_1, ..., gamma_p, beta_1, ..., beta_p]

    y

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
    Calcula la energía variacional:

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
    Error relativo respecto al ground energy exacto.
    """
    if abs(ground_energy) < 1e-12:
        return float(abs(energy - ground_energy))

    return float(abs(energy - ground_energy) / abs(ground_energy))


# ============================================================
# 4. Warm-start entre profundidades
# ============================================================

def expand_params_previous_p(best_params_previous, p):
    """
    Expande parámetros de p-1 layers a p layers.

    Entrada p-1:
        [gamma_1, ..., gamma_{p-1}, beta_1, ..., beta_{p-1}]

    Salida p:
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
# 5. Optimización para p fijo
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
    Optimiza QAOA/QOAO para una profundidad p.

    Usa:
    - Warm-start desde p-1, si existe.
    - num_restarts puntos iniciales aleatorios.

    Parámetros
    ----------
    p : int
        Número de layers QAOA.
    num_restarts : int
        Número de reinicios aleatorios.
    maxiter : int
        Máximo de iteraciones del optimizador local.
    use_warmstart : bool
        Si True, agrega como primer punto inicial el mejor resultado de p-1.

    Retorna
    -------
    result_dict : dict
        Mejor resultado encontrado para ese p.
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
# 6. Barrido en p para un grafo
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
    Ejecuta QAOA/QOAO para p = 1, ..., p_max sobre un grafo.

    Retorna
    -------
    results : list[dict]
        Lista de resultados por profundidad.
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
            print("\nMejor resultado:")
            print(f"p = {p}")
            print(f"num_params = {2 * p}")
            print(f"E_best = {result_p['best_energy']}")
            print(f"E0 = {ground_energy}")
            print(f"relative_error = {result_p['relative_error']}")
            print(f"runtime_min = {runtime_min:.4f}")

    return results


# ============================================================
# 7. Ejecutar varios grafos
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
    Ejecuta QAOA/QOAO para varios grafos.

    Retorna
    -------
    all_results : list[dict]
        Lista con todos los resultados.
    """
    all_results = []

    for grafo_id, edges in enumerate(grafos, start=1):

        if show:
            print("\n" + "#" * 90)
            print(f"GRAFO {grafo_id} | n = {n} | edges = {edges}")
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
# 8. Conversión a DataFrame y guardado
# ============================================================

def qaoa_results_to_dataframe(results):
    """
    Convierte resultados completos de QAOA a DataFrame resumen.
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
    Guarda resultados de QAOA en CSV resumen y opcionalmente JSON completo.
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
# 9. Comparación CD-ADAPT-VQE vs QAOA (usadas en cuadernillos/comparacion_QAOA.ipynb)
# ============================================================

# --- Funciones auxiliares generales ---

def cargar_resultados_json(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def error_relativo_desde_energias(energies, ground_energy):
    """
    Calcula:
        |E_k - E0| / |E0|
    """
    E0 = float(ground_energy)

    if abs(E0) < 1e-14:
        raise ValueError(
            "La ground energy es demasiado cercana a 0; no se puede normalizar así."
        )

    errores = [abs(float(E) - E0) / abs(E0) for E in energies]
    return errores


def cortar_por_max_params(x, y, max_params):
    """
    Corta una curva usando x <= max_params.
    """
    if max_params is None:
        return x, y

    mask = x <= max_params
    return x[mask], y[mask]


def cortar_qaoa_con_banda(x, y_central, y_low, y_high, y_best, max_params):
    """
    Corta simultáneamente la curva central, la banda y la curva best de QAOA.
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
    Carga una imagen y vuelve transparente el fondo blanco o casi blanco.

    threshold:
        valores más altos eliminan solo blancos muy claros.
        valores más bajos eliminan más fondo.
    """
    image_path = Path(image_path)

    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)

    r, g, b, a = data[..., 0], data[..., 1], data[..., 2], data[..., 3]

    # Detectar píxeles casi blancos
    fondo_blanco = (r > threshold) & (g > threshold) & (b > threshold)

    # Hacer transparente el fondo
    data[..., 3] = np.where(fondo_blanco, 0, a)

    return data


def agregar_imagen_dentro(ax, image_path, zoom=0.13, xy=(0.21, 0.43)):
    """
    Agrega una imagen dentro del gráfico con fondo transparente.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"Advertencia: no se encontró la imagen {image_path}")
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


# --- Funciones para CD-ADAPT-VQE ---

def convertir_adapt_a_diccionario_por_grafo(data):
    """
    Convierte la lista de resultados ADAPT en:
        grafo_id -> resultado
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
    Construye curva ADAPT:
        x = número de operadores agregados al ansatz
        y = error relativo
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

    # Usar energía final real en el último punto si existe.
    if "final_energy" in item and len(energies) > 0:
        energies[-1] = float(item["final_energy"])
    elif "energia_final" in item and len(energies) > 0:
        energies[-1] = float(item["energia_final"])

    x = np.arange(len(energies))
    y = np.array(error_relativo_desde_energias(energies, ground_energy))

    return x, y


# --- Funciones para QAOA ---

def convertir_qaoa_a_diccionario_por_grafo(data):
    """
    Convierte la lista de resultados QAOA en:
        grafo_id -> lista de resultados por profundidad p
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
    Extrae los errores relativos desde item["all_runs"].
    """
    if "all_runs" not in item:
        raise KeyError(
            "Este resultado QAOA no tiene 'all_runs'. "
            "Asegúrate de usar el JSON completo, no el CSV resumen."
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
                "Una corrida dentro de all_runs no tiene ni 'relative_error' ni 'energy'."
            )

    errores = np.array(errores, dtype=float)

    if len(errores) == 0:
        raise ValueError(
            f"No encontré errores válidos en all_runs para p={item.get('p', 'desconocido')}."
        )

    return errores


def curva_qaoa_con_banda_desde_all_runs(
    items_qaoa,
    band_type="iqr",
    incluir_warmstart=False,
):
    """
    Construye curva QAOA con banda estadística usando all_runs.

    Eje x:
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
            raise ValueError("band_type debe ser 'iqr' o 'std'.")

        best.append(np.min(errores))

    return (
        np.array(xs),
        np.array(central),
        np.array(low),
        np.array(high),
        np.array(best),
    )

    return df