"""
main_n5.py  —  CD-ADAPT-VQE ensemble for n=5 random graphs
===============================================================
Usage:
    python main_n5.py <start_idx> <end_idx> <l>
Example (full run, first-order pool):
    python main_n5.py 1 300 1
Example (graphs 101-200, second-order pool):
    python main_n5.py 101 200 2

Improvements over main.py
--------------------------
- Incremental JSON saving: every completed graph is flushed to disk
  immediately, so a crash never loses already-computed results.
- Resume from checkpoint: if the output CSV already exists (from a
  previous interrupted run), completed graphs are detected and skipped.
- Real-time progress line: shows graph index, approximation ratio,
  energy error, per-graph runtime, and estimated time remaining.
- Summary statistics printed at the end of the run.
- Uses tqdm progress bar when available, falls back to plain prints.
"""

from __future__ import annotations

import ast
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funciones.utilidades import (
    cd_adapt_vqe_algorithm,
    generar_m_grafos,
    to_jsonable,
)

# ──────────────────────────────────────────────────────────────
# TRY TO USE TQDM (optional dependency)
# ──────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ══════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════

N           = 5          # number of qudits (vertices)
M           = 300        # ensemble size
MIN_EDGES   = None       # defaults to n=5 in generar_m_grafos
MAX_EDGES   = None       # defaults to C(5,2)=10
SEED        = 42         # reproducibility seed for graph generation

EPSILON     = 1e-2       # gradient-norm convergence threshold
MAX_ITER    = 50         # maximum adaptive iterations

GRAFOS_FILE = "grafos_n5.txt"

CSV_FIELDS = [
    "grafo_id",
    "num_edges",
    "ground_energy",
    "ground_degeneracy",
    "spectral_gap",
    "iteraciones",
    "energia_final",
    "diferencia_ground",
    "approx_ratio",
    "gradiente_final",
    "pool_size",
    "runtime_min",
    "initial_energy",
    "converged",
]

# ══════════════════════════════════════════════════════════════
# CLI ARGUMENTS
# ══════════════════════════════════════════════════════════════

if len(sys.argv) != 4:
    raise SystemExit(
        "Uso:     python main_n5.py <start_idx> <end_idx> <l>\n"
        "Ejemplo: python main_n5.py 1 300 1"
    )

start_idx = int(sys.argv[1])
end_idx   = int(sys.argv[2])
l         = int(sys.argv[3])

if l not in (1, 2):
    raise SystemExit("El parámetro l debe ser 1 o 2.")

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════

BASE_DIR   = PROJECT_ROOT
DATOS_DIR  = BASE_DIR / "datos"
CSV_DIR    = BASE_DIR / "resultados" / "csv"
JSON_DIR   = BASE_DIR / "resultados" / "json"

DATOS_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

ruta_grafos = DATOS_DIR / GRAFOS_FILE
csv_path    = CSV_DIR  / f"resultados_resumen_n5_l{l}_{start_idx}_{end_idx}.csv"
json_path   = JSON_DIR / f"resultados_completos_n5_l{l}_{start_idx}_{end_idx}.json"

# ══════════════════════════════════════════════════════════════
# BANNER
# ══════════════════════════════════════════════════════════════

print("=" * 62)
print("  CD-ADAPT-VQE  |  n=5 random-graph ensemble")
print(f"  Pool order : ℓ = {l}")
print(f"  Graphs     : {start_idx} – {end_idx}  ({end_idx - start_idx + 1} total)")
print(f"  ε          : {EPSILON}   |   r_max : {MAX_ITER}")
print(f"  CSV  → {csv_path.name}")
print(f"  JSON → {json_path.name}")
print("=" * 62)

# ══════════════════════════════════════════════════════════════
# GENERATE GRAPHS (only if the file does not exist yet)
# ══════════════════════════════════════════════════════════════

if not ruta_grafos.exists():
    generar_m_grafos(
        n=N, m=M,
        min_edges=MIN_EDGES, max_edges=MAX_EDGES,
        seed=SEED, filename=GRAFOS_FILE,
    )
    print(f"[setup] {M} grafos generados → {ruta_grafos}")
else:
    print(f"[setup] Reutilizando grafos existentes: {ruta_grafos}")

# ══════════════════════════════════════════════════════════════
# RESUME: detect already-completed graphs from existing CSV
# ══════════════════════════════════════════════════════════════

already_done: set[int] = set()
if csv_path.exists():
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                gid = int(row["grafo_id"])
                if row.get("energia_final", "ERROR") != "ERROR":
                    already_done.add(gid)
            except (ValueError, KeyError):
                pass

if already_done:
    print(f"[resume] {len(already_done)} grafos ya completados — se saltarán.")

# ══════════════════════════════════════════════════════════════
# LOAD GRAPH LIST
# ══════════════════════════════════════════════════════════════

with open(ruta_grafos, encoding="utf-8") as f:
    all_lines = f.readlines()

lines_slice = all_lines[start_idx - 1 : end_idx]

# ══════════════════════════════════════════════════════════════
# LOAD EXISTING JSON RESULTS (for resume)
# ══════════════════════════════════════════════════════════════

json_results: list = []
if already_done and json_path.exists():
    try:
        with open(json_path, encoding="utf-8") as f:
            json_results = json.load(f)
        print(f"[resume] {len(json_results)} resultados JSON previos cargados.")
    except json.JSONDecodeError:
        print("[resume] JSON previo corrupto — se empezará desde el CSV.")
        json_results = []

# ══════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════

csv_mode = "a" if already_done else "w"
timings: list[float] = []
errors: list[float]  = []

t_total_start = time.time()
graphs_to_run = [
    (start_idx + k, line)
    for k, line in enumerate(lines_slice)
    if (start_idx + k) not in already_done
]
n_pending = len(graphs_to_run)

if n_pending == 0:
    print("[info] Todos los grafos del bloque ya están completados.")
    sys.exit(0)

print(f"[run]  {n_pending} grafos pendientes. Iniciando...\n")

iterator = (
    tqdm(graphs_to_run, desc=f"ℓ={l}", unit="grafo")
    if HAS_TQDM
    else graphs_to_run
)

with open(csv_path, csv_mode, newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
    if csv_mode == "w":
        writer.writeheader()

    for done_count, (graph_id, linea) in enumerate(iterator, start=1):
        linea = linea.strip()
        if not linea:
            continue

        t0 = time.time()

        try:
            edges = ast.literal_eval("[" + linea + "]")

            result = cd_adapt_vqe_algorithm(
                n=N,
                edges=edges,
                l=l,
                epsilon=EPSILON,
                max_iteration=MAX_ITER,
                show=False,
            )

            dt = time.time() - t0
            timings.append(dt)

            rho         = result["final_energy"] / result["ground_energy"]
            delta_e     = result["difference_ground"]
            converged   = result["final_gradient_norm"] < EPSILON
            pool_size   = result.get("pool_size", "?")

            errors.append(delta_e)

            # ── CSV row (extended vs original) ──────────────────────
            writer.writerow({
                "grafo_id"        : graph_id,
                "num_edges"       : result["num_edges"],
                "ground_energy"   : result["ground_energy"],
                "ground_degeneracy": result.get("ground_degeneracy", "?"),
                "spectral_gap"    : result.get("spectral_gap", "?"),
                "iteraciones"     : result["iterations"],
                "energia_final"   : result["final_energy"],
                "diferencia_ground": delta_e,
                "approx_ratio"    : rho,
                "gradiente_final" : result["final_gradient_norm"],
                "pool_size"       : pool_size,
                "runtime_min"     : result["runtime_min"],
                "initial_energy"  : result["initial_energy"],
                "converged"       : int(converged),
            })
            csvfile.flush()

            # ── Incremental JSON ─────────────────────────────────────
            json_results.append(to_jsonable({"grafo_id": graph_id, **result}))
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(json_results, jf, indent=4, ensure_ascii=False)

            # ── Progress line (when tqdm is not available) ───────────
            if not HAS_TQDM:
                avg_t     = sum(timings) / len(timings)
                remaining = (n_pending - done_count) * avg_t
                conv_sym  = "✓" if converged else "✗"
                print(
                    f"  [{done_count:3d}/{n_pending}] "
                    f"grafo {graph_id:3d} | "
                    f"ρ={rho:.6f} | "
                    f"ΔE={delta_e:.2e} {conv_sym} | "
                    f"t={dt/60:.2f}min | "
                    f"ETA≈{remaining/60:.0f}min"
                )

        except Exception as exc:
            dt = time.time() - t0
            print(f"\n[ERROR] grafo {graph_id}: {exc}")

            writer.writerow({k: "ERROR" for k in CSV_FIELDS} | {"grafo_id": graph_id})
            csvfile.flush()

            json_results.append({"grafo_id": graph_id, "error": str(exc)})
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(json_results, jf, indent=4, ensure_ascii=False)

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

total_elapsed = (time.time() - t_total_start) / 60

print("\n" + "=" * 62)
print("  RESUMEN")
print("=" * 62)

if errors:
    rhos = []
    for r in json_results:
        if "final_energy" in r and "ground_energy" in r:
            try:
                rhos.append(float(r["final_energy"]) / float(r["ground_energy"]))
            except Exception:
                pass
    if rhos:
        rhos = np.array(rhos)
        errors_arr = np.array(errors)
        print(f"  Approximation ratio ρ")
        print(f"    mediana : {np.median(rhos):.6f}")
        print(f"    media   : {np.mean(rhos):.6f}")
        print(f"    mín     : {np.min(rhos):.6f}")
        print(f"    ρ ≥ 0.99: {100*np.mean(rhos >= 0.99):.1f}%")
        print(f"  Error absoluto ΔE")
        print(f"    mediana : {np.median(errors_arr):.4e}")
        print(f"    media   : {np.mean(errors_arr):.4e}")
        print(f"    máx     : {np.max(errors_arr):.4e}")
    print(f"  Grafos completados : {len(errors)}/{n_pending}")
    if timings:
        print(f"  Tiempo por grafo   : {np.mean(timings)/60:.3f} min (media)")
    print(f"  Tiempo total       : {total_elapsed:.1f} min")
print("=" * 62)
print(f"  CSV  → {csv_path}")
print(f"  JSON → {json_path}")
print("=" * 62)
