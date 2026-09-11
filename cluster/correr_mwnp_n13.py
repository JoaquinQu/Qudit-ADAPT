"""
Una instancia de multiway number partitioning con n = 13 qutrits, de punta a punta.

Corre Qudit-ADAPT sobre trece números, verifica el resultado contra la
enumeración exacta de las 3^13 = 1 594 323 asignaciones, y traduce el estado
final a las particiones que uno realmente mediría.

    python cluster/correr_mwnp_n13.py
    python cluster/correr_mwnp_n13.py --l 1 --max_iteration 40 --seed 2026
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import time

import numpy as np

from funciones.utilidades_mwnp import (
    adapt_mwnp,
    energia_a_objetivo,
    fuerza_bruta,
    particion_desde_indice,
    sumas_de_particion,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=13)
    p.add_argument("--l", type=int, default=1, choices=[1, 2])
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--max_iteration", type=int, default=40)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--salida", type=str,
                   default=str(PROJECT_ROOT / "resultados" / "json" / "mwnp_n13.json"))
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    a = sorted(rng.integers(10, 100, size=args.n).tolist(), reverse=True)

    print("=" * 72)
    print(f"MULTIWAY NUMBER PARTITIONING  |  n = {args.n} qutrits, k = 3 cajas")
    print("=" * 72)
    print(f"números : {a}")
    print(f"suma    : {sum(a)}    promedio ideal por caja: {sum(a)/3:.2f}")
    print(f"espacio : 3^{args.n} = {3**args.n:,} asignaciones", flush=True)

    print("\n--- respuesta exacta, por enumeración ---", flush=True)
    t = time.time()
    fb = fuerza_bruta(a)
    print(f"  objetivo mínimo sum(S_s^2) : {fb['objetivo']:.0f}")
    print(f"  desbalance mínimo          : {fb['desbalance']:.0f}")
    print(f"  E_0                        : {fb['energia']:.4f}")
    print(f"  degeneración               : {fb['degeneracion']} estados")
    print(f"  ({time.time()-t:.1f} s)", flush=True)

    print(f"\n--- Qudit-ADAPT, l = {args.l} ---", flush=True)
    ck = str(Path(args.salida).with_suffix(".checkpoint.json"))
    res = adapt_mwnp(a, l=args.l, epsilon=args.epsilon,
                     max_iteration=args.max_iteration, mostrar=True,
                     checkpoint=ck)

    print("\n" + "=" * 72)
    print("RESULTADO")
    print("=" * 72)
    print(f"parámetros variacionales : {res['num_ansatz_ops']}  ({res['stop_reason']})")
    print(f"energía final            : {res['final_energy']:.6f}   (E_0 = {res['ground_energy']:.6f})")
    print(f"error relativo           : {res['rel_error']:.4e}")
    print(f"objetivo alcanzado       : {res['objetivo']:.2f}   (óptimo {res['objetivo_optimo']:.0f})")
    print(f"tiempo                   : {res['runtime_s']/60:.1f} min")

    print(f"\nprobabilidad de medir una partición ÓPTIMA: "
          f"{res['prob_subespacio_optimo']:.4f}")
    print(f"probabilidad de la cadena más probable    : {res['mejor_probabilidad']:.4f}")

    print("\n--- las cadenas más probables del estado final ---")
    print(f"  {'cadena de trits':>16s}{'prob':>9s}{'sumas':>22s}{'desbal':>8s}{'óptima':>8s}")
    for t_ in res["top_particiones"]:
        s = "[" + ", ".join(f"{int(x)}" for x in t_["sumas"]) + "]"
        opt = "sí" if t_["desbalance"] <= fb["desbalance"] else "no"
        print(f"  {t_['trits']:>16s}{t_['probabilidad']:>9.4f}{s:>22s}"
              f"{t_['desbalance']:>8.0f}{opt:>8s}")

    mejor = res["top_particiones"][0]
    print("\n--- qué significa esa cadena ---")
    for c in range(3):
        caja = mejor["cajas"][c]
        print(f"  caja {c}  (trit = {c}) : {caja}")
        print(f"            suma = {int(mejor['sumas'][c])}")

    salida = dict(res)
    salida["fuerza_bruta"] = {k: v for k, v in fb.items() if k != "indices_optimos"}
    salida["args"] = vars(args)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=1, default=float)
    print(f"\nguardado en {args.salida}")


if __name__ == "__main__":
    main()
