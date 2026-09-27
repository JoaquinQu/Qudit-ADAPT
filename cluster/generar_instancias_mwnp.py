"""
Genera y fija el conjunto de instancias del benchmark de multiway partitioning.

Criterios de cada instancia (ver `utilidades_mwnp.instancia_unica`):
  * n enteros DISTINTOS en [1, 3n];
  * existe un reparto perfecto en tres cajas de suma idéntica;
  * ese reparto es único: degeneración exactamente 6.

Se generan desde ya las 20 instancias por tamaño, aunque la primera tanda de
corridas use sólo las 10 primeras. Fijar la segunda tanda antes de ver
resultados evita elegirla a posteriori. La generación es determinista por
(seed, n, índice), así que regenerar reproduce exactamente el mismo archivo.

    python cluster/generar_instancias_mwnp.py
    python cluster/generar_instancias_mwnp.py --n_min 5 --n_max 10 --por_n 20
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from funciones.utilidades_mwnp import instancias_unicas

SALIDA = PROJECT_ROOT / "datos" / "mwnp_instancias.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_min", type=int, default=5)
    p.add_argument("--n_max", type=int, default=10)
    p.add_argument("--por_n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--salida", type=str, default=str(SALIDA))
    args = p.parse_args()

    todas = []
    for n in range(args.n_min, args.n_max + 1):
        lote = instancias_unicas(n, args.por_n, seed=args.seed)
        todas += lote
        print(f"n={n:2d}  rango [1,{3*n}]  {len(lote)} instancias  "
              f"p_azar = 6/3^{n} = {lote[0]['p_azar']:.2e}")
        for inst in lote[:3]:
            print(f"    #{inst['id']:<2d} {str(inst['a']):42s} -> "
                  f"cajas {inst['cajas_optimas']}  (suma {inst['suma_por_caja']} c/u)")

    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump({"criterios": {
                       "numeros_distintos": True,
                       "rango": "[1, 3n]",
                       "particion_perfecta": True,
                       "degeneracion": 6,
                       "seed": args.seed,
                   },
                   "instancias": todas}, f, indent=1)
    print(f"\n{len(todas)} instancias guardadas en {Path(args.salida).relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
