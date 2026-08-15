# Qudit-Adapt

Codebase numérica que implementa y compara dos algoritmos variacionales para
**Max-3-Cut sobre qutrits** (sistemas de dimensión 3, representación de espín
1): **CD-ADAPT-VQE** (Counterdiabatic Adaptive-VQE) y **QAOA/QOAO** (versión
qudit de QAOA). Es el companion numérico de una tesis de B.Sc. en Física
sobre CD-ADAPT-VQE aplicado a qutrits.

## Contenido

- [Arquitectura](#arquitectura)
- [Estructura de carpetas](#estructura-de-carpetas)
- [Entorno y dependencias](#entorno-y-dependencias)
- [`funciones/` — motor de los algoritmos](#funciones--motor-de-los-algoritmos)
- [`cluster/` — scripts para correr experimentos](#cluster--scripts-para-correr-experimentos)
- [`cuadernillos/` — notebooks de análisis](#cuadernillos--notebooks-de-análisis)
- [`datos/` y `resultados/`](#datos-y-resultados)
- [`papers/`](#papers)
- [Convenciones importantes](#convenciones-importantes)
- [Nota sobre `funciones/utilidades.py`](#nota-sobre-funcionesutilidadespy)

## Arquitectura

El flujo de trabajo del proyecto es siempre el mismo, en tres etapas:

1. **Generar/definir grafos** de entrada en `datos/*.txt`.
2. **Correr el algoritmo** (CD-ADAPT-VQE o QAOA) sobre esos grafos con un
   script de `cluster/`, pensado para ejecutarse en un servidor o cluster
   sin supervisión. Los resultados (CSV resumen + JSON completo con trazas)
   quedan en `resultados/csv/` y `resultados/json/`.
3. **Analizar los resultados** en un notebook de `cuadernillos/`, que lee
   esos CSV/JSON y genera las figuras y estadísticas.

Toda la lógica de los algoritmos (Hamiltonianos, construcción del pool de
operadores, el loop ADAPT, QAOA) vive en `funciones/`, no en los scripts ni
en los notebooks — estos solo la importan y la orquestan.

```
datos/*.txt  →  cluster/*.py  (usa funciones/*.py)  →  resultados/{csv,json}/  →  cuadernillos/*.ipynb
```

## Estructura de carpetas

```
Qudit-Adapt/
├── funciones/          Motor: Hamiltonianos, algoritmos CD-ADAPT-VQE y QAOA
├── cluster/             Scripts para correr experimentos (pensados para HPC)
├── cuadernillos/         Notebooks de análisis de resultados
├── datos/               Grafos de entrada (.txt) e imágenes por grafo
├── resultados/          Resultados generados: csv/, json/, images/
├── papers/              PDFs de referencia teórica
├── Tesis_Joaquín_Molina.pdf   Tesis de B.Sc. que este código acompaña
├── requirements_venv.txt
└── .venv/               Entorno virtual (no se sube al repo)
```

## Entorno y dependencias

Python 3.11, entorno virtual en `.venv/`. Para crearlo desde cero:

```bash
py -3.11 -m venv .venv
.venv/Scripts/pip install -r requirements_venv.txt   # Windows
# source .venv/bin/activate && pip install -r requirements_venv.txt   # Unix
```

Dependencias principales: `numpy`, `scipy`, `pandas`, `qutip` (álgebra de
operadores cuánticos), `matplotlib`, `sympy` (álgebra simbólica de
conmutadores), `jupyter`/`ipykernel`.

## `funciones/` — motor de los algoritmos

### `utilidades.py` — CD-ADAPT-VQE

- **Dos bases operatoriales distintas, no intercambiables:**
  - `Jx1, Jy1, Jz1` (`qt.jmat(1, 'x'|'y'|'z')`): momento angular real (spin-1).
  - `X_local()`: matriz custom de qutrit `[[1,1,0],[1,0,1],[0,1,1]]`
    (`= Jz² + √2·Jx`), la base "computacional" del qutrit.
- `Hi_qutip(n, omega0=1) = -omega0 · Σⱼ X_local_j` — Hamiltoniano inicial,
  el mismo en todo el proyecto (CD-ADAPT-VQE y QAOA por igual).
- `Hp_qutip(n, edges)` — Hamiltoniano del problema Max-3-Cut.
- Álgebra simbólica de conmutadores (`nested_commutators`, `comm_expr_expr`,
  etc.) construye el pool de operadores CD a partir de
  `Had(n, edges, lam) = (1-lam)·Hi + lam·Hp` y su derivada `dHad_dlam`.
  `l=1` usa solo el sector $O_1$; `l=2` usa $O_1 \cup O_3$.
- `cd_adapt_vqe_algorithm` / `cd_adapt_vqe_algorithm_profundo`: el loop
  ADAPT completo (selecciona el operador de mayor gradiente, reoptimiza con
  BFGS). La versión "profunda" además guarda trazas completas (gradientes
  top-k, orden exacto de operadores) para análisis y reconstrucción del
  circuito.
- `generar_m_grafos` / `ejecutar_grafos`: generación de grafos aleatorios y
  ejecución en lote, con guardado automático en `datos/` y `resultados/`.
- `compute_error_stats`, `truncate_result_to_budget`, `save_json`,
  `save_csv`, etc.: utilidades de post-procesamiento usadas por los
  notebooks de análisis (estadísticas de convergencia, truncamiento a un
  "presupuesto común" de iteraciones).

### `utilidades_bp.py` — robustez frente a barren plateaus

Módulo para el estudio de barren plateaus / paisajes rugosos. Reimplementa el
loop de CD-ADAPT-VQE en `numpy`/`scipy.sparse` con **gradiente analítico**
(retropropagación sobre el producto de exponenciales, `energy_and_grad`) en vez
de las diferencias finitas de `utilidades.py`: es ~$k$ veces más barato y evita
que el optimizador se detenga por ruido numérico, algo importante justamente
en un estudio de gradientes. Los Hamiltonianos y el pool CD se construyen
llamando a `utilidades.py`, así que son idénticos a los del algoritmo original.

- `adapt_bp_scan`: corre ADAPT y, en cada iteración, re-optimiza el **mismo**
  ansatz desde tres puntos iniciales — reciclado (warm start, el algoritmo
  real), frío ($\theta=0$, análogo de la curva "HF") y `n_random` instancias
  con parámetros uniformes en $[-\pi,\pi]$ (por defecto 100). De cada instancia
  se guarda sólo el **óptimo final**, que es lo que revela la estructura de
  mínimos locales del paisaje; con `store_history=True` guarda además todos los
  valores visitados.
- `gate_expression`, `ansatz_gates`, `ansatz_latex`: forma explícita de cada
  compuerta $e^{-i\theta_k G_k}$ del ansatz, con los $L_x,L_y,L_z$ actuando
  sobre cada sitio. Distingue el caso hermítico directo (factores en sitios
  distintos, que conmutan) del que requiere simetrización
  ($G=\tfrac12\{L_a^{(s)},L_b^{(s)}\}$ cuando hay dos ejes en el mismo sitio).
  `verificar_gate_expressions` chequea la reconstrucción contra qutip.
- `gradient_stats_at_random_inits`: reevalúa el gradiente exacto en los mismos
  $\theta^0$ aleatorios guardados, para medir si el paisaje se aplana al crecer
  el ansatz (el diagnóstico de barren plateau propiamente tal). Es post-hoc:
  no requiere volver a optimizar.
- `plot_bp_landscape`, `plot_bp_gradients`: las figuras.
- `verificar_gradiente`: chequeo del gradiente analítico contra diferencias
  finitas.

### `utilidades_QAOA.py` — QAOA/QOAO

- Dos mixers: `mixer="jx"` → $H_M = \sum_j J_{x,j}$ (momento angular real,
  **default**); `mixer="custom"` → $H_M = \sum_j X_{\text{local},j}$ (misma
  base que `Hi_qutip`).
- El estado inicial (`psi0`) es **siempre** el estado fundamental de
  `Hi_qutip(n, 1)` (la superposición uniforme), independientemente del
  mixer elegido — así QAOA y CD-ADAPT-VQE parten del mismo punto.
- `scan_qaoa_p`: barre profundidad $p=1,\dots,p_{\max}$ con warm-start
  entre capas. `optimize_qaoa_for_p`: multi-start (reinicios aleatorios +
  warm-start), L-BFGS-B por defecto.
- Funciones de comparación (usadas en `cuadernillos/comparacion_QAOA.ipynb`):
  carga de resultados, construcción de curvas de error con banda
  estadística, superposición de la imagen del grafo sobre los gráficos.

## `cluster/` — scripts para correr experimentos

| Script | Qué hace |
|---|---|
| `main.py` | CD-ADAPT-VQE sobre `datos/grafos_n6.txt` ($n=6$, $M=300$), para un rango de grafos y un $\ell$ dado |
| `main_n5.py` | Igual que `main.py` pero para $n=5$, con guardado incremental y resume desde checkpoint |
| `main_comparaciones.py` | CD-ADAPT-VQE sobre un archivo de grafos arbitrario (por defecto `grafos_comparacion.txt`); el nombre de salida se deriva automáticamente del archivo de entrada |
| `main_QAOA.py` | QAOA/QOAO sobre un archivo de grafos, con mixer, profundidad y reinicios configurables por línea de comandos |
| `run_kn.py` | CD-ADAPT-VQE sobre grafos completos $K_n$ ($n=4,\dots,10$), generados internamente (no usa `datos/`) |
| `main_bp.py` | Estudio de barren plateaus sobre un grafo: ADAPT + reinicios aleatorios por iteración (usa `funciones/utilidades_bp.py`) |

Todos siguen la misma convención de rutas: resuelven
`PROJECT_ROOT = Path(__file__).resolve().parent.parent` e insertan ese
directorio en `sys.path` antes de importar `funciones`, así funcionan sin
importar desde qué directorio se invoquen. Ejemplos de uso:

```bash
python cluster/main.py 1 20 1                 # grafos 1-20, l=1
python cluster/main_comparaciones.py 2         # l=2, grafos_comparacion.txt
python cluster/main_comparaciones.py 1 grafos_regulares.txt
python cluster/main_QAOA.py --n 6 --p_max 20 --num_restarts 25 --mixer jx \
    --input_file datos/grafos_comparacion.txt
python cluster/run_kn.py --l 2 --n_min 4 --n_max 10
python cluster/main_bp.py --grafo 2 --l 2 --max_iteration 30 --n_random 100
```

## `cuadernillos/` — notebooks de análisis

| Notebook | Contenido |
|---|---|
| `analisis_kn.ipynb` | CD-ADAPT-VQE sobre grafos completos $K_n$ ($n=4$–$10$), $\ell=1$ vs $\ell=2$ |
| `comparacion_QAOA.ipynb` | CD-ADAPT-VQE vs QAOA sobre $n=6$: grafos de referencia y grafos regulares (grado 2–5) |
| `grafos_aleatorios_n6.ipynb` | Ensemble de 300 grafos aleatorios $n=6$: $\ell=1$ vs $\ell=2$, con y sin presupuesto común de iteraciones |
| `analisis_localidad.ipynb` | Localidad de los operadores seleccionados por el ansatz (estructura del pool, contribución energética por localidad) |
| `barren_plateaus.ipynb` | Robustez frente a barren plateaus: curva de ADAPT (parámetros reciclados) contra la nube de reinicios aleatorios sobre el mismo ansatz |

Cada notebook parte con una sección `0. Set-up` (imports, estilo,
constantes) y, cuando aplica, una sección `0. Carga de datos` dentro de
cada bloque de análisis. Las rutas dentro de los notebooks son relativas
con prefijo `../` porque viven un nivel bajo la raíz del proyecto.

## `datos/` y `resultados/`

- `datos/grafos_n6.txt` — 300 grafos aleatorios $n=6$.
- `datos/grafos_comparacion.txt` — 4 grafos de referencia usados para
  comparar CD-ADAPT-VQE contra QAOA.
- `datos/grafos_regulares.txt` — 4 grafos regulares (grado 2, 3, 4 y 5).
- `datos/imagenes/` — imágenes de los grafos de referencia y regulares,
  usadas para ilustrar los gráficos de comparación.
- `resultados/csv/` y `resultados/json/` — salida de los scripts de
  `cluster/`: un CSV resumen y un JSON completo (con trazas de energía y
  gradiente) por cada corrida.
- `resultados/images/` — figuras ya generadas por los notebooks de `K_n`.

## `papers/`

Papers de referencia teórica: ADAPT-VQE y ADAPT-QAOA originales,
counterdiabatic driving (Sels & Polkovnikov, y su aplicación a qudits),
QAOA para sistemas qudit, barren plateaus, shortcuts to adiabaticity vía
espacio de Krylov, y circuitos comprimidos para AQC.

## Convenciones importantes

1. **Qutrits = dimensión 3 fija (spin-1)**, sitios indexados de 1 a $n$
   (no de 0 a $n-1$).
2. **No mezclar las dos bases operatoriales** (`Jx1/Jy1/Jz1` vs
   `X_local()`) sin verificar cuál corresponde en cada Hamiltoniano.
3. **Estado inicial siempre uniforme**: tanto CD-ADAPT-VQE como QAOA parten
   del estado fundamental de `Hi_qutip(n, 1)`, independientemente del
   mixer o algoritmo.
4. **Exponenciación repetida**: diagonalizar una sola vez
   (`precompute_operator_spectra_numpy`) y reusar la base espectral — nunca
   `expm` dentro del loop de un optimizador.
5. Cualquier script nuevo en `cluster/` debe seguir el mismo patrón de
   `PROJECT_ROOT` + `sys.path.insert` antes de importar `funciones`.
   Cualquier notebook nuevo en `cuadernillos/` debe usar rutas relativas
   con prefijo `../`.

## Nota sobre `funciones/utilidades.py`

Se hizo una limpieza de `utilidades.py` para sacar código muerto (funciones
duplicadas y nunca usadas) y ordenarlo. Se verificó cuidadosamente que
ninguna función realmente usada se haya eliminado, y se dejó
`funciones/utilidades_original.py` como respaldo intacto del archivo antes
de la limpieza, por si algo llegara a fallar.

**Si algo se rompe después de este cambio**, la solución es simple: borrar
`funciones/utilidades.py` y renombrar `funciones/utilidades_original.py` a
`funciones/utilidades.py`. Con eso todo vuelve a funcionar exactamente
como antes de la limpieza.

