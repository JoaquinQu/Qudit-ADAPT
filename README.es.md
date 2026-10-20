# Qudit-ADAPT

Implementación de referencia y datos de

> **Qudit-ADAPT-VQE: an adaptive variational algorithm with counterdiabatic-inspired improvements for qudits**
> J. Molina, H. Díaz-Moraga, D. Goyeneche, D. Tancara
> Facultad de Física, Pontificia Universidad Católica de Chile

El código resuelve **Max 3-Cut** sobre qutrits ($d=3$, representación de espín 1)
con dos algoritmos y los compara:

- **Qudit-ADAPT** — ADAPT-VQE cuyo pool de operadores sale de un potencial de
  gauge adiabático aproximado, construido con conmutadores anidados del mixer y
  el Hamiltoniano de costo, truncado a orden $\ell$.
- **QAOA de qudits** — la referencia de ansatz fijo.

Toda figura y tabla del paper se puede regenerar desde este repositorio. En
[`REPRODUCING.md`](REPRODUCING.md) está el comando exacto de cada una, y en
[`DATA.md`](DATA.md) dónde se produjo cada dataset y quién lo corrió.

*(English version: [`README.md`](README.md) — es la versión de referencia.)*

---

## Por dónde empezar

Los cuadernillos de `examples/` son la puerta de entrada. Corren en segundos en
un notebook, cada uno es autocontenido y entre los cuatro cubren todo el flujo:

| Cuadernillo | Qué muestra |
|---|---|
| [`01_max3cut_and_hamiltonians`](examples/01_max3cut_and_hamiltonians.ipynb) | El problema: variables ternarias, $H_C$, $H_M$, y la solución exacta de un grafo de seis vértices por fuerza bruta |
| [`02_operator_pool`](examples/02_operator_pool.ipynb) | De dónde sale el pool: conmutadores anidados, por qué hay que hermitizar, $\ell=1$ contra $\ell=2$ |
| [`03_running_qudit_adapt`](examples/03_running_qudit_adapt.ipynb) | Una corrida completa sobre $G_1$: selección de operadores, warm start, convergencia |
| [`04_native_gate_count`](examples/04_native_gate_count.ipynb) | Apéndice B: compilación a compuertas nativas de iones atrapados y la comparación con QAOA |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab examples/
```

## Organización

```
functions/                     el motor — toda la física vive acá, un solo paquete
examples/                      cuadernillos comentados, empieza acá
Max-3-Cut/                     Figs. 1-4 y Tabla I, autocontenido (ver su propio README)
local_minima/                  Figs. 5-6 y Sec. III C, autocontenido (ver su propio README)
multiway_number_partitioning/  Sec. III B, autocontenido (ver su propio README)
```

Cada una de las tres carpetas de tema tiene su propio `cluster/` con los scripts que la producen
(ej. `Max-3-Cut/cluster/main_comparaciones.py`) — ver el README de esa carpeta para la lista
completa, o [`REPRODUCING.md`](REPRODUCING.md) para el comando exacto detrás de cada figura.

La raíz también tiene carpetas `data/`, `results/` y `cluster/`, ninguna parte del repo público: son
donde leen y escriben por defecto `main_bp.py` y los scripts de mwnp (la mayoría de los scripts
resuelven rutas relativas a la raíz del repositorio, no a donde viven ellos mismos), y `Max-3-Cut/`,
`local_minima/` y `multiway_number_partitioning/` llevan las copias limpias y trackeadas de lo
que realmente necesitan.

(`multiway number partition/` — una versión vieja y exploratoria de multiway, en español,
con un import roto conocido — se queda en el disco pero gitignorada; la de arriba,
`multiway_number_partitioning/`, es la que es parte del paper.)

Nada en ningún `cluster/` implementa física: importan de `functions/` y orquestan. El
grafo de dependencias es un árbol, sin ciclos:

```
utilidades.py  ──────────┬──> utilidades_QAOA.py
                         └──> utilidades_bp.py ──┬──> utilidades_gellmann.py
                                                 └──> utilidades_ringbauer.py
```

### `functions/` — el motor

| Módulo | Contenido |
|---|---|
| `utilidades.py` | Operadores de qutrit, $H_C$ y $H_M$, conmutadores anidados simbólicos, el loop CD-ADAPT original |
| `utilidades_QAOA.py` | QAOA de qudits: construcción de capas, energía, barrido en profundidad con warm start |
| `utilidades_bp.py` | Motor ADAPT vectorizado con gradiente analítico, barrido del paisaje de mínimos locales, y conteo de compuertas nativas |
| `utilidades_gellmann.py` | El mismo pool reconstruido en la base de Gell-Mann de $\mathfrak{su}(3)$ (Apéndice B) |
| `utilidades_ringbauer.py` | Descomposición de una unitaria de un qudit en rotaciones de dos niveles, siguiendo a Ringbauer *et al.* |

`utilidades.py` y `utilidades_QAOA.py` se tradujeron al inglés, porque son lo que importan
los notebooks de `Max-3-Cut/` y la única parte del código de cara al paper pensada para
que alguien fuera del grupo la lea de principio a fin. El resto del módulo mantiene sus
comentarios en español. Hubo una segunda carpeta, `funciones/`, que quedó como shim de
compatibilidad mientras solo `utilidades.py`/`utilidades_QAOA.py` se habían movido; ya se
reincorporó — todo lo que depende del motor importa de `functions/` directamente,
`funciones/` ya no existe.

### `<tema>/cluster/` — experimentos

| Script | Produce |
|---|---|
| `Max-3-Cut/cluster/main_comparaciones.py` | Qudit-ADAPT sobre los grafos de benchmark → Figs. 1–3, Tabla I |
| `Max-3-Cut/cluster/main_QAOA.py` | La referencia QAOA sobre los mismos grafos → Figs. 1–3, Tabla I |
| `Max-3-Cut/cluster/main.py` | El ensemble de 300 grafos aleatorios → Fig. 4 |
| `local_minima/cluster/main_bp.py` | Barrido del paisaje warm/cold/reinicios aleatorios → Figs. 5–6 (escribe en la `data/`/`results/` de la raíz; `local_minima/` lleva una copia limpia de su salida) |
| `local_minima/cluster/figuras_paper.py` | Dibuja las Figs. 5–6 con la tipografía del manuscrito |
| `local_minima/cluster/tablas_apendice.py` | Tablas II y III |
| `local_minima/cluster/conteo_qaoa.py` | Tabla IV |
| `local_minima/cluster/lanzar_bp.sh` | Lanzador para corridas largas desatendidas |

Los scripts resuelven rutas relativas a la raíz del repositorio
(`Path(__file__).resolve().parent.parent.parent`), no a donde viven ellos mismos, así que moverlos
bajo el `cluster/` de su tema no les rompe nada. `main_comparaciones.py`, `main_QAOA.py`, `main.py`,
`main_n5.py` y `run_kn.py` leen sus grafos de entrada desde `Max-3-Cut/data/` y escriben en
`Max-3-Cut/results/`. `main_bp.py` (y todo lo que alimenta a `local_minima/`,
`multiway_number_partitioning/` o el Apéndice B) sigue resolviendo rutas relativas contra
`data/`/`results/` en la raíz. Ninguna de las dos carpetas de la raíz es parte del repo público, sin
embargo: todo archivo que sí se publica —los grafos fijos de benchmark, las figuras, los datos de
las corridas— tiene su copia trackeada dentro de la carpeta de su tema (`Max-3-Cut/data/`,
`local_minima/data/`, ...), y `data/`/`results/` de la raíz solo guardan las copias locales sin
trackear que esos scripts necesitan para correr.

## Convenciones

Tres cosas fáciles de malinterpretar al leer el código:

**Los sitios se numeran desde 1.** Las aristas son `(1,2)`, no `(0,1)`. Los
niveles dentro de un qutrit sí van desde 0, así que `lambda_1` actúa sobre el
par `(0,1)`.

**Las etiquetas de operador son strings canónicos.** Un operador del pool se
identifica con `"((1, 'y'), (3, 'x'), (4, 'z'))"` en momento angular o
`"((2, 2), (3, 8), (4, 6))"` en Gell-Mann. El pool se ordena por ese string, de
modo que los índices de operador son reproducibles entre corridas.

**El ansatz se aplica de derecha a izquierda.** `ansatz_op_labels[0]` es el
operador más cercano al estado de referencia $|\phi_g\rangle$, o sea el primero
en aplicarse.

Los gradientes son analíticos (adjunto/backpropagation), no diferencias
finitas: $\partial E/\partial\theta_j = 2\,\mathrm{Im}\langle\sigma_j|A_j|\varphi_j\rangle$.
Eso es lo que permite que las corridas con $\ell=2$ lleguen a precisión de máquina.

## Idioma

Los comentarios del código están en español, como los escribieron los autores.
La documentación, los ejemplos y los scripts agregados para la publicación
están en inglés. Ambos README se mantienen sincronizados; el inglés es la
versión de referencia.

Excepción: `functions/utilidades.py` y `functions/utilidades_QAOA.py` — el motor detrás
de `Max-3-Cut/` — se tradujeron al inglés, porque es la única parte del código de cara
al paper pensada para que alguien fuera del grupo la lea de principio a fin.
`functions/utilidades_bp.py`, `utilidades_gellmann.py` y `utilidades_ringbauer.py`
mantienen sus comentarios en español, sin cambios.

## Cómo citar

```bibtex
@article{molina2026quditadapt,
  title   = {Qudit-ADAPT-VQE: an adaptive variational algorithm with
             counterdiabatic-inspired improvements for qudits},
  author  = {Molina, Joaqu\'in and D\'iaz-Moraga, Herbert and
             Goyeneche, Dardo and Tancara, Diego},
  year    = {2026}
}
```
