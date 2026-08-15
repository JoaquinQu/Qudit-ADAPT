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
funciones/      algoritmos — toda la física vive acá
cluster/        scripts ejecutables: un experimento o una figura cada uno
examples/       cuadernillos comentados, empieza acá
cuadernillos/   los notebooks de análisis que produjeron las figuras del paper
datos/          grafos de entrada
resultados/     salidas: json (trazas completas), csv (resúmenes), imágenes, logs
papers/         literatura de referencia
```

Nada en `cluster/` ni en `cuadernillos/` implementa física: importan de
`funciones/` y orquestan. El grafo de dependencias es un árbol, sin ciclos:

```
utilidades.py  ──────────┬──> utilidades_QAOA.py
                         └──> utilidades_bp.py ──┬──> utilidades_gellmann.py
                                                 └──> utilidades_ringbauer.py
```

### `funciones/` — el motor

| Módulo | Contenido |
|---|---|
| `utilidades.py` | Operadores de qutrit, $H_C$ y $H_M$, conmutadores anidados simbólicos, el loop CD-ADAPT original |
| `utilidades_QAOA.py` | QAOA de qudits: construcción de capas, energía, barrido en profundidad con warm start |
| `utilidades_bp.py` | Motor ADAPT vectorizado con gradiente analítico, barrido del paisaje de mínimos locales, y conteo de compuertas nativas |
| `utilidades_gellmann.py` | El mismo pool reconstruido en la base de Gell-Mann de $\mathfrak{su}(3)$ (Apéndice B) |
| `utilidades_ringbauer.py` | Descomposición de una unitaria de un qudit en rotaciones de dos niveles, siguiendo a Ringbauer *et al.* |

### `cluster/` — experimentos

| Script | Produce |
|---|---|
| `main_comparaciones.py` | Qudit-ADAPT sobre los grafos de benchmark → Figs. 1–3, Tabla I |
| `main_QAOA.py` | La referencia QAOA sobre los mismos grafos → Figs. 1–3, Tabla I |
| `main.py` | El ensemble de 300 grafos aleatorios → Fig. 4 |
| `main_bp.py` | Barrido del paisaje warm/cold/reinicios aleatorios → Figs. 5–6 |
| `figuras_paper.py` | Dibuja las Figs. 5–6 con la tipografía del manuscrito |
| `tablas_apendice.py` | Tablas II y III |
| `conteo_qaoa.py` | Tabla IV |
| `lanzar_bp.sh` | Lanzador para corridas largas desatendidas |

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
