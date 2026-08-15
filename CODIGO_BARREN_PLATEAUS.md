# Código del estudio de entrenabilidad y barren plateaus

Guía del código añadido al repositorio para estudiar la entrenabilidad de
Qudit-ADAPT y su robustez frente a *barren plateaus*, y para comparar pools
construidos en distintas bases operatoriales.

Todo lo que sigue reproduce el Hamiltoniano y el algoritmo del paper
*"Qudit-ADAPT-VQE: an adaptive variational algorithm with counterdiabatic-inspired
improvements for qudits"* (Molina, Goyeneche & Tancara).

## Contenido

- [Convenciones del paper](#convenciones-del-paper)
- [Archivos añadidos](#archivos-añadidos)
- [`utilidades_bp.py` — el motor](#utilidades_bppy--el-motor)
- [`utilidades_gellmann.py` — pool en base de Gell-Mann](#utilidades_gellmannpy--pool-en-base-de-gell-mann)
- [`utilidades_heisenberg.py` — pool en base de Heisenberg-Weyl](#utilidades_heisenbergpy--pool-en-base-de-heisenberg-weyl)
- [Scripts de `cluster/`](#scripts-de-cluster)
- [Cómo reproducir cada figura](#cómo-reproducir-cada-figura)
- [Validaciones](#validaciones)
- [Notas de implementación](#notas-de-implementación)

## Convenciones del paper

Las que usa este código, para que todo sea directamente comparable:

| Cantidad | Definición | Ecuación en el paper |
|---|---|---|
| Hamiltoniano de costo | $H_C=\sum_{(i,j)\in E}[L_z^{(i)}L_z^{(j)}-2((L_z^{(i)})^2+(L_z^{(j)})^2)+3(L_z^{(i)})^2(L_z^{(j)})^2]$ | (32) |
| Mixer | $H_M=-\sum_j[\sqrt2 L_x^{(j)}+(L_z^{(j)})^2]$ | (12) |
| Estado inicial | $\ket{\phi_g}=\bigotimes_j\ket{+_3}_j$, superposición uniforme | (15) |
| Pool | $\mathcal V=\{V_j\}$, $V_j=(G_j+G_j^\dagger)/2$ | (28) |
| Gradiente | $g_j^{(k)}=i\langle\psi_k\vert[V_j,H_C]\vert\psi_k\rangle$ | (29) |
| Criterio de parada | $\Vert g^{(k)}\Vert_2<\epsilon$ | — |
| **Métrica** | $\epsilon_{\rm rel}=\vert E_{\rm final}-E_0\vert/\vert E_0\vert$ | (33) |

**Ojo con la métrica**: el paper reporta error **relativo**. Las funciones de
graficado aceptan `relativo=True` para usarlo; por defecto usan el error
absoluto $\vert E-E_0\vert$.

Base computacional $\ket{+1}=(1,0,0)^T$, $\ket{0}=(0,1,0)^T$,
$\ket{-1}=(0,0,1)^T$, que es la de `qutip` (`jmat(1,'z')=diag(1,0,-1)`).

## Archivos añadidos

```
funciones/
  utilidades_bp.py            motor del experimento, diagnósticos y figuras
  utilidades_gellmann.py      álgebra y pool en base de Gell-Mann
  utilidades_heisenberg.py    álgebra y pool en base de Heisenberg-Weyl
cluster/
  main_bp.py                  una corrida (grafo, pool, nº de instancias)
  main_ensemble_pools.py      comparación de pools sobre un ensemble
cuadernillos/
  barren_plateaus.ipynb       el análisis completo
informe/
  informe_barren_plateaus.tex informe técnico con las figuras
```

Los Hamiltonianos y el pool se construyen **llamando a `funciones/utilidades.py`**,
no reimplementándolos, así que son los mismos del algoritmo publicado.

## `utilidades_bp.py` — el motor

### El experimento

`adapt_bp_scan(n, edges, l, ...)` corre Qudit-ADAPT y, en **cada** iteración,
re-optimiza el **mismo** ansatz de $k$ parámetros desde tres tipos de punto
inicial:

| Estrategia | Punto inicial | Qué representa |
|---|---|---|
| reciclada | $(\theta^*_{k-1},0)$ | el algoritmo real (*warm start* del paper) |
| fría | $\theta=0$ | reinicia el estado a $\ket{\phi_g}$ en cada paso |
| aleatoria ×`n_random` | $\theta_j\sim\mathcal U(-\pi,\pi)$ | ansatz de tamaño fijo sin información previa |

De cada instancia aleatoria se guarda **sólo el óptimo final** (`store_history=False`,
por defecto). Ésa es la diferencia importante: cada marca del gráfico es
entonces un mínimo donde efectivamente terminó una optimización, así que la
columna muestra la estructura de mínimos locales del ansatz de ese tamaño.

Parámetros relevantes:

```python
adapt_bp_scan(
    n, edges,
    l=1,                    # 1 -> pool O_1 ; 2 -> O_1 U O_3
    epsilon=1e-2,           # corte por norma del gradiente
    max_iteration=20,       # máximo de operadores en el ansatz
    n_random=100,           # instancias aleatorias por iteración (0 = sólo la curva)
    store_history=False,    # True guarda todos los valores visitados
    base="angular",         # "angular" | "gellmann" | "gellmann_local" | "heisenberg"
    checkpoint_path=None,   # guarda resultado parcial en CADA iteración
)
```

**El checkpoint importa**: con `n_random=100` y 40 parámetros una corrida toma
horas. Con `checkpoint_path` el resultado parcial se escribe tras cada
iteración, con el mismo esquema que un resultado completo
(`stop_reason="en_progreso"`), así que se puede cargar y graficar sin esperar
a que termine.

### Gradiente analítico

El costo se evalúa con **gradiente analítico** por retropropagación sobre el
producto de exponenciales, en vez de las diferencias finitas que usa
`scipy.minimize` por defecto:

$$\frac{\partial E}{\partial\theta_j}=2\,\mathrm{Im}\langle\sigma_j\vert A_j\vert\phi_j\rangle,\qquad
\ket{\phi_j}=U_j\cdots U_1\ket{\psi_0},\quad \ket{\sigma_j}=R_j^\dagger H_C\ket{\psi}$$

con $R_j=U_m\cdots U_{j+1}$ y la recursión $\sigma_{j-1}=U_j^\dagger\sigma_j$.
Cuesta $O(m)$ productos matriz-vector para las $m$ derivadas en vez de
$O(m^2)$.

Es una decisión deliberada: en un estudio sobre gradientes no se quiere que el
optimizador se detenga por ruido de diferencias finitas en vez de por la
geometría real del paisaje. `verificar_gradiente()` lo chequea contra
diferencias finitas centradas.

### Diagnósticos

| Función | Qué calcula |
|---|---|
| `gradient_stats_at_random_inits` | Var$(\partial E/\partial\theta)$ en los mismos $\theta^0$ aleatorios guardados. Es el diagnóstico de *barren plateau* propiamente tal: mide si el paisaje se aplana en puntos aleatorios al crecer el ansatz. Es post-hoc, no requiere re-optimizar. |
| `qfim`, `qfim_stats`, `qfim_scan_aleatorio` | Matriz de información cuántica de Fisher, $k\times k$ con $k$ = nº de parámetros. Autovalores y rango. |
| `dla_dimension`, `dla_del_ansatz`, `dla_del_pool` | Dimensión del álgebra de Lie dinámica por clausura de Lie exacta. |
| `solucion_max3cut` | Decodifica la coloración Max-3-Cut del estado final y la verifica contra $\vert E_0\vert/2$. |
| `ansatz_gates`, `ansatz_latex` | Forma explícita de cada compuerta $e^{-i\theta_k G_k}$ con los $L$ por sitio. |

**Advertencia sobre la QFIM.** Por órbita-estabilizador,
$\mathrm{rank}\,F(\theta)\le\min(k,\dim\mathfrak g-\dim\mathfrak k)$. Un rango
igual a $k$ sólo establece la cota inferior $\dim\mathfrak g-\dim\mathfrak k\ge k$:
significa que aún no satura, **no** calcula el álgebra de Lie.

**Advertencia sobre los gradientes.** Los gradientes de selección a lo largo de
la trayectoria de ADAPT bajan porque el algoritmo converge. Eso **no** dice
nada sobre *barren plateaus*. El diagnóstico correcto es el panel de varianza
en puntos aleatorios.

### Métricas de costo

| Función | Definición | Por qué |
|---|---|---|
| `costo_compuertas` | $\sum_k 2(w_k-1)$, con $w_k$ = nº de sitios del operador $k$ | Comparar pools por nº de parámetros es injusto: un pool con operadores de mayor peso baja más el error por parámetro pero cada compuerta cuesta más. |
| `costo_mediciones` | $\vert\mathrm{pool}\vert\times$ iteraciones | Cada iteración de ADAPT mide el gradiente de **todo** el pool. En hardware suele ser el costo dominante. |

### Conteo de compuertas nativas

Sigue el conjunto de compuertas del procesador universal de qudits con iones
atrapados de Ringbauer *et al.*, *Nat. Phys.* **18**, 1053 (2022)
[arXiv:2109.06903], que es la referencia de hardware [34] del paper:

$$R^{(i,j)}(\theta,\varphi)=e^{-i\theta\sigma_\varphi^{(i,j)}/2},\qquad
\mathrm{MS}^{(i,j)}(\theta,\varphi)=e^{-i\frac{\theta}{4}\left(\sigma_\varphi^{(i,j)}\otimes\mathbb1+\mathbb1\otimes\sigma_\varphi^{(i,j)}\right)^2}$$

donde $(i,j)$ selecciona un **par de niveles**. Una unitaria arbitraria de un
qudit se descompone en $\mathcal O(d^2)$ rotaciones de dos niveles vía Givens;
para $d=3$ hacen falta a lo más 3 (los pares 01, 02, 12).

Para $e^{-i\theta G}$ con $G=\bigotimes_s A_s$ sobre $w$ sitios se usa la
escalera estándar: diagonalizar cada factor local, aplicar la fase con
$2(w-1)$ compuertas MS, y deshacer.

| Función | Qué da |
|---|---|
| `conteo_compuertas(label, base)` | rotaciones de dos niveles, MS, peso, y si la escalera aplica |
| `conteo_compuertas_ansatz(result)` | conteo acumulado del ansatz completo |
| `plot_conteo_compuertas({...}, umbral=1e-3)` | error vs. $R$, vs. MS y vs. total |

**Qué es exacto y qué es modelo.** Exacto: el número de rotaciones de dos
niveles de cada factor local, calculado de la matriz contando cuántos pares de
niveles conecta. Modelo: las $2(w-1)$ compuertas MS de la escalera,
extrapoladas de la construcción de qubits y de la compuerta Cex que
reporta ese trabajo. No es un circuito compilado.

**Por qué las dos bases difieren tanto.** Las matrices de Gell-Mann casi
coinciden con la compuerta nativa: $\lambda_3,\lambda_8$ son diagonales
(**0** rotaciones) y $\lambda_1,\lambda_2,\lambda_4,\lambda_5,\lambda_6,\lambda_7$
actúan sólo dentro de un par de niveles (**1** rotación). Los monomios de
momento angular hermitizados son hermíticas $3\times3$ genéricas que necesitan
la descomposición completa. Medido sobre el ansatz real:

```
angular  : media 5.88 rotaciones/generador,  {3: 4, 5: 6, 7: 1, 9: 5}
gellmann : media 3.87 rotaciones/generador,  {3: 17, 5: 13}
```

**Un costo oculto del pool angular.** Cuando dos o más sitios tienen factor
local no hermítico, $(O+O^\dagger)/2$ **no** es un producto tensorial sino una
suma de dos productos que no conmutan: la escalera simple no sirve y haría
falta trotterizar. `conteo_compuertas` lo señala con `es_producto=False`. En el
ansatz medido, 2 de 16 generadores angulares caen en ese caso y 0 de los de
Gell-Mann, por construcción.

### Métricas de aproximación

| Función | Definición |
|---|---|
| `razon_aproximacion(result)` | $r=\vert E_{\rm final}/E_0\vert$, la métrica de la Tabla I del paper (Ec. 35) |
| `resumen_tabla_i(result)` | fila comparable directamente contra la Tabla I |
| `plot_replica_fig1({...})` | réplica de la Fig. 1 con $\epsilon_{\rm rel}$ y los colores del paper |

### Figuras

```python
bp.figura_bp_con_grafo(res, relativo=True)   # paisaje + grafo con la solución
bp.plot_bp_gradients(res)                    # gradientes: trayectoria y aleatorios
bp.plot_comparacion_pools({...}, axes=axes)  # comparación de pools, 3 métricas
```

El `jitter` horizontal en el paisaje no es cosmético: en Max-3-Cut el espectro
es discreto y muy degenerado, así que muchas instancias caen en **exactamente**
el mismo mínimo. Sin jitter, una columna con 40 instancias en un mínimo se ve
igual que una con 1.

## `utilidades_gellmann.py` — pool en base de Gell-Mann

El pool se arma tomando los **términos individuales** de los conmutadores
anidados, así que depende de en qué base se los descomponga.

En la base de momento angular aparecen productos como $L_y^{(1)}L_z^{(1)}$ —dos
ejes en el mismo sitio— que no son hermíticos, y por eso el paper los hermitiza
con $(G+G^\dagger)/2$ (Ec. 28). En Gell-Mann eso no hace falta, porque el
álgebra cierra linealmente:

$$[\lambda_a,\lambda_b]=2i\sum_c f_{abc}\lambda_c$$

Cada término queda como producto de $\lambda$'s sobre sitios **distintos**, que
conmutan y son hermíticas: multiplicado por $i$, el generador es hermítico por
construcción.

Escrito en esta base el Hamiltoniano de costo es más compacto —**dos** términos
de interacción por arista en vez de cuatro— y sin parte de un solo sitio:

$$H_C\big\vert_{(i,j)}=\lambda_3^{(i)}\lambda_3^{(j)}+\lambda_8^{(i)}\lambda_8^{(j)}-\tfrac43\mathbb 1$$

Las constantes de estructura $f_{abc}$ y $d_{abc}$ se calculan **numéricamente**
de las matrices (`_estructura()`), no se tipean a mano.

Representación: un operador es un `dict {string: coeficiente}`, donde `string`
es una tupla de pares `(sitio, a)` ordenada por sitio y con sitios distintos.

## `utilidades_heisenberg.py` — pool en base de Heisenberg-Weyl

Tercera base, con $\omega=e^{2\pi i/3}$:

$$X\ket{k}=\ket{k+1\bmod 3},\qquad Z\ket{k}=\omega^k\ket{k},\qquad
(X^aZ^b)(X^cZ^d)=\omega^{bc}X^{a+c}Z^{b+d}$$

Sirve como control porque separa dos variables que en las otras dos bases van
juntas:

| Base | Descomposición | Elementos hermíticos |
|---|---|---|
| momento angular | gruesa | no → hay que hermitizar |
| Gell-Mann | fina | **sí** → no hace falta |
| Heisenberg-Weyl | fina | no → hay que hermitizar |

Si el pool de Gell-Mann converge peor porque sus elementos son *atómicos* (y no
por la hermiticidad), HW debería comportarse como Gell-Mann.

## Scripts de `cluster/`

### `main_bp.py`

Una corrida sobre un grafo. Guarda checkpoint en cada iteración.

```bash
# curva pura de Qudit-ADAPT (replica la Fig. 1a del paper)
python cluster/main_bp.py --grafo 1 --l 1 --max_iteration 40 --n_random 0

# con 100 instancias aleatorias por iteración
python cluster/main_bp.py --grafo 1 --l 2 --max_iteration 40 --n_random 100

# con otro pool
python cluster/main_bp.py --grafo 1 --l 2 --base gellmann
```

| Opción | Default | Qué hace |
|---|---|---|
| `--grafo` | 2 | índice (1-indexado) dentro de `--input_file` |
| `--l` | 2 | truncación del AGP: 1 usa $O_1$, 2 usa $O_1\cup O_3$ |
| `--n_random` | 100 | instancias aleatorias por iteración; 0 sólo da la curva |
| `--base` | `angular` | `angular`, `gellmann`, `gellmann_local`, `heisenberg` |
| `--n_jobs` | min(6,ncpu) | hilos para los reinicios |

**Sobre `--n_jobs`**: se midió que los hilos **no** dan speedup ($1.02\times$ a
$n=6$) porque el cuello es overhead de Python, no BLAS, así que el GIL nunca se
suelta. La paralelización que sí sirve es lanzar varios procesos
independientes. Se verificó que serial y paralelo dan resultados idénticos.

### `main_ensemble_pools.py`

Compara pools sobre muchos grafos. Reanudable: al relanzarlo salta los pares
(grafo, base) que ya estén en el CSV.

```bash
python cluster/main_ensemble_pools.py --start 1 --end 300 --l 2 --n_random 0

# repartido en 4 procesos
for i in 0 1 2 3; do
  python cluster/main_ensemble_pools.py --start $((i*75+1)) --end $(((i+1)*75)) \
      --l 2 --output ensemble_l2_part$i.csv &
done
```

## Cómo reproducir cada figura

**Figura 1a del paper** (curvas de Qudit-ADAPT, error relativo):

```bash
python cluster/main_bp.py --grafo 1 --l 1 --max_iteration 40 --n_random 0 \
    --output fig1a_l1_curva.json
python cluster/main_bp.py --grafo 1 --l 2 --max_iteration 40 --n_random 0 \
    --output fig1a_l2_curva.json
```

y graficar con `relativo=True`. El grafo 1 de `datos/grafos_comparacion.txt` es
el de la Fig. 1a: da $\epsilon_{\rm rel}=6.56\times10^{-3}$ con $\ell=1$ en 21
parámetros y $6.15\times10^{-4}$ con $\ell=2$ en 40, que es donde están las
curvas roja y azul del paper.

**Análisis de entrenabilidad** (reciclado vs. frío vs. aleatorio):

```bash
python cluster/main_bp.py --grafo 1 --l 2 --max_iteration 40 --n_random 100 \
    --output fig1a_l2_bp100.json
```

Costo: con 40 parámetros y 100 instancias el costo escala como $\sum_k k^2$ y
la corrida toma del orden de horas. Por eso el checkpoint.

## Validaciones

Todas superadas y reproducibles desde el cuadernillo:

| Qué | Contra qué | Resultado |
|---|---|---|
| gradiente analítico | diferencias finitas | $4\times10^{-9}$ |
| energía del ansatz | `qutip` | $9\times10^{-16}$ |
| pool contradiabático | `cd_adapt_vqe_algorithm_profundo` | idéntico elemento a elemento |
| curva reciclada | `energy_trace` del original | $6\times10^{-11}$ |
| generadores del ansatz | reconstrucción desde la etiqueta | $0$ exacto |
| $H_C$, $H_M$ en Gell-Mann y HW | implementación original | $\sim10^{-15}$ |
| $f_{abc}$ de Gell-Mann | valores tabulados | exactas |
| DLA | $\mathfrak{su}(2)\to3$, $\mathfrak{su}(3)\to8$, abelianas | correctas |
| QFIM | $F_{11}=4\,\mathrm{Var}(A)$ | $10^{-16}$ |

## Notas de implementación

Cosas que costaron y conviene no volver a tropezar:

**Empates en `argmax`.** Los índices de operador seleccionados difieren de los
del código original. No es un error: en grafos simétricos varios operadores
empatan en el gradiente hasta $\sim10^{-15}$ y `argmax` rompe el empate según
el último bit. Los operadores son equivalentes por simetría y las energías
coinciden.

**$\lambda$ genérico al construir el pool.** El código original mantiene
$\lambda$ simbólico (`sympy`), así que un término desaparece sólo si su
coeficiente se anula como polinomio. Al evaluar numéricamente en un valor
particular pueden cancelarse términos por accidente: con $\lambda=0.5$ el pool
$\ell=2$ de Gell-Mann perdía 24 strings (888 en vez de 912). Los módulos de
Gell-Mann y HW evalúan en **dos** valores genéricos y toman la unión.

**Cota del DLA.** Los operadores del pool **no** son de traza nula
($\mathrm{Tr}\,L_z^2\neq0$), así que el álgebra puede contener la identidad y la
cota dura es $\dim\mathfrak u(D)=D^2$, no $\dim\mathfrak{su}(D)=D^2-1$.

**Degeneración y la solución decodificada.** El fundamental de $H_C$ es muy
degenerado (180 veces en uno de los grafos). El estado final es una
superposición de muchas coloraciones óptimas equivalentes, y la que reporta
`solucion_max3cut` es la del estado de la base computacional más probable. Con
ruido numérico distinto (p.ej. otro número de hilos BLAS) el `argmax` puede
elegir otra coloración igualmente óptima.

**Convención de colores del grafo.** En Max-3-Cut las clases deben ser
conjuntos **independientes**: cada arista *dentro* de una clase es una arista
sin cortar. Que dos vértices del mismo color no estén unidos es la señal de que
la solución es buena, no un error.
