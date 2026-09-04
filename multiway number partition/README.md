# Multiway Number Partitioning (k=3) en qutrits

Línea exploratoria del proyecto [Qudit-Adapt](../README.md): repartir $N$
números positivos en $k=3$ partes lo más parejas posible, usando **CD-ADAPT-VQE**
y **QAOA** con un qutrit por elemento (a diferencia de qubits, donde etiquetar
3 valores necesita one-hot + penalización auxiliar, acá el estado del qutrit
*es* la etiqueta). Derivación completa del Hamiltoniano de costo en
[`formulacion_qubo.tex`](formulacion_qubo.tex) / [`formulacion_qubo.pdf`](formulacion_qubo.pdf).

Es autocontenida: no depende de `cluster/`, `cuadernillos/`, `datos/` ni
`resultados/` de la raíz del proyecto, tiene su propia copia de ese patrón
acá adentro.

## Estructura

```
multiway number partition/
├── utilidades/utilidades_multiway.py   Motor: Hp, pool CD, ADAPT, QAOA (reutiliza funciones/ de la raíz)
├── cluster/                             Scripts batch: main_multiway.py (CD-ADAPT-VQE), main_multiway_qaoa.py (QAOA)
├── datos/                               Instancias de prueba (.txt) + caché del toy
├── resultados/csv|json/                 Resultados ya corridos (CD-ADAPT l=1,2 y QAOA p=1..10, n=5,6)
├── comparacion_multiway.ipynb           Notebook de análisis: toy + batch de 20 instancias
└── formulacion_qubo.tex / .pdf          Derivación del Hamiltoniano de costo
```

## Uso rápido

Desde esta carpeta:

```bash
python cluster/main_multiway.py --instancias_file datos/casos_n5.txt --l 1
python cluster/main_multiway_qaoa.py --instancias_file datos/casos_n5.txt --p_max 10
```

Para el detalle de resultados y conclusiones, ver `comparacion_multiway.ipynb`.
