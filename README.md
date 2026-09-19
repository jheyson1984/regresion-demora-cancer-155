# Regresión lineal: demora síntomas → consulta en cáncer de mama y cuello uterino (Colombia, 2025)

Curso: Introducción a Machine Learning (604027) — Especialización en Analítica y Ciencia de Datos, Universidad de Cundinamarca.
Autor: Jheyson Morales.

## Objetivo
Estimar con regresión lineal la demora, en días, entre el inicio de síntomas y la consulta en casos del evento 155 notificados a SIVIGILA en 2025, e identificar qué variables sociodemográficas y territoriales se asocian con ella.

## Datos
- Fuente: Instituto Nacional de Salud (INS), Portal SIVIGILA, búsqueda de microdatos.
- Evento 155, año 2025: 18.717 registros, 69 columnas. Base nominal depurada, sin identificación personal.
- La base no se incluye en el repositorio. Ver `data/LEEME_DATOS.md` para descargarla.

## Estructura
```
.
├── data/
│   └── LEEME_DATOS.md                 # cómo obtener la base (no se publica)
├── src/
│   └── analisis_regresion.py          # flujo completo y reproducible
├── figures/                           # 8 gráficos exportados por el script
├── Regresion_demora_cancer_155.ipynb  # cuaderno de exploración inicial (Colab)
├── requirements.txt
└── README.md
```

## Pasos de ejecución
1. Python 3.10 o superior.
2. `pip install -r requirements.txt`
3. Descargar la base y guardarla como `data/Datos_2025_155.xlsx`.
4. Desde la carpeta raíz del repositorio: `python src/analisis_regresion.py`
   El script imprime los resultados y guarda los gráficos en `figures/`.

## Flujo del análisis
1. Carga y control de calidad (duplicados, vacíos, fechas inconsistentes).
2. Variable objetivo: `demora = FEC_CON − INI_SIN` (días).
3. Limpieza: exclusión de demoras > 365 días, estrato vacío y residencia en el exterior (17.326 casos analizados).
4. EDA: estadísticos, correlaciones, asimetría y gráficos por régimen, área y departamento.
5. Codificación de variables categóricas (referencias: Bogotá, régimen contributivo, cabecera).
6. Multicolinealidad (VIF): el modelo inicial con EAPB presentó colinealidad perfecta con el régimen; el modelo final excluye la EAPB.
7. División 75/25 con semilla fija (`random_state=42`), regresión lineal, métricas en entrenamiento y prueba, comparación con línea base.
8. Supuestos: linealidad, homocedasticidad y normalidad de residuos (gráficos y prueba de Shapiro-Wilk).

## Resultados del modelo final (conjunto de prueba)
| Métrica | Valor |
|---|---|
| MAE | 49,44 días |
| MSE | 4.547,49 |
| RMSE | 67,44 días |
| R² | 0,057 |

## Reproducibilidad
- Semilla fija: `random_state=42`.
- Rutas relativas (`data/`, `figures/`); no se usan rutas absolutas.
