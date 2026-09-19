# Regresión lineal: demora síntomas → consulta en cáncer de mama y cuello uterino (Colombia, 2025)

Curso: Introducción a Machine Learning (604027) — Especialización en Analítica y Ciencia de Datos, Universidad de Cundinamarca.
Autor: Jheyson Morales.

## Objetivo
Estimar, con regresión lineal, la demora en días entre el inicio de síntomas y la consulta en casos notificados a SIVIGILA del evento 155 (cáncer de la mama y cuello uterino), usando como predictores: departamento de residencia, edad, régimen de afiliación, EAPB, estrato y área.

## Datos
- Fuente: Instituto Nacional de Salud, Portal SIVIGILA, búsqueda de microdatos: https://portalsivigila.ins.gov.co/Paginas/Buscador.aspx
- Evento: Cáncer de la mama y cuello uterino. Año: 2025. Archivo: `Datos_2025_155.xlsx` (18.717 registros, 69 columnas).
- La base de datos NO se incluye en este repositorio. Para reproducir el análisis, descárguela desde el portal del INS (requiere diligenciar un formulario de registro) y guárdela con el nombre `Datos_2025_155.xlsx` en la misma carpeta del cuaderno.

## Estructura
```
.
├── Regresion_demora_cancer_155.ipynb   # cuaderno con todo el flujo
├── figuras/                            # gráficos generados por el cuaderno
├── requirements.txt
└── README.md
```

## Pasos de ejecución
1. Python 3.10 o superior.
2. Instalar dependencias: `pip install -r requirements.txt`
3. Colocar `Datos_2025_155.xlsx` en la carpeta del proyecto (rutas relativas; no se usan rutas absolutas).
4. Abrir y ejecutar `Regresion_demora_cancer_155.ipynb` de arriba hacia abajo (en Google Colab: subir el cuaderno y el archivo de datos, y usar "Ejecutar todo").

## Reproducibilidad
- Semilla de aleatoriedad fijada: `random_state=42` en la división entrenamiento/prueba (75 % / 25 %).

## Resultados principales (conjunto de prueba)
| Métrica | Valor |
|---|---|
| MAE | 48,85 días |
| MSE | 4.523,21 |
| RMSE | 67,25 días |
| R² | 0,062 |
