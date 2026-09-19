"""
Regresión lineal: demora (días) entre inicio de síntomas y consulta
Evento 155 - Cáncer de la mama y cuello uterino, Colombia 2025 (microdatos INS).

Ejecutar desde la carpeta raíz del repositorio:
    python src/analisis_regresion.py
Requiere el archivo data/Datos_2025_155.xlsx (ver data/LEEME_DATOS.md).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

SEED = 42                                   # semilla de aleatoriedad
RUTA_DATOS = os.path.join("data", "Datos_2025_155.xlsx")   # ruta relativa
RUTA_FIG = "figures"
os.makedirs(RUTA_FIG, exist_ok=True)


def guardar(nombre):
    plt.savefig(os.path.join(RUTA_FIG, nombre), dpi=150, bbox_inches="tight")
    plt.close()


# 1. Carga ------------------------------------------------------------------
datos = pd.read_excel(RUTA_DATOS)
print("Registros y columnas:", datos.shape)

# 2. Calidad: duplicados ------------------------------------------------------
print("Filas duplicadas:", datos.duplicated().sum())
print("Consecutivos únicos:", datos["CONSECUTIVE"].nunique())

# 3. Variable objetivo ---------------------------------------------------------
for col in ["INI_SIN", "FEC_CON"]:
    datos[col] = pd.to_datetime(datos[col].str[:10].str.strip(), format="%d/%m/%Y")
datos["demora"] = (datos["FEC_CON"] - datos["INI_SIN"]).dt.days
print("Vacíos en fechas:", datos[["INI_SIN", "FEC_CON"]].isna().sum().sum())
print("Demoras negativas:", (datos["demora"] < 0).sum())

# 4. Limpieza -----------------------------------------------------------------
datos["estrato"] = datos["estrato"].astype(str).str.strip()
limpios = datos[(datos["demora"] <= 365) & (datos["estrato"] != "")
                & (datos["Departamento_residencia"] != "EXTERIOR")].copy()
limpios["estrato"] = limpios["estrato"].astype(int)
print("Excluidos:", len(datos) - len(limpios), "| Analizados:", len(limpios))

# 5. EDA ---------------------------------------------------------------------
print(limpios[["EDAD", "estrato", "demora"]].describe().round(1))
print(limpios[["EDAD", "estrato", "demora"]].corr().round(3))
print("Asimetría de la demora:", round(stats.skew(limpios["demora"]), 2))

plt.hist(limpios["demora"], bins=30)
plt.xlabel("Demora síntomas → consulta (días)"); plt.ylabel("Pacientes")
plt.title("Distribución de la demora"); guardar("01_histograma_demora.png")

limpios.groupby("TIP_SS")["demora"].median().sort_values().plot(kind="bar")
plt.xlabel("Régimen"); plt.ylabel("Mediana de demora (días)")
plt.title("Demora por régimen"); guardar("02_demora_regimen.png")

etiquetas_area = {1: "Cabecera", 2: "Centro poblado", 3: "Rural disperso"}
grupos = [limpios.loc[limpios["AREA"] == a, "demora"] for a in [1, 2, 3]]
plt.boxplot(grupos, showfliers=False)
plt.xticks([1, 2, 3], [etiquetas_area[a] for a in [1, 2, 3]])
plt.ylabel("Demora (días)"); plt.title("Demora por área de residencia")
guardar("03_boxplot_area.png")
print(limpios.groupby("AREA")["demora"].agg(["count", "median", "mean"]).round(1))

dep = limpios.groupby("Departamento_residencia")["demora"].agg(["count", "median"])
dep = dep[dep["count"] >= 100].sort_values("median")
dep["median"].plot(kind="barh", figsize=(6, 7))
plt.xlabel("Mediana de demora (días)"); plt.ylabel("")
plt.title("Demora por departamento (≥100 casos)"); guardar("04_demora_departamento.png")
print(dep)

# 6. Preparación de X y y -----------------------------------------------------
def preparar(columnas, categoricas, referencias):
    X = limpios[columnas].copy()
    X = pd.get_dummies(X, columns=categoricas, dtype=int)
    return X.drop(columns=referencias)

# Modelo A (inicial): 6 predictores, incluye EAPB
XA = preparar(["Departamento_residencia", "EDAD", "TIP_SS", "COD_ASE", "estrato", "AREA"],
              ["Departamento_residencia", "TIP_SS", "COD_ASE", "AREA"],
              ["Departamento_residencia_BOGOTA", "TIP_SS_C", "COD_ASE_EPS005", "AREA_1"])
# Modelo B (final): 5 predictores, sin EAPB
XB = preparar(["Departamento_residencia", "EDAD", "TIP_SS", "estrato", "AREA"],
              ["Departamento_residencia", "TIP_SS", "AREA"],
              ["Departamento_residencia_BOGOTA", "TIP_SS_C", "AREA_1"])
y = limpios["demora"]
print("Matriz A:", XA.shape, "| Matriz B:", XB.shape)

# Cada EAPB pertenece casi siempre a un solo régimen
print("Regímenes por EAPB:", limpios.groupby("COD_ASE")["TIP_SS"].nunique().value_counts().to_dict())

# 7. Multicolinealidad (VIF) --------------------------------------------------
def vif(X):
    Xc = X.astype(float).copy()
    Xc.insert(0, "const", 1.0)
    return pd.Series([variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])],
                     index=X.columns)

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vif_A, vif_B = vif(XA), vif(XB)
print("Modelo A - VIF máximo:", f"{vif_A.max():.2e}", "| variables con VIF > 10:", (vif_A > 10).sum())
print("Modelo B - VIF máximo:", round(vif_B.max(), 2), "| variables con VIF > 5:", (vif_B > 5).sum())
print(vif_B.loc[["EDAD", "estrato", "TIP_SS_S", "AREA_3"]].round(2))

# 8. División, entrenamiento, predicción -------------------------------------
def metricas(real, pred):
    mse = mean_squared_error(real, pred)
    return {"MAE": mean_absolute_error(real, pred), "MSE": mse,
            "RMSE": np.sqrt(mse), "R2": r2_score(real, pred)}

resultados = {}
for nombre, X in {"A": XA, "B": XB}.items():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)
    m = LinearRegression().fit(X_train, y_train)
    resultados[nombre] = dict(modelo=m, X=X, X_test=X_test, y_test=y_test,
                              y_train=y_train, y_pred=m.predict(X_test))
    resultados[nombre]["tabla"] = pd.DataFrame({
        "Entrenamiento": metricas(y_train, m.predict(X_train)),
        "Prueba": metricas(y_test, resultados[nombre]["y_pred"])})
    print(f"Modelo {nombre}"); print(resultados[nombre]["tabla"].round(3))

# Línea base: predecir siempre la media del entrenamiento
b = resultados["B"]
base = np.full(len(b["y_test"]), b["y_train"].mean())
print("Línea base MAE:", round(mean_absolute_error(b["y_test"], base), 2),
      "| RMSE:", round(np.sqrt(mean_squared_error(b["y_test"], base)), 2))

# 9. Gráficos de evaluación y supuestos (modelo final B) ---------------------
y_test, y_pred = b["y_test"], b["y_pred"]
residuos = y_test - y_pred

plt.scatter(y_test, y_pred, alpha=0.3, s=5)
plt.plot([0, 365], [0, 365], color="red")
plt.xlabel("Demora real (días)"); plt.ylabel("Demora predicha (días)")
plt.title("Demora real vs predicha (modelo B)"); guardar("05_real_vs_predicha.png")

plt.scatter(y_pred, residuos, alpha=0.3, s=5)
plt.axhline(0, color="red")
plt.xlabel("Demora predicha (días)"); plt.ylabel("Residuo (días)")
plt.title("Residuos vs predicciones (modelo B)"); guardar("06_residuos_vs_predicciones.png")

tramos = pd.cut(y_pred, [-1, 40, 60, 80, 500])
print(pd.Series(residuos.values).groupby(tramos, observed=True).agg(["count", "mean", "std"]).round(1))

plt.hist(residuos, bins=40)
plt.xlabel("Residuo (días)"); plt.ylabel("Pacientes")
plt.title("Distribución de los residuos"); guardar("07_histograma_residuos.png")

stats.probplot(residuos, dist="norm", plot=plt)
plt.xlabel("Cuantiles teóricos (normal)"); plt.ylabel("Residuos ordenados (días)")
plt.title("Gráfico Q-Q de los residuos"); guardar("08_qq_residuos.png")
print("Asimetría de residuos:", round(stats.skew(residuos), 2),
      "| Curtosis:", round(stats.kurtosis(residuos), 2))
muestra = residuos.sample(min(5000, len(residuos)), random_state=SEED)
print("Shapiro-Wilk (muestra de residuos) p =", f"{stats.shapiro(muestra).pvalue:.1e}")

# 10. Coeficientes del modelo final -------------------------------------------
coef = pd.Series(b["modelo"].coef_, index=b["X"].columns)
print(coef.loc[["EDAD", "estrato", "TIP_SS_S", "TIP_SS_P", "AREA_2", "AREA_3"]].round(2))
conteo = limpios["Departamento_residencia"].value_counts()
cd = coef[coef.index.str.startswith("Departamento_residencia_")]
cd.index = cd.index.str.replace("Departamento_residencia_", "")
cd = cd[conteo.reindex(cd.index) >= 100].sort_values()
print("Departamentos con ≥100 casos, diferencia frente a Bogotá:")
print(cd.round(1).head(4)); print(cd.round(1).tail(5))
