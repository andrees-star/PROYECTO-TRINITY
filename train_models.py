import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)

# =========================
# 1. Configuración
# =========================

INPUT_FILE = "data/2df_final_completo_23.xlsx"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "riesgo_24"
THRESHOLD = 0.40  # Umbral de prueba para regresión logística

print("Entrenando modelo de Regresión Logística")
print(f"Archivo: {INPUT_FILE}")
print(f"Target: {TARGET}")

# =========================
# 2. Leer base
# =========================

df = pd.read_excel(INPUT_FILE)

print(f"Filas originales: {df.shape[0]}")
print(f"Columnas originales: {df.shape[1]}")

# =========================
# 3. Variables del modelo
# =========================

variables_modelo = [
    "raz",
    "teso",
    "rota",
    "margenb",
    "margen",
    "ractiv",
    "rpatri",
    "niven",
    "apalc",
    "apaltot",
    "activos_pasivos",
    "pasivo_corto_pasivo_total",
    "margen_operacional",
    "ctno_ventas_preciso"
]

id_cols_posibles = [
    "NIT",
    "Razón social de la sociedad_balance"
]

columnas_necesarias = variables_modelo + [TARGET]

columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]

if columnas_faltantes:
    raise ValueError(f"Faltan columnas en la base: {columnas_faltantes}")

# =========================
# 4. Preparar datos
# =========================

base = df.copy()

for col in variables_modelo:
    base[col] = pd.to_numeric(base[col], errors="coerce")

base[TARGET] = pd.to_numeric(base[TARGET], errors="coerce")

base = base.replace([np.inf, -np.inf], np.nan)

# Quitar filas sin target
base = base.dropna(subset=[TARGET])

# Dejar solo clases 0 y 1
base = base[base[TARGET].isin([0, 1])]

X = base[variables_modelo]
y = base[TARGET].astype(int)

print(f"Filas usadas para modelar: {base.shape[0]}")
print("Distribución de clases:")
print(y.value_counts())

# =========================
# 5. Train / validation
# =========================

X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X,
    y,
    base.index,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Filas entrenamiento: {X_train.shape[0]}")
print(f"Filas validación: {X_val.shape[0]}")

# =========================
# 6. Pipeline logístico
# =========================

modelo_logistico = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=True)),
    ("logistica", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    ))
])

# =========================
# 7. Entrenar
# =========================

modelo_logistico.fit(X_train, y_train)

# =========================
# 8. Predecir
# =========================

proba_val = modelo_logistico.predict_proba(X_val)[:, 1]
pred_val = (proba_val >= THRESHOLD).astype(int)

# =========================
# 9. Métricas
# =========================

tn, fp, fn, tp = confusion_matrix(y_val, pred_val).ravel()

precision = precision_score(y_val, pred_val, zero_division=0)
recall = recall_score(y_val, pred_val, zero_division=0)
f1 = f1_score(y_val, pred_val, zero_division=0)
mcc = matthews_corrcoef(y_val, pred_val)
roc_auc = roc_auc_score(y_val, proba_val)
pr_auc = average_precision_score(y_val, proba_val)

error_tipo_i = fp / (fp + tn) if (fp + tn) > 0 else np.nan
error_tipo_ii = fn / (fn + tp) if (fn + tp) > 0 else np.nan
especificidad = tn / (tn + fp) if (tn + fp) > 0 else np.nan

metricas = pd.DataFrame([{
    "modelo": "Regresion Logistica",
    "target": TARGET,
    "threshold": THRESHOLD,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "mcc": mcc,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "error_tipo_i": error_tipo_i,
    "error_tipo_ii": error_tipo_ii,
    "especificidad": especificidad,
    "tp": tp,
    "fn": fn,
    "fp": fp,
    "tn": tn
}])

matriz_confusion = pd.DataFrame({
    "Real": ["No insolvente", "No insolvente", "Insolvente", "Insolvente"],
    "Prediccion": ["No insolvente", "Insolvente", "No insolvente", "Insolvente"],
    "Cantidad": [tn, fp, fn, tp]
})

# =========================
# 10. Predicciones de validación
# =========================

cols_id_existentes = [col for col in id_cols_posibles if col in base.columns]

predicciones = base.loc[idx_val, cols_id_existentes].copy()
predicciones["real_riesgo_24"] = y_val.values
predicciones["probabilidad_riesgo_24"] = proba_val
predicciones["prediccion_riesgo_24"] = pred_val

def clasificar_alerta(prob):
    if prob < 0.10:
        return "Bajo"
    elif prob < 0.25:
        return "Medio"
    elif prob < 0.40:
        return "Alto"
    else:
        return "Critico"

predicciones["nivel_alerta"] = predicciones["probabilidad_riesgo_24"].apply(clasificar_alerta)

# =========================
# 11. Guardar resultados
# =========================

output_excel = OUTPUT_DIR / "resultados_regresion_logistica.xlsx"

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    metricas.to_excel(writer, sheet_name="metricas", index=False)
    matriz_confusion.to_excel(writer, sheet_name="matriz_confusion", index=False)
    predicciones.to_excel(writer, sheet_name="predicciones_val", index=False)
    pd.DataFrame({"variable": variables_modelo}).to_excel(writer, sheet_name="variables", index=False)

joblib.dump(modelo_logistico, OUTPUT_DIR / "modelo_regresion_logistica.joblib")

print("Modelo entrenado correctamente")
print(f"Resultados guardados en: {output_excel}")
print("Modelo guardado en: outputs/modelo_regresion_logistica.joblib")
