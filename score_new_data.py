import numpy as np
import pandas as pd
from pathlib import Path
import joblib

# =========================
# 1. Configuración
# =========================

NEW_DATA_FILE = "data/new_firms_5.xlsx"
MODEL_FILE = "outputs/modelo_regresion_logistica.joblib"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.40

print("Iniciando scoring de nuevas empresas")
print(f"Archivo nuevo: {NEW_DATA_FILE}")
print(f"Modelo: {MODEL_FILE}")

# =========================
# 2. Variables del modelo
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

# =========================
# 3. Leer nueva data
# =========================

df_new = pd.read_excel(NEW_DATA_FILE)

print(f"Empresas nuevas cargadas: {df_new.shape[0]}")
print(f"Columnas recibidas: {df_new.shape[1]}")

columnas_faltantes = [col for col in variables_modelo if col not in df_new.columns]

if columnas_faltantes:
    raise ValueError(f"Faltan columnas en el Excel nuevo: {columnas_faltantes}")

# =========================
# 4. Preparar variables
# =========================

X_new = df_new[variables_modelo].copy()

for col in variables_modelo:
    X_new[col] = pd.to_numeric(X_new[col], errors="coerce")

X_new = X_new.replace([np.inf, -np.inf], np.nan)

# =========================
# 5. Cargar modelo entrenado
# =========================

modelo = joblib.load(MODEL_FILE)

# =========================
# 6. Predecir riesgo
# =========================

probabilidades = modelo.predict_proba(X_new)[:, 1]
predicciones = (probabilidades >= THRESHOLD).astype(int)

def clasificar_alerta(prob):
    if prob < 0.10:
        return "Bajo"
    elif prob < 0.25:
        return "Medio"
    elif prob < 0.40:
        return "Alto"
    else:
        return "Critico"

# =========================
# 7. Crear salida
# =========================

cols_id_existentes = [col for col in id_cols_posibles if col in df_new.columns]

resultado = df_new[cols_id_existentes].copy()
resultado["probabilidad_riesgo_24"] = probabilidades
resultado["prediccion_riesgo_24"] = predicciones
resultado["nivel_alerta"] = resultado["probabilidad_riesgo_24"].apply(clasificar_alerta)

# Agregar variables financieras para revisión
for col in variables_modelo:
    resultado[col] = df_new[col]

# =========================
# 8. Guardar resultados
# =========================

output_file = OUTPUT_DIR / "scoring_new_firms.xlsx"
resultado.to_excel(output_file, index=False)

print("Scoring finalizado correctamente")
print(f"Archivo generado: {output_file}")
