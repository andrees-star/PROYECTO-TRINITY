import pandas as pd
from pathlib import Path

# =========================
# 1. Rutas
# =========================

INPUT_FILE = "data/2df_final_completo_23.xlsx"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Iniciando pipeline de preparación de base modelo")
print(f"Leyendo archivo: {INPUT_FILE}")

# =========================
# 2. Leer Excel
# =========================

df = pd.read_excel(INPUT_FILE)

print("Archivo cargado correctamente")
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

variables_objetivo = [
    "riesgo_24",
    "riesgo_2425"
]

variables_identificacion = [
    "NIT",
    "Razón social de la sociedad_balance"
]

# =========================
# 4. Validar columnas
# =========================

columnas_necesarias = variables_identificacion + variables_modelo + variables_objetivo

columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]

if columnas_faltantes:
    raise ValueError(f"Faltan columnas en la base: {columnas_faltantes}")

print("Todas las columnas necesarias están disponibles")

# =========================
# 5. Crear base modelo
# =========================

base_modelo = df[columnas_necesarias].copy()

# Convertir variables financieras a numéricas
for col in variables_modelo:
    base_modelo[col] = pd.to_numeric(base_modelo[col], errors="coerce")

# Convertir variables objetivo a numéricas
for col in variables_objetivo:
    base_modelo[col] = pd.to_numeric(base_modelo[col], errors="coerce")

# =========================
# 6. Reporte de nulos
# =========================

reporte_nulos = pd.DataFrame({
    "variable": base_modelo.columns,
    "nulos": base_modelo.isna().sum().values,
    "porcentaje_nulos": (base_modelo.isna().mean().values * 100).round(2)
}).sort_values(by="porcentaje_nulos", ascending=False)

# =========================
# 7. Base completa sin nulos en variables del modelo
# =========================

base_modelo_completa = base_modelo.dropna(subset=variables_modelo + variables_objetivo)

print(f"Filas base modelo: {base_modelo.shape[0]}")
print(f"Filas base modelo completa: {base_modelo_completa.shape[0]}")

# =========================
# 8. Distribución de clases
# =========================

dist_riesgo_24 = base_modelo_completa["riesgo_24"].value_counts(dropna=False).reset_index()
dist_riesgo_24.columns = ["riesgo_24", "cantidad"]

dist_riesgo_2425 = base_modelo_completa["riesgo_2425"].value_counts(dropna=False).reset_index()
dist_riesgo_2425.columns = ["riesgo_2425", "cantidad"]

# =========================
# 9. Guardar resultados
# =========================

with pd.ExcelWriter(OUTPUT_DIR / "base_modelo_insolvencia.xlsx", engine="openpyxl") as writer:
    base_modelo.to_excel(writer, sheet_name="base_modelo", index=False)
    base_modelo_completa.to_excel(writer, sheet_name="base_completa", index=False)
    reporte_nulos.to_excel(writer, sheet_name="reporte_nulos", index=False)
    dist_riesgo_24.to_excel(writer, sheet_name="distribucion_2024", index=False)
    dist_riesgo_2425.to_excel(writer, sheet_name="distribucion_2425", index=False)

print("Archivo generado correctamente:")
print("outputs/base_modelo_insolvencia.xlsx")
