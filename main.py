import pandas as pd
from pathlib import Path

# =========================
# 1. Configuración de rutas
# =========================

INPUT_FILE = "data/2df_final_completo_23.xlsx"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Iniciando pipeline de diagnóstico financiero")
print(f"Leyendo archivo: {INPUT_FILE}")

# =========================
# 2. Leer archivo Excel
# =========================

df = pd.read_excel(INPUT_FILE)

print("Archivo cargado correctamente")
print(f"Filas: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")

# =========================
# 3. Resumen de columnas
# =========================

resumen_columnas = pd.DataFrame({
    "columna": df.columns,
    "tipo_dato": [df[col].dtype for col in df.columns],
    "nulos": [df[col].isna().sum() for col in df.columns],
    "porcentaje_nulos": [round(df[col].isna().mean() * 100, 2) for col in df.columns],
    "valores_unicos": [df[col].nunique(dropna=True) for col in df.columns]
})

# =========================
# 4. Resumen de nulos
# =========================

resumen_nulos = resumen_columnas.sort_values(
    by="porcentaje_nulos",
    ascending=False
)

# =========================
# 5. Estadísticas de variables numéricas
# =========================

variables_numericas = df.select_dtypes(include=["number"])

if not variables_numericas.empty:
    estadisticas_numericas = variables_numericas.describe().T
else:
    estadisticas_numericas = pd.DataFrame({
        "mensaje": ["No se detectaron variables numéricas"]
    })

# =========================
# 6. Muestra de datos
# =========================

muestra = df.head(100)

# =========================
# 7. Guardar resultados
# =========================

with pd.ExcelWriter(OUTPUT_DIR / "diagnostico_base_financiera.xlsx", engine="openpyxl") as writer:
    resumen_columnas.to_excel(writer, sheet_name="columnas", index=False)
    resumen_nulos.to_excel(writer, sheet_name="nulos", index=False)
    estadisticas_numericas.to_excel(writer, sheet_name="estadisticas")
    muestra.to_excel(writer, sheet_name="muestra_100", index=False)

print("Diagnóstico generado correctamente")
print("Archivo creado: outputs/diagnostico_base_financiera.xlsx")
