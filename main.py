import pandas as pd
from pathlib import Path

# Crear carpeta de resultados
Path("outputs").mkdir(exist_ok=True)

# Leer Excel desde la carpeta data
df = pd.read_excel("data/2df_final_completo_23.xlsx")

# Vista rápida
print("Archivo cargado correctamente")
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])
print(df.head())

# Guardar una prueba de salida
df.head(100).to_csv("outputs/prueba_carga_excel.csv", index=False)

print("Archivo de salida generado correctamente en outputs/prueba_carga_excel.csv")
