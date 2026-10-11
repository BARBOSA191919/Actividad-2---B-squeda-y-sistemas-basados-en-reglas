
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path


# Ubicar y cargar el archivo de datos
ruta_dataset = Path(__file__).with_name("dataset_no_supervisado.csv")
datos = pd.read_csv(ruta_dataset)

# Mostrar las primeras cinco filas
print(datos.head())

# Seleccionar las variables numéricas para el modelo
variables = [
    "pasajeros_promedio",
    "tiempo_espera_min",
    "velocidad_promedio_kmh",
    "num_transbordos_frecuentes"
]

X = datos[variables]

# Mostrar las variables seleccionadas
print("\nVariables seleccionadas para el modelo:")
print(X.head())

# Normalizar las variables numéricas
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

print("\nPrimeras cinco filas de los datos normalizados:")
print(X_escalado[:5])


# Crear y entrenar el modelo K-Means
modelo = KMeans(n_clusters=3, random_state=42, n_init=10)
modelo.fit(X_escalado)

# Guardar el grupo asignado a cada fila
datos["grupo"] = modelo.labels_

print("\nGrupos asignados por el modelo:")
print(datos[["parada", "grupo"]].head(10))

# Analizar las características promedio de cada grupo
print("\nResumen de características por grupo:")
print(datos.groupby("grupo")[variables].mean().round(2))


# Guardar los resultados del agrupamiento
ruta_resultados = Path(__file__).with_name("resultados_kmeans.csv")
datos.to_csv(ruta_resultados, index=False)

print(f"\nResultados guardados en: {ruta_resultados.name}")
