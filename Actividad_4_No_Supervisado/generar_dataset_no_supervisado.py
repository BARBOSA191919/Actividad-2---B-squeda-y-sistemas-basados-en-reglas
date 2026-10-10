import pandas as pd
import numpy as np

# Fijamos una semilla para que los datos sean reproducibles
np.random.seed(42)
n = 300  # Cantidad de registros simulados

# Paradas reales de Neiva utilizadas en el proyecto
paradas = [
    "Terminal de Transporte", 
    "Aeropuerto Benito Salas", 
    "Parque Santander", 
    "San Pedro Plaza", 
    "Estadio Plazas Alcid", 
    "Universidad Surcolombiana"
]

# Creamos el DataFrame con las variables requeridas para K-Means
datos = pd.DataFrame({
    "parada": np.random.choice(paradas, n),
    "pasajeros_promedio": np.random.randint(10, 120, n),
    "tiempo_espera_min": np.random.randint(2, 25, n),
    "velocidad_promedio_kmh": np.random.uniform(10.0, 35.0, n).round(1),
    "num_transbordos_frecuentes": np.random.randint(0, 4, n)
})

# Guardamos el resultado en un archivo CSV dentro de la carpeta
datos.to_csv("dataset_no_supervisado.csv", index=False)

print("Dataset de aprendizaje no supervisado generado con éxito.")
print("Dimensiones del dataset:", datos.shape)
print("\nPrimeros registros:")
print(datos.head())