import pandas as pd
import numpy as np

# Para generar los mismos datos en cada ejecución
np.random.seed(42)

# Cantidad de registros
n = 500

# Paradas disponibles
paradas = [
    "Terminal",
    "Aeropuerto",
    "Parque Santander",
    "San Pedro Plaza",
    "Estadio",
    "Universidad"
]

# Creamos el DataFrame con datos simulados
datos = pd.DataFrame({
    "parada_origen": np.random.choice(paradas, n),
    "parada_destino": np.random.choice(paradas, n),
    "dia_semana": np.random.choice(
        ["Lunes", "Martes", "Miércoles", "Jueves",
         "Viernes", "Sábado", "Domingo"], n
    ),
    "hora": np.random.randint(5, 23, n),
    "clima": np.random.choice(
        ["Soleado", "Nublado", "Lluvioso"], n
    ),
    "retraso_minutos": np.random.randint(0, 31, n),
    "cantidad_personas": np.random.randint(5, 101, n),
    "cantidad_rutas": np.random.randint(1, 5, n)
})

# Evitamos que origen y destino sean iguales
iguales = datos["parada_origen"] == datos["parada_destino"]

while iguales.any():
    datos.loc[iguales, "parada_destino"] = np.random.choice(
        paradas, iguales.sum()
    )
    iguales = datos["parada_origen"] == datos["parada_destino"]

# Ajustamos los retrasos según el clima y la hora
datos["retraso_minutos"] = (
    np.where(datos["clima"] == "Lluvioso", 10, 0)
    + np.where(
        datos["hora"].between(7, 9)
        | datos["hora"].between(17, 19),
        8, 2
    )
    + np.random.randint(0, 6, n)
)

# Simulamos la cantidad de personas según la hora
datos["cantidad_personas"] = (
    np.where(
        datos["hora"].between(7, 9)
        | datos["hora"].between(17, 19),
        60, 25
    )
    + np.random.randint(0, 21, n)
)

# Estimamos la cantidad de rutas según la demanda
datos["cantidad_rutas"] = np.where(
    datos["cantidad_personas"] > 70, 4,
    np.where(datos["cantidad_personas"] > 40, 3, 2)
)

# Guardamos el dataset en un archivo CSV
datos.to_csv("dataset_transporte.csv", index=False)

# Verificamos los resultados
print("Dataset generado correctamente.")
print("Tamaño del dataset:", datos.shape)
print(
    "Rutas con origen y destino iguales:",
    (datos["parada_origen"] == datos["parada_destino"]).sum()
)
print("\nPrimeros cinco registros:")
print(datos.head())