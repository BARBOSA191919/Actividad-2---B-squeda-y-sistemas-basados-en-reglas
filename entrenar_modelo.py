import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# MÓDULO DE SANTIAGO - ENTRENAMIENTO Y EVALUACIÓN DEL MODELO SUPERVISADO
# ==============================================================================

def cargar_y_preparar_datos(ruta_csv="dataset_transporte.csv"):
    """
    Carga el dataset generado por Jairo y realiza la codificacion de variables
    categoricas para permitir el entrenamiento del modelo.
    """
    df = pd.read_csv(ruta_csv)
    
    # Convertimos variables categoricas a numericas (One-Hot Encoding)
    df_encoded = pd.get_dummies(
        df, 
        columns=["parada_origen", "parada_destino", "dia_semana", "clima"], 
        drop_first=True
    )
    
    X = df_encoded.drop(columns=["retraso_minutos"])
    y = df_encoded["retraso_minutos"]
    
    # Division de datos (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def entrenar_y_evaluar_modelo():
    # 1. Cargar datos preparados
    X_train, X_test, y_train, y_test = cargar_y_preparar_datos()
    
    # 2. Instanciar y entrenar el modelo de Regresion con RandomForest
    print("Entrenando el modelo RandomForestRegressor...")
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    # 3. Generar predicciones sobre el conjunto de prueba
    y_pred = modelo.predict(X_test)
    
    # 4. Calcular metricas de evaluacion
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 5. Mostrar resultados por pantalla
    print("\n==================================================")
    print("     RESULTADOS DEL MODELO - SANTIAGO BARBOSA    ")
    print("==================================================")
    print(f"MAE (Error Absoluto Medio)  : {mae:.2f} minutos")
    print(f"MSE (Error Cuadratico Medio): {mse:.2f}")
    print(f"R² Score (Coeficiente R²)   : {r2:.2f}")
    print("==================================================\n")
    
    # 6. Muestra comparativa para la exposicion en video
    comparacion = pd.DataFrame({
        "Retraso Real (min)": y_test.values,
        "Retraso Predicho (min)": np.round(y_pred, 1),
        "Diferencia (min)": np.round(np.abs(y_test.values - y_pred), 1)
    })
    
    print("Muestra de predicciones frente a datos reales:")
    print(comparacion.head(10).to_string(index=False))
    
    return modelo, mae, mse, r2

if __name__ == "__main__":
    entrenar_y_evaluar_modelo()