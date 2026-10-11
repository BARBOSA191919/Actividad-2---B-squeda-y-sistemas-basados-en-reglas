from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


CSV_PATH = Path(__file__).resolve().parent / "dataset_transporte.csv"
TARGET = "retraso_minutos"
FEATURES_REQUERIDAS = ["cantidad_personas", "cantidad_rutas"]


def cargar_y_preparar_datos(ruta_csv=CSV_PATH):
    """Carga el CSV, codifica predictores categóricos y separa X e y."""
    df = pd.read_csv(ruta_csv)

    columnas_requeridas = [TARGET, *FEATURES_REQUERIDAS]
    faltantes = [col for col in columnas_requeridas if col not in df.columns]
    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas en {ruta_csv}: {', '.join(faltantes)}"
        )

    filas_iniciales = len(df)
    df = df.dropna().copy()
    if len(df) < 2:
        raise ValueError("El CSV debe tener al menos dos filas completas.")

    for column in [TARGET, *FEATURES_REQUERIDAS]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    y = df[TARGET]
    X = df.drop(columns=TARGET)

    if X.empty:
        raise ValueError("No hay variables predictoras disponibles en el CSV.")

    if len(df) != filas_iniciales:
        print(f"Se descartaron {filas_iniciales - len(df)} filas incompletas.")

    return X, y


def entrenar_y_evaluar_modelo(ruta_csv=CSV_PATH):
    X, y = cargar_y_preparar_datos(ruta_csv)

    # Se divide antes de ajustar el preprocesamiento para evitar usar información
    # del conjunto de prueba durante el entrenamiento.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    categorical_features = X.select_dtypes(include=["object", "category"]).columns
    numeric_features = X.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )
    modelo = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "regressor",
                RandomForestRegressor(n_estimators=100, random_state=42),
            ),
        ]
    )
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Registros usados: {len(X)}")
    print(f"Entrenamiento: {len(X_train)} ({len(X_train) / len(X):.0%})")
    print(f"Prueba: {len(X_test)} ({len(X_test) / len(X):.0%})")
    print(f"Variables originales usadas: {', '.join(X.columns)}")
    print("\nEvaluación en los datos de prueba:")
    print(f"MAE  (error absoluto medio): {mae:.2f} minutos")
    print(f"MSE  (error cuadrático medio): {mse:.2f}")
    print(f"RMSE (raíz del error cuadrático medio): {rmse:.2f} minutos")
    print(f"R²   (coeficiente de determinación): {r2:.3f}")

    comparacion = pd.DataFrame(
        {
            "retraso_real_minutos": y_test.to_numpy(),
            "retraso_predicho_minutos": np.round(y_pred, 1),
        },
        index=y_test.index,
    )
    print("\nComparación de predicciones (primeras 10 filas):")
    print(comparacion.head(10).to_string())

    feature_names = modelo.named_steps["preprocessing"].get_feature_names_out()
    importance_values = modelo.named_steps["regressor"].feature_importances_
    importancia = pd.Series(importance_values, index=feature_names)
    print("\nVariables más importantes:")
    print(importancia.sort_values(ascending=False).head(10).round(3).to_string())

    return modelo, (X_train, X_test, y_train, y_test), {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


if __name__ == "__main__":
    entrenar_y_evaluar_modelo()
