import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def entrenar_modelo(df, ruta_guardar, variable_objetivo="Rent"):
    """
    Entrena un modelo de regresión lineal usando variables numéricas y guarda el modelo.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
        ruta_guardar (str): Ruta donde guardar el modelo y archivo de columnas
        variable_objetivo (str): Nombre de la columna objetivo (default: "Rent")
    
    Returns:
        bool: True si el entrenamiento fue exitoso, False en caso contrario
    """
    try:
        # Verificar que la variable objetivo exista en el dataframe
        if variable_objetivo not in df.columns:
            print(f"Error: La variable objetivo '{variable_objetivo}' no existe en el DataFrame")
            return False
        
        # Seleccionar solo variables numéricas
        X = df.select_dtypes(include=[np.number])
        
        # Eliminar la variable objetivo de las características si está presente
        if variable_objetivo in X.columns:
            X = X.drop(columns=[variable_objetivo])
        
        # Verificar que queden columnas numéricas
        if X.shape[1] == 0:
            print("Error: No hay variables numéricas en el DataFrame para el entrenamiento")
            return False
        
        # Obtener la variable objetivo
        y = df[variable_objetivo]
        
        # Guardar nombres de columnas para uso futuro
        nombres_columnas = X.columns.tolist()
        ruta_columnas = os.path.join(ruta_guardar, "columnas_entrenamiento.pkl")
        
        # Crear directorio si no existe
        os.makedirs(ruta_guardar, exist_ok=True)
        
        # Guardar nombres de columnas
        with open(ruta_columnas, 'wb') as f:
            pickle.dump(nombres_columnas, f)
        
        print(f"Columnas de entrenamiento guardadas en: {ruta_columnas}")
        print(f"Variables usadas en el entrenamiento: {nombres_columnas}")
        
        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Crear y entrenar el modelo de regresión lineal
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        
        # Evaluar el modelo
        y_pred = modelo.predict(X_test)
        
        # Calcular métricas
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Imprimir rendimiento
        print("\nRendimiento del modelo:")
        print(f"R² (Coeficiente de determinación): {r2:.4f}")
        print(f"Error cuadrático medio (MSE): {mse:.4f}")
        print(f"Raíz del error cuadrático medio (RMSE): {rmse:.4f}")
        print(f"Error absoluto medio (MAE): {mae:.4f}")
        
        # Guardar el modelo
        ruta_modelo = os.path.join(ruta_guardar, "modelo_regresion.pkl")
        with open(ruta_modelo, 'wb') as f:
            pickle.dump(modelo, f)
            
        print(f"\nModelo entrenado y guardado en: {ruta_modelo}")
        return True
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        return False
