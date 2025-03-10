import pandas as pd
import pickle
import numpy as np
import os

def predecir_renta_2025(datos, ruta_modelo, ruta_columnas):
    """
    Predice el precio de renta utilizando un modelo preentrenado.
    
    Args:
        datos (pandas.DataFrame o dict): Datos para los que se desea predecir la renta
        ruta_modelo (str): Ruta al archivo del modelo guardado
        ruta_columnas (str): Ruta al archivo con los nombres de las columnas
    
    Returns:
        float o array: Predicción o predicciones del precio de renta
    """
    try:
        # Cargar modelo
        with open(ruta_modelo, 'rb') as f:
            modelo = pickle.load(f)
        
        # Cargar nombres de columnas
        with open(ruta_columnas, 'rb') as f:
            columnas = pickle.load(f)
        
        # Convertir diccionario a DataFrame si es necesario
        if isinstance(datos, dict):
            datos = pd.DataFrame([datos])
        
        # Verificar que todas las columnas estén presentes
        columnas_faltantes = [col for col in columnas if col not in datos.columns]
        if columnas_faltantes:
            print(f"Advertencia: Faltan las siguientes columnas en los datos: {columnas_faltantes}")
            # Crear columnas faltantes con valores 0
            for col in columnas_faltantes:
                datos[col] = 0
                
        # Seleccionar solo las columnas utilizadas en el entrenamiento y en el mismo orden
        X = datos[columnas]
        
        # Hacer predicción
        predicciones = modelo.predict(X)
        
        # Si solo hay una predicción, devolver valor escalar
        if len(predicciones) == 1:
            print(f"Precio de renta estimado: ${predicciones[0]:.2f}")
            return predicciones[0]
        else:
            print(f"Se generaron {len(predicciones)} predicciones de precio de renta")
            return predicciones
            
    except Exception as e:
        print(f"Error al realizar la predicción: {str(e)}")
        return None

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo para predicción (un solo registro)
    datos_ejemplo = {
        "BHK":2,
        "Size": 1000,
        "bedrooms": 2,
        "bathrooms": 1,
        "Posted On": "2022-05-18",
        "Floor": "Ground out of 2",
        "Area Type": "Super Area",
        "Area Locality": "Bandel",
        "City": "Unfurnished",
        "Furnishing Status": "Unfurnished",
        "Tenant Preferred": "Bachelors/Family",
        "Point of Contact": "Contact Owner"
    }
    
    # Convertir a DataFrame
    df_ejemplo = pd.DataFrame([datos_ejemplo])
    
    # Realizar predicción
    renta_estimada = predecir_renta(
        df_ejemplo,
        ruta_modelo="C:/Users/Dell/Documents/ucentral/deep learning/clase 3/clasificador_tumores/modelos/modelo_regresion.pkl",
        ruta_columnas="C:/Users/Dell/Documents/ucentral/deep learning/clase 3/clasificador_tumores/modelos/columnas_entrenamiento.pkl"
    )
    
    if renta_estimada is not None:
        print(f"El precio de renta estimado es: ${renta_estimada:.2f}")
