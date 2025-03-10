import pandas as pd

def cargar_datos1(ruta_archivo):
    """
    Carga datos desde un archivo CSV y retorna un DataFrame.
    
    Args:
        ruta_archivo (str): Ruta al archivo CSV de datos
    
    Returns:
        pandas.DataFrame: DataFrame con los datos cargados
    """
    try:
        # Cargar el archivo CSV
        df = pd.read_csv(ruta_archivo)
        print(f"Datos cargados exitosamente: {df.shape[0]} filas y {df.shape[1]} columnas")
        return df
    except Exception as e:
        print(f"Error al cargar los datos: {str(e)}")
        return None
    
