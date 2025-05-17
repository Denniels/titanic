"""
Script de validación para el análisis del Titanic.
Prueba todas las funcionalidades antes de crear el notebook.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def test_environment():
    """Prueba el entorno y las dependencias."""
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✅ Dependencias básicas instaladas correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error: {e}")
        return False

def test_custom_modules():
    """Prueba los módulos personalizados."""
    try:
        from utils import configurar_visualizacion
        from data_loader import cargar_datos, preparar_datos
        from eda import (
            plot_supervivencia_general,
            plot_supervivencia_por_clase,
            plot_piramide_edad_genero,
            plot_familias,
            plot_tarifas_supervivencia
        )
        print("✅ Módulos personalizados cargados correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error en módulos personalizados: {e}")
        return False

def test_data_loading():
    """Prueba la carga y preparación de datos."""
    try:
        from data_loader import cargar_datos, preparar_datos
        df = cargar_datos()
        print(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None

def test_visualizations(df):
    """Prueba todas las visualizaciones."""
    try:
        from eda import (
            plot_supervivencia_general,
            plot_supervivencia_por_clase,
            plot_piramide_edad_genero,
            plot_familias,
            plot_tarifas_supervivencia
        )
        
        funciones = [
            plot_supervivencia_general,
            plot_supervivencia_por_clase,
            plot_piramide_edad_genero,
            plot_familias,
            plot_tarifas_supervivencia
        ]
        
        for func in funciones:
            fig = func(df)
            plt.close(fig)
        
        print("✅ Todas las visualizaciones funcionan correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en visualizaciones: {e}")
        return False

def main():
    """Función principal de pruebas."""
    print("\n🔍 Iniciando pruebas del análisis del Titanic...\n")
    
    # Configurar path
    src_path = Path(__file__).parent.absolute()
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
    
    # Ejecutar pruebas
    env_ok = test_environment()
    if not env_ok:
        print("\n⚠️ Error en el entorno. Por favor, verifica la instalación de dependencias.")
        return False
    
    modules_ok = test_custom_modules()
    if not modules_ok:
        print("\n⚠️ Error en los módulos. Verifica que todos los archivos estén presentes.")
        return False
    
    df = test_data_loading()
    if df is None:
        print("\n⚠️ Error cargando datos. Verifica que los archivos CSV estén presentes.")
        return False
    
    vis_ok = test_visualizations(df)
    if not vis_ok:
        print("\n⚠️ Error en las visualizaciones.")
        return False
    
    print("\n✨ ¡Todas las pruebas completadas con éxito!")
    return True

if __name__ == "__main__":
    main()
