"""
Script para probar todas las importaciones y funcionalidades del proyecto Titanic.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def test_imports():
    """Prueba todas las importaciones necesarias."""
    try:
        # Importar módulos propios
        from utils import configurar_visualizacion
        from data_loader import cargar_datos, preparar_datos
        from eda import (
            analizar_valores_faltantes,
            plot_supervivencia_general,
            plot_supervivencia_por_clase,
            plot_piramide_edad_genero,
            plot_familias,
            plot_tarifas_supervivencia
        )
        print("✅ Todas las importaciones funcionan correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error de importación: {str(e)}")
        return False

def test_data_loader():
    """Prueba las funciones de carga de datos."""
    try:
        from data_loader import cargar_datos, preparar_datos
        df = cargar_datos()
        df_prep = preparar_datos(df)
        print("✅ Funciones de carga de datos funcionan correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en carga de datos: {str(e)}")
        return False

def test_visualization():
    """Prueba las funciones de visualización."""
    try:
        from eda import (
            plot_supervivencia_general,
            plot_supervivencia_por_clase,
            plot_piramide_edad_genero,
            plot_familias,
            plot_tarifas_supervivencia
        )
        from data_loader import cargar_datos
        
        df = cargar_datos()
        
        # Probar cada función de visualización
        plot_supervivencia_general(df)
        plt.close()
        
        plot_supervivencia_por_clase(df)
        plt.close()
        
        plot_piramide_edad_genero(df)
        plt.close()
        
        plot_familias(df)
        plt.close()
        
        plot_tarifas_supervivencia(df)
        plt.close()
        
        print("✅ Todas las funciones de visualización funcionan correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en visualizaciones: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n🔍 Iniciando pruebas de importación y funcionalidad...\n")
    
    # Asegurar que src está en el path
    src_path = Path(__file__).parent.absolute()
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
    
    # Ejecutar pruebas
    imports_ok = test_imports()
    data_ok = test_data_loader()
    vis_ok = test_visualization()
    
    # Resumen
    print("\n📋 Resumen de pruebas:")
    print(f"{'✅' if imports_ok else '❌'} Importaciones")
    print(f"{'✅' if data_ok else '❌'} Carga de datos")
    print(f"{'✅' if vis_ok else '❌'} Visualizaciones")
