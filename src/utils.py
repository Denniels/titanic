"""
Módulo de utilidades para el análisis del Titanic.
Contiene funciones auxiliares para la configuración y verificación del entorno.
"""

import pkg_resources
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

def configurar_visualizacion():
    """
    Configura el estilo y las opciones de visualización para el análisis.
    
    Establece:
    - Configuración básica de seaborn con estilo darkgrid
    - Paleta de colores HUSL
    - Opciones de visualización de pandas para mostrar todas las columnas
    """
    sns.set_theme(style='darkgrid')  # Usa el tema darkgrid de seaborn
    sns.set_palette('husl')  # Paleta de colores vibrante
    plt.rcParams['figure.figsize'] = [10, 6]  # Tamaño por defecto de las figuras
    plt.rcParams['axes.titlesize'] = 14  # Tamaño del título
    plt.rcParams['axes.labelsize'] = 12  # Tamaño de las etiquetas
    
    # Configuración de pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)

def verificar_dependencias():
    """
    Verifica que todas las dependencias requeridas estén instaladas y actualizadas.
    
    Comprueba:
    - Presencia de los paquetes requeridos
    - Versiones mínimas necesarias
    
    Returns:
        tuple: (bool, list, list) - (todo_ok, paquetes_faltantes, paquetes_desactualizados)
    """
    dependencias_requeridas = {
        'pandas': '1.3.0',
        'numpy': '1.20.0',
        'matplotlib': '3.4.0',
        'seaborn': '0.11.0',
        'scikit-learn': '0.24.0'
    }
    
    faltantes = []
    desactualizadas = []
    
    for paquete, version_min in dependencias_requeridas.items():
        try:
            version_actual = pkg_resources.get_distribution(paquete).version
            if pkg_resources.parse_version(version_actual) < pkg_resources.parse_version(version_min):
                desactualizadas.append(f"{paquete} (actual: {version_actual}, requerida: {version_min})")
        except pkg_resources.DistributionNotFound:
            faltantes.append(paquete)
    
    todo_ok = not (faltantes or desactualizadas)
    return todo_ok, faltantes, desactualizadas

def mostrar_estado_dependencias():
    """
    Muestra en formato amigable el estado de las dependencias del proyecto.
    
    Imprime:
    - Lista de paquetes faltantes
    - Lista de paquetes desactualizados
    - Instrucciones para instalar/actualizar si es necesario
    """
    todo_ok, faltantes, desactualizadas = verificar_dependencias()
    
    if not todo_ok:
        print("🔥 Dependencias faltantes o desactualizadas encontradas:")
        if faltantes:
            print("\nFaltantes:")
            for paquete in faltantes:
                print(f"- {paquete}")
        if desactualizadas:
            print("\nDesactualizadas:")
            for paquete in desactualizadas:
                print(f"- {paquete}")
        print("\nPuede instalar/actualizar las dependencias usando:")
        print("pip install -r requirements.txt")
    else:
        print("✅ Todas las dependencias están instaladas y actualizadas!")
