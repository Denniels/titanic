"""
Módulo para el análisis exploratorio de datos del Titanic.
Contiene funciones para generar visualizaciones y estadísticas descriptivas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

def analizar_valores_faltantes(df):
    """
    Analiza y visualiza los valores faltantes en el dataset.
    
    Args:
        df (pandas.DataFrame): DataFrame a analizar
    
    Returns:
        pandas.DataFrame: Resumen de valores faltantes por columna
    """
    valores_faltantes = df.isnull().sum()
    porcentajes = valores_faltantes / len(df) * 100
    
    resumen = pd.DataFrame({
        'Valores Faltantes': valores_faltantes,
        'Porcentaje': porcentajes
    })
    
    return resumen[resumen['Valores Faltantes'] > 0].sort_values('Valores Faltantes', ascending=False)

def plot_supervivencia_general(df):
    """
    Crea un gráfico de la distribución general de supervivencia con porcentajes.
    """
    plt.figure(figsize=(10, 6))
    
    # Preparar datos
    df_plot = pd.DataFrame({
        'Estado': ['No Sobrevivió', 'Sobrevivió'],
        'Porcentaje': (df['Survived'].value_counts(normalize=True) * 100).round(1).values
    })
    
    # Crear gráfico
    ax = sns.barplot(data=df_plot, x='Estado', y='Porcentaje', 
                    hue='Estado', palette=['#ff6b6b', '#4ecdc4'],
                    legend=False)
    
    # Añadir porcentajes sobre las barras
    for i, v in enumerate(df_plot['Porcentaje']):
        ax.text(i, v + 1, f'{v}%', ha='center', fontsize=12)
    
    plt.title('Distribución de Supervivencia en el Titanic', pad=20, fontsize=14)
    plt.ylabel('Porcentaje de Pasajeros (%)')
    plt.ylim(0, 100)
    
    return plt.gcf()

def plot_supervivencia_por_clase(df):
    """
    Crea un gráfico de supervivencia por clase con información detallada.
    """
    plt.figure(figsize=(12, 6))
    
    # Calcular porcentajes por clase
    clase_surv = df.groupby('Pclass')['Survived'].agg(['count', 'mean']).round(3)
    clase_surv['mean'] = clase_surv['mean'] * 100
    
    # Preparar datos para el gráfico
    df_plot = df.copy()
    df_plot['Clase'] = df_plot['Pclass'].map({1: 'Primera', 2: 'Segunda', 3: 'Tercera'})
    
    # Crear gráfico
    ax = sns.barplot(data=df_plot, x='Clase', y='Survived', 
                    hue='Clase', palette=['#3498db', '#2ecc71', '#e74c3c'],
                    legend=False)
    
    # Añadir porcentajes y cantidades
    for i, row in enumerate(clase_surv.itertuples()):
        total = row.count
        porcentaje = row.mean
        ax.text(i, porcentaje/100 + 0.02, 
                f'{porcentaje:.1f}%\n({int(total*porcentaje/100)}/{total})', 
                ha='center', fontsize=10)
    
    plt.title('Tasa de Supervivencia por Clase', pad=20, fontsize=14)
    plt.xlabel('Clase del Pasajero')
    plt.ylabel('Tasa de Supervivencia')
    plt.xticks([0, 1, 2], ['Primera Clase', 'Segunda Clase', 'Tercera Clase'])
    
    return plt.gcf()

def plot_piramide_edad_genero(df):
    """
    Crea una pirámide de edad por género y supervivencia.
    """
    plt.figure(figsize=(12, 8))
    
    # Preparar datos
    male_survived = df[(df['Sex'] == 'male') & (df['Survived'] == 1)]['Age']
    male_died = df[(df['Sex'] == 'male') & (df['Survived'] == 0)]['Age']
    female_survived = df[(df['Sex'] == 'female') & (df['Survived'] == 1)]['Age']
    female_died = df[(df['Sex'] == 'female') & (df['Survived'] == 0)]['Age']
    
    bins = range(0, 81, 10)
    
    # Crear subplots
    plt.subplot(121)
    plt.hist(male_died, bins=bins, orientation='horizontal', alpha=0.8, 
            color='#ff6b6b', label='Fallecidos')
    plt.hist(male_survived, bins=bins, orientation='horizontal', alpha=0.8,
            color='#4ecdc4', label='Sobrevivientes')
    plt.title('Hombres', pad=20)
    plt.ylabel('Edad')
    plt.xlabel('Cantidad')
    plt.legend()
    
    plt.subplot(122)
    plt.hist(female_died, bins=bins, orientation='horizontal', alpha=0.8,
            color='#ff6b6b', label='Fallecidas')
    plt.hist(female_survived, bins=bins, orientation='horizontal', alpha=0.8,
            color='#4ecdc4', label='Sobrevivientes')
    plt.title('Mujeres', pad=20)
    plt.ylabel('Edad')
    plt.xlabel('Cantidad')
    plt.legend()
    
    plt.suptitle('Distribución de Edad y Supervivencia por Género', 
                 y=1.02, fontsize=14)
    plt.tight_layout()
    
    return plt.gcf()

def plot_familias(df):
    """
    Crea visualizaciones sobre el impacto del tamaño familiar en la supervivencia.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
      # Tamaño familiar vs Supervivencia
    familia_surv = df.groupby('FamilySize')['Survived'].agg(['count', 'mean'])
    sns.barplot(data=df, x='FamilySize', y='Survived', ax=ax1,
                hue='FamilySize', palette='viridis', legend=False)
    ax1.set_title('Tasa de Supervivencia por Tamaño Familiar', pad=20)
    ax1.set_xlabel('Tamaño de la Familia')
    ax1.set_ylabel('Tasa de Supervivencia')
    
    # Añadir etiquetas con cantidades
    for i, row in enumerate(familia_surv.itertuples()):
        if row.count > 0:  # Solo mostrar si hay pasajeros
            ax1.text(i, row.mean + 0.02, 
                    f'n={row.count}\n{row.mean*100:.1f}%', 
                    ha='center', fontsize=9)
      # Distribución de tamaños familiares por clase
    df_plot = df.copy()
    df_plot['Clase'] = df_plot['Pclass'].map({1: 'Primera', 2: 'Segunda', 3: 'Tercera'})
    sns.boxplot(data=df_plot, x='Clase', y='FamilySize', ax=ax2,
                hue='Clase', palette=['#3498db', '#2ecc71', '#e74c3c'], legend=False)
    ax2.set_title('Distribución de Tamaño Familiar por Clase', pad=20)
    ax2.set_xlabel('Clase')
    ax2.set_ylabel('Tamaño de la Familia')
    
    plt.tight_layout()
    return fig

def plot_tarifas_supervivencia(df):
    """
    Analiza la relación entre tarifas y supervivencia.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
      # Distribución de tarifas por supervivencia
    df_plot = df.copy()
    df_plot['Estado'] = df_plot['Survived'].map({0: 'Fallecidos', 1: 'Sobrevivientes'})
    sns.boxplot(data=df_plot, x='Estado', y='Fare', ax=ax1,
                hue='Estado', palette=['#ff6b6b', '#4ecdc4'], legend=False)
    ax1.set_title('Distribución de Tarifas por Supervivencia', pad=20)
    ax1.set_xlabel('Estado')
    ax1.set_ylabel('Tarifa (£)')
      # Tarifas por clase
    df_plot = df.copy()
    df_plot['Clase'] = df_plot['Pclass'].map({1: 'Primera', 2: 'Segunda', 3: 'Tercera'})
    sns.boxplot(data=df_plot, x='Clase', y='Fare', ax=ax2,
                hue='Clase', palette=['#3498db', '#2ecc71', '#e74c3c'], legend=False)
    ax2.set_title('Distribución de Tarifas por Clase', pad=20)
    ax2.set_xlabel('Clase')
    ax2.set_ylabel('Tarifa (£)')
    
    plt.tight_layout()
    return fig

def plot_correlaciones(df):
    """
    Genera un mapa de calor con las correlaciones entre variables numéricas.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
    """
    # Seleccionar solo variables numéricas
    numericas = df.select_dtypes(include=['int64', 'float64'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(numericas.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlaciones entre Variables Numéricas')
    plt.tight_layout()
    plt.show()
