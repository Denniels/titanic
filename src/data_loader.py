#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for loading and preparing Titanic dataset.
Handles data loading and feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def cargar_datos(tipo='train'):
    """
    Carga y prepara los datos del Titanic.
    
    Args:
        tipo (str): 'train' o 'test' para cargar el conjunto correspondiente
        
    Returns:
        pandas.DataFrame: DataFrame con los datos cargados y preparados
    """
    # Definir rutas de archivos
    data_dir = Path(__file__).parent.parent / 'datasets'
    
    if tipo == 'train':
        file_path = data_dir / 'train.csv'
    else:
        file_path = data_dir / 'test.csv'
    
    # Cargar datos
    df = pd.read_csv(file_path)
    
    # Preparar datos si no es el conjunto de submission
    if tipo != 'submission':
        df = preparar_datos(df)
    
    return df

def preparar_datos(df):
    """
    Prepara los datos para el análisis.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con características adicionales
    """
    # Copiar DataFrame
    df = df.copy()
    
    # Manejar valores faltantes
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Crear características de familia
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Extraer título del nombre
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Agrupar títulos poco comunes
    titulo_map = {
        'Mr': 'Mr',
        'Mrs': 'Mrs',
        'Miss': 'Miss',
        'Master': 'Master',
        'Don': 'Rare',
        'Rev': 'Rare',
        'Dr': 'Rare',
        'Mme': 'Mrs',
        'Ms': 'Miss',
        'Major': 'Rare',
        'Lady': 'Rare',
        'Sir': 'Rare',
        'Mlle': 'Miss',
        'Col': 'Rare',
        'Capt': 'Rare',
        'Countess': 'Rare',
        'Jonkheer': 'Rare'
    }
    df['Title'] = df['Title'].map(titulo_map)
    
    # Crear rangos de edad
    df['AgeBin'] = pd.cut(df['Age'], 
                         bins=[0, 12, 20, 40, 60, np.inf],
                         labels=['Niño', 'Joven', 'Adult', 'MiddleAge', 'Senior'])
    
    # Crear rangos de tarifa
    df['FareBin'] = pd.qcut(df['Fare'], 
                           q=4,
                           labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # Extraer cubierta de la cabina
    df['CabinDeck'] = df['Cabin'].str[0]
    df['CabinDeck'] = df['CabinDeck'].fillna('U')
    
    return df
