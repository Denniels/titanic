"""
Módulo para feature engineering del dataset del Titanic.

Este módulo contiene la clase principal para realizar todas las
transformaciones de características.
"""

import pandas as pd

class TitanicFeatureEngineering:
    """
    Clase para realizar feature engineering en el dataset del Titanic.
    Incluye métodos para extraer y transformar características.
    """
    
    def __init__(self):
        """Inicializa los atributos necesarios para el feature engineering"""
        self.title_mapping = {}
        self.fare_bins = None
        self.age_bins = None
    
    def extract_title(self, name: str) -> str:
        """
        Extrae el título del nombre del pasajero.

        Args:
            name (str): Nombre completo del pasajero

        Returns:
            str: Título extraído y normalizado
        """
        title = name.split(',')[1].split('.')[0].strip()
        if title in ['Mr', 'Mrs', 'Miss', 'Master']:
            return title
        return 'Other'
    
    def create_family_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características relacionadas con el tamaño de la familia.

        Args:
            df (pd.DataFrame): DataFrame original

        Returns:
            pd.DataFrame: DataFrame con nuevas características
        """
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        return df
    
    def create_age_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea bins de edad para categorizar a los pasajeros.

        Args:
            df (pd.DataFrame): DataFrame original

        Returns:
            pd.DataFrame: DataFrame con edad categorizada
        """
        df['AgeBin'] = pd.qcut(df['Age'], q=4, labels=['Young', 'Adult', 'MiddleAge', 'Senior'])
        return df
    
    def create_fare_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea bins de tarifa para categorizar los precios.

        Args:
            df (pd.DataFrame): DataFrame original

        Returns:
            pd.DataFrame: DataFrame con tarifa categorizada
        """
        df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
        return df
    
    def extract_cabin_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae información de la cabina.

        Args:
            df (pd.DataFrame): DataFrame original

        Returns:
            pd.DataFrame: DataFrame con información de cabina procesada
        """
        df['CabinDeck'] = df['Cabin'].str[0]
        df['CabinDeck'] = df['CabinDeck'].fillna('U')
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas las transformaciones de feature engineering.

        Args:
            df (pd.DataFrame): DataFrame original

        Returns:
            pd.DataFrame: DataFrame con todas las nuevas características
        """
        df = df.copy()
        df['Title'] = df['Name'].apply(self.extract_title)
        df = self.create_family_size(df)
        df = self.create_age_bins(df)
        df = self.create_fare_bins(df)
        df = self.extract_cabin_info(df)
        return df
