"""
Módulo para preprocesamiento de datos del Titanic.

Este módulo contiene la clase principal para realizar el
preprocesamiento de datos antes del modelado.
"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class TitanicPreprocessor:
    """
    Clase para preprocesar los datos del Titanic.
    Implementa un pipeline completo de preprocesamiento.
    """
    
    def __init__(self):
        """
        Inicializa las listas de características numéricas y categóricas
        """
        self.numerical_features = ['Age', 'Fare', 'FamilySize']
        self.categorical_features = ['Sex', 'Embarked', 'Title', 'CabinDeck', 'AgeBin', 'FareBin']
        
    def create_pipeline(self) -> ColumnTransformer:
        """
        Crea un pipeline de preprocesamiento que combina:
        - Imputación de valores faltantes
        - Escalado de variables numéricas
        - Codificación one-hot de variables categóricas

        Returns:
            ColumnTransformer: Pipeline completo de preprocesamiento
        """
        numerical_pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_features),
                ('cat', categorical_pipeline, self.categorical_features)
            ])
        
        return preprocessor
