"""
Módulo para modelado y evaluación del problema del Titanic.

Este módulo contiene la clase principal para entrenar y evaluar
múltiples modelos de machine learning.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

class TitanicModeling:
    """
    Clase para entrenar y evaluar múltiples modelos en el dataset del Titanic.
    Incluye optimización de hiperparámetros y validación cruzada.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Inicializa los modelos y sus grids de hiperparámetros.

        Args:
            random_state (int): Semilla para reproducibilidad
        """
        self.random_state = random_state
        self.models = {
            'logistic': LogisticRegression(random_state=random_state),
            'random_forest': RandomForestClassifier(random_state=random_state),
            'xgboost': XGBClassifier(random_state=random_state),
            'lightgbm': lgb.LGBMClassifier(random_state=random_state)
        }
        
        self.param_grids = {
            'logistic': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1]
            },
            'lightgbm': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1]
            }
        }
    
    def train_and_evaluate(self, X, y, cv: int = 5) -> dict:
        """
        Entrena y evalúa múltiples modelos usando validación cruzada y
        búsqueda de hiperparámetros.

        Args:
            X: Features de entrenamiento
            y: Variable objetivo
            cv (int): Número de folds para validación cruzada

        Returns:
            dict: Resultados de cada modelo con sus mejores parámetros
        """
        results = {}
        
        for name, model in self.models.items():
            # Búsqueda de hiperparámetros
            grid_search = GridSearchCV(
                model,
                self.param_grids[name],
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
                scoring='accuracy',
                n_jobs=-1
            )
            
            # Ajustar el modelo
            grid_search.fit(X, y)
            
            # Guardar resultados
            results[name] = {
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'model': grid_search.best_estimator_
            }
            
            print(f"\nResultados para {name}:")
            print(f"Mejor puntuación: {grid_search.best_score_:.4f}")
            print(f"Mejores parámetros: {grid_search.best_params_}")
        
        return results
    
    @staticmethod
    def plot_model_comparison(results: dict) -> None:
        """
        Visualiza la comparación de modelos.

        Args:
            results (dict): Diccionario con los resultados de cada modelo
        """
        models = list(results.keys())
        scores = [results[model]['best_score'] for model in models]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=models, y=scores)
        plt.title('Comparación de Modelos - Accuracy en Validación Cruzada')
        plt.ylim(0.7, 1.0)
        plt.xticks(rotation=45)
        plt.show()
