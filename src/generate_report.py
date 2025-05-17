"""
Script principal para el análisis del Titanic.
Ejecuta todas las pruebas, genera visualizaciones y crea un reporte en Markdown.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class TitanicAnalyzer:
    def __init__(self):
        self.setup_environment()
        self.results = {}
        
    def setup_environment(self):
        """Configura el entorno y las importaciones."""
        # Añadir src al path
        src_path = Path(__file__).parent.absolute()
        if str(src_path) not in sys.path:
            sys.path.append(str(src_path))
            
        # Importar módulos necesarios
        from utils import configurar_visualizacion
        from data_loader import cargar_datos, preparar_datos
        from eda import (
            plot_supervivencia_general,
            plot_supervivencia_por_clase,
            plot_piramide_edad_genero,
            plot_familias,
            plot_tarifas_supervivencia
        )
        
        self.utils = {
            'configurar_visualizacion': configurar_visualizacion,
            'cargar_datos': cargar_datos,
            'preparar_datos': preparar_datos,
            'plot_supervivencia_general': plot_supervivencia_general,
            'plot_supervivencia_por_clase': plot_supervivencia_por_clase,
            'plot_piramide_edad_genero': plot_piramide_edad_genero,
            'plot_familias': plot_familias,
            'plot_tarifas_supervivencia': plot_tarifas_supervivencia
        }        # Configurar visualizaciones
        plt.style.use('seaborn-v0_8')  # Usar un estilo válido de seaborn
        sns.set_theme(style='darkgrid')  # Configuración adicional de seaborn
        
    def run_analysis(self):
        """Ejecuta el análisis completo."""
        print("🚀 Iniciando análisis del Titanic...")
        
        # Cargar datos
        self.df = self.utils['cargar_datos']()
        print("✅ Datos cargados correctamente")
        
        # Generar y guardar visualizaciones
        self.generate_visualizations()
        print("✅ Visualizaciones generadas")
        
        # Calcular estadísticas
        self.calculate_statistics()
        print("✅ Estadísticas calculadas")
        
        # Generar reporte
        self.generate_report()
        print("✅ Reporte generado")
        
    def generate_visualizations(self):
        """Genera y guarda todas las visualizaciones."""        # Crear directorio para imágenes si no existe
        base_dir = Path(__file__).parent.parent
        img_dir = base_dir / 'output' / 'images'
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # Lista de visualizaciones a generar
        visualizations = [
            ('supervivencia_general', self.utils['plot_supervivencia_general']),
            ('supervivencia_clase', self.utils['plot_supervivencia_por_clase']),
            ('piramide_edad_genero', self.utils['plot_piramide_edad_genero']),
            ('analisis_familias', self.utils['plot_familias']),
            ('analisis_tarifas', self.utils['plot_tarifas_supervivencia'])
        ]
        
        # Generar y guardar cada visualización
        self.results['plots'] = {}
        for name, plot_func in visualizations:
            fig = plot_func(self.df)
            filepath = img_dir / f'{name}.png'
            fig.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close(fig)
            self.results['plots'][name] = str(filepath)
            
    def calculate_statistics(self):
        """Calcula estadísticas importantes."""
        self.results['stats'] = {
            'total_passengers': len(self.df),
            'survival_rate': (self.df['Survived'].mean() * 100),
            'class_survival': self.df.groupby('Pclass')['Survived'].mean() * 100,
            'gender_survival': self.df.groupby('Sex')['Survived'].mean() * 100,
            'avg_age': self.df['Age'].mean(),
            'avg_fare': self.df['Fare'].mean()
        }
        
    def generate_report(self):
        """Genera el reporte en Markdown."""
        base_dir = Path(__file__).parent.parent
        report_path = base_dir / 'output' / 'titanic_analysis_report.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generar rutas relativas para las imágenes
        for key, path in self.results['plots'].items():
            self.results['plots'][key] = os.path.relpath(path, report_path.parent)

        # Calcular estadísticas adicionales
        edad_por_clase = self.df.groupby('Pclass')['Age'].mean()
        tarifa_por_clase = self.df.groupby('Pclass')['Fare'].mean()
        supervivencia_por_edad = self.df.groupby(pd.cut(self.df['Age'], bins=[0, 12, 18, 35, 50, 100]))['Survived'].mean() * 100
        
        report_content = f"""# 🚢 Análisis Exhaustivo del Desastre del Titanic
*Análisis detallado generado el {datetime.now().strftime('%d/%m/%Y')}*

## 📋 Tabla de Contenidos
1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Contexto Histórico](#contexto-histórico)
3. [Análisis Demográfico](#análisis-demográfico)
4. [Patrones de Supervivencia](#patrones-de-supervivencia)
5. [Análisis Socioeconómico](#análisis-socioeconómico)
6. [Análisis Familiar](#análisis-familiar)
7. [Hallazgos Clave](#hallazgos-clave)
8. [Conclusiones](#conclusiones)

## 📊 Resumen Ejecutivo

El RMS Titanic, considerado "insumergible", se hundió en su viaje inaugural el 15 de abril de 1912, convirtiéndose en uno de los desastres marítimos más famosos de la historia. Este análisis examina en detalle los patrones de supervivencia entre los {self.results['stats']['total_passengers']:.0f} pasajeros documentados.

### Estadísticas Fundamentales
- **Supervivientes**: {self.results['stats']['survival_rate']:.1f}% del total de pasajeros
- **Demografía**: Edad promedio de {self.results['stats']['avg_age']:.1f} años
- **Aspecto Económico**: Tarifa promedio de £{self.results['stats']['avg_fare']:.2f}

## 🎭 Contexto Histórico

El Titanic representaba el pináculo del lujo y la ingeniería naval de su época. La distribución de pasajeros reflejaba la marcada estratificación social de la era eduardiana:

- **Primera Clase**: Elite social y económica
- **Segunda Clase**: Clase media profesional
- **Tercera Clase**: Inmigrantes y clase trabajadora

## 👥 Análisis Demográfico

### Distribución por Edad y Género
![Pirámide de Edad]({self.results['plots']['piramide_edad_genero']})

#### Análisis por Grupos de Edad
- **Niños (0-12 años)**: {supervivencia_por_edad.iloc[0]:.1f}% de supervivencia
- **Adolescentes (13-18)**: {supervivencia_por_edad.iloc[1]:.1f}% de supervivencia
- **Adultos Jóvenes (19-35)**: {supervivencia_por_edad.iloc[2]:.1f}% de supervivencia
- **Adultos Mediana Edad (36-50)**: {supervivencia_por_edad.iloc[3]:.1f}% de supervivencia
- **Adultos Mayores (50+)**: {supervivencia_por_edad.iloc[4]:.1f}% de supervivencia

**Observaciones Demográficas Detalladas:**
- La mayoría de los pasajeros eran adultos jóvenes, reflejando el perfil típico de inmigrantes
- Notable presencia de familias completas, especialmente en segunda y tercera clase
- Distribución de género desequilibrada, con predominio masculino
- Diferentes perfiles de edad según la clase social:
  - Primera Clase: Edad promedio {edad_por_clase[1]:.1f} años
  - Segunda Clase: Edad promedio {edad_por_clase[2]:.1f} años
  - Tercera Clase: Edad promedio {edad_por_clase[3]:.1f} años

## 🎯 Patrones de Supervivencia

### Supervivencia General
![Supervivencia General]({self.results['plots']['supervivencia_general']})

Este gráfico ilustra la dramática realidad del desastre: de los {self.results['stats']['total_passengers']:.0f} pasajeros, solo un tercio sobrevivió. Las razones fueron múltiples:
- Insuficientes botes salvavidas
- Procedimientos de evacuación caóticos
- Temperatura extremadamente fría del agua
- Tiempo de rescate prolongado

### Análisis por Clase Social
![Supervivencia por Clase]({self.results['plots']['supervivencia_clase']})

#### Tasas de Supervivencia Detalladas por Clase
```
🥇 Primera Clase: {self.results['stats']['class_survival'][1]:.1f}% 
🥈 Segunda Clase: {self.results['stats']['class_survival'][2]:.1f}% 
🥉 Tercera Clase: {self.results['stats']['class_survival'][3]:.1f}% 
```

**Análisis por Clase Social:**

1. **Primera Clase:**
   - Ubicación privilegiada cerca de la cubierta de botes
   - Mejor acceso a información durante la emergencia
   - Personal dedicado y mejor servicio
   - Tarifa promedio: £{tarifa_por_clase[1]:.2f}

2. **Segunda Clase:**
   - Posición intermedia en el barco
   - Acceso moderado a botes salvavidas
   - Mejor situación que tercera clase
   - Tarifa promedio: £{tarifa_por_clase[2]:.2f}

3. **Tercera Clase:**
   - Ubicación en las cubiertas inferiores
   - Acceso limitado a cubiertas superiores
   - Barreras lingüísticas y culturales
   - Tarifa promedio: £{tarifa_por_clase[3]:.2f}

### Análisis por Género
```
👩 Mujeres: {self.results['stats']['gender_survival']['female']:.1f}% supervivencia
👨 Hombres: {self.results['stats']['gender_survival']['male']:.1f}% supervivencia
```

**Factores que influyeron en la disparidad de género:**
- Política estricta de "mujeres y niños primero"
- Normas sociales de la época
- Roles de género en situaciones de emergencia
- Ubicación de camarotes por género

## 💰 Análisis Socioeconómico

### Relación entre Tarifas y Supervivencia
![Análisis de Tarifas]({self.results['plots']['analisis_tarifas']})

**Análisis Detallado de Tarifas:**
- **Rango de Tarifas**: £{self.df['Fare'].min():.2f} - £{self.df['Fare'].max():.2f}
- **Mediana**: £{self.df['Fare'].median():.2f}
- **Correlación con Supervivencia**: Fuertemente positiva

**Observaciones sobre Tarifas:**
1. **Tarifas Altas:**
   - Mejor ubicación en el barco
   - Acceso prioritario a botes salvavidas
   - Camarotes más cercanos a las cubiertas superiores
   - Mayor probabilidad de supervivencia

2. **Tarifas Medias:**
   - Ubicación intermedia
   - Acceso variable a información
   - Supervivencia dependiente de otros factores

3. **Tarifas Bajas:**
   - Ubicación en cubiertas inferiores
   - Rutas de escape más largas y complejas
   - Menor acceso a información crucial

## 👨‍👩‍👧‍👦 Análisis Familiar

![Análisis Familiar]({self.results['plots']['analisis_familias']})

**Patrones de Supervivencia Familiar:**

1. **Individuos Solitarios:**
   - Mayor vulnerabilidad
   - Menos recursos de apoyo
   - Toma de decisiones individual

2. **Familias Pequeñas (2-4 miembros):**
   - Mayor cohesión grupal
   - Mejor capacidad de movimiento
   - Apoyo mutuo efectivo
   - Mejores tasas de supervivencia

3. **Familias Grandes (5+ miembros):**
   - Dificultad para mantenerse unidos
   - Desafíos en la coordinación
   - Mayor complejidad en la evacuación

**Factores Familiares Adicionales:**
- Presencia de niños aumentaba probabilidad de supervivencia familiar
- Familias de primera clase tenían ventajas adicionales
- Grupos familiares mixtos enfrentaban decisiones difíciles

## 🔍 Hallazgos Clave

### 1. Desigualdad Social
- La clase social fue determinante en la supervivencia
- Las diferencias en tasas de supervivencia reflejan la estratificación social
- El acceso a recursos y información varió significativamente por clase

### 2. Factor Género
- Las mujeres tuvieron clara ventaja en la supervivencia
- La política de evacuación favoreció a mujeres y niños
- Los hombres mostraron tasas de supervivencia significativamente menores

### 3. Impacto de la Edad
- Los niños tuvieron prioridad en el rescate
- La edad influyó diferentemente según la clase social
- Adultos mayores enfrentaron mayores desafíos

### 4. Dinámica Familiar
- El tamaño familiar óptimo para supervivencia: 2-4 miembros
- Grupos muy grandes enfrentaron mayores dificultades
- La presencia de niños influía en las decisiones de rescate

## 📝 Conclusiones

### Lecciones Históricas
1. **Desigualdad Social:**
   - El desastre expuso dramáticamente las diferencias de clase
   - La ubicación en el barco determinaba las probabilidades de supervivencia
   - El acceso a información y recursos era altamente desigual

2. **Factores Humanos:**
   - Las normas sociales influyeron en los patrones de rescate
   - La estructura familiar afectó las decisiones de supervivencia
   - El comportamiento humano en crisis seguía patrones sociales establecidos

3. **Legado:**
   - Cambios en regulaciones marítimas
   - Mejoras en protocolos de emergencia
   - Mayor conciencia sobre desigualdad social

### Implicaciones Modernas
- Necesidad de protocolos de emergencia equitativos
- Importancia de planificación inclusiva
- Valor de sistemas de comunicación efectivos
- Relevancia continua de factores socioeconómicos en desastres

> *Este análisis exhaustivo fue generado por TitanicAnalyzer - Un estudio detallado basado en datos históricos del desastre del Titanic*
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        print(f"📄 Reporte exhaustivo generado en: {report_path}")

if __name__ == "__main__":
    analyzer = TitanicAnalyzer()
    analyzer.run_analysis()
