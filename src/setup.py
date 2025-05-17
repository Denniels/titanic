"""
Script de inicialización para el proyecto Titanic.
Configura el entorno y valida todas las dependencias necesarias.
"""

import sys
import importlib
from pathlib import Path
import pkg_resources

def verificar_dependencias():
    """Verifica que todas las dependencias necesarias estén instaladas."""
    dependencias_requeridas = {
        'pandas': '1.5.0',
        'numpy': '1.20.0',
        'matplotlib': '3.5.0',
        'seaborn': '0.11.0'
    }
    
    faltantes = []
    for package, version in dependencias_requeridas.items():
        try:
            pkg_resources.require(f"{package}>={version}")
        except pkg_resources.VersionConflict:
            print(f"⚠️ {package} está instalado pero necesita actualización")
            faltantes.append(package)
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package} no está instalado")
            faltantes.append(package)
    
    return len(faltantes) == 0

def configurar_path():
    """Configura el path para incluir el directorio src."""
    src_path = Path(__file__).parent.absolute()
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
        print(f"✅ Directorio src añadido al path: {src_path}")
    return True

def verificar_modulos():
    """Verifica que todos los módulos necesarios estén presentes y sean importables."""
    modulos = ['utils', 'data_loader', 'eda', 'feature_engineering', 'modeling', 'preprocessor']
    
    for modulo in modulos:
        try:
            importlib.import_module(modulo)
            print(f"✅ Módulo {modulo} importado correctamente")
        except ImportError as e:
            print(f"❌ Error importando {modulo}: {str(e)}")
            return False
    return True

def verificar_datos():
    """Verifica que los archivos de datos necesarios estén presentes."""
    archivos_requeridos = [
        Path('../datasets/train.csv'),
        Path('../datasets/test.csv'),
        Path('../datasets/gender_submission.csv')
    ]
    
    for archivo in archivos_requeridos:
        if not archivo.exists():
            print(f"❌ No se encuentra el archivo {archivo}")
            return False
    
    print("✅ Todos los archivos de datos están presentes")
    return True

def configurar_entorno():
    """Función principal que configura todo el entorno."""
    print("\n🚀 Iniciando configuración del entorno...\n")
    
    # 1. Verificar dependencias
    print("📦 Verificando dependencias...")
    deps_ok = verificar_dependencias()
    
    # 2. Configurar path
    print("\n🔧 Configurando path...")
    path_ok = configurar_path()
    
    # 3. Verificar módulos
    print("\n📚 Verificando módulos...")
    modulos_ok = verificar_modulos()
    
    # 4. Verificar datos
    print("\n📂 Verificando archivos de datos...")
    datos_ok = verificar_datos()
    
    # Resumen
    print("\n📋 Resumen de configuración:")
    print(f"{'✅' if deps_ok else '❌'} Dependencias")
    print(f"{'✅' if path_ok else '❌'} Path")
    print(f"{'✅' if modulos_ok else '❌'} Módulos")
    print(f"{'✅' if datos_ok else '❌'} Datos")
    
    return all([deps_ok, path_ok, modulos_ok, datos_ok])

if __name__ == "__main__":
    if configurar_entorno():
        print("\n✨ Entorno configurado correctamente")
    else:
        print("\n⚠️ Hubo problemas en la configuración del entorno")
        sys.exit(1)
