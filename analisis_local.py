import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Cargar las variables del .env
load_dotenv()

##################################  CONFIGURACIÓN  ##################################
    
# Paths
DATA_DIR = os.getenv('DATA_DIR')
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PERIODOS_A_CARGAR = [2, 3]  # Períodos a cargar en análisis local

# Configuración de visualización
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# Periodos de análisis
PERIODOS = {
    1: {
        'nombre': 'Pre-pandemia',
        'inicio': datetime(2019, 1, 1),
        'fin': datetime(2020, 1, 31),
        'carpeta': 'periodo_1_PrePandemia'
    },
    2: {
        'nombre': 'Primeros casos',
        'inicio': datetime(2020, 2, 1),
        'fin': datetime(2020, 3, 15),
        'carpeta': 'periodo_2_PrimerosCasos'
    },
    3: {
        'nombre': 'Confinamiento estricto',
        'inicio': datetime(2020, 3, 16),
        'fin': datetime(2020, 5, 2),
        'carpeta': 'periodo_3_Confinamiento' 
    },
    4: {
        'nombre': 'Nueva normalidad',
        'inicio': datetime(2020, 5, 3),
        'fin': datetime(2020, 12, 31),
        'carpeta': 'periodo_4_NuevaNormalidad'
    }
}

##################################  CARGA Y LIMPIEZA  ##################################
    
def cargar_y_etiquetar_periodo(carpeta_path, num_periodo):
    """
    Carga todos los CSVs de una carpeta y añade etiqueta de período.
    
    Args:
        carpeta_path: Ruta a la carpeta del período
        num_periodo: Número del período (1, 2, 3, 4)
    
    Returns:
        DataFrame con datos + columna 'periodo'
    """
    dfs = []
    
    # Recorrer todos los CSVs en la carpeta
    csv_files = sorted(Path(carpeta_path).glob('*.csv'))
    
    if not csv_files:
        print(f"⚠ No se encontraron CSVs en {carpeta_path}")
        return None
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, sep=';') # Lee el csv como un df
        dfs.append(df) # Añade el df a la lista dfs
        print(f"  ✓ Cargado: {csv_file.name}")
    
    df_periodo = pd.concat(dfs, ignore_index=True) #Concatena todos los dfs de ese periodo
    
    # Añadir numero y nombre de período
    df_periodo['periodo_num'] = num_periodo
    df_periodo['periodo'] = PERIODOS[num_periodo]['nombre']
    
    return df_periodo


def cargar_datos_multiples_periodos(periodos_a_cargar=[1, 2]):
    """
    Carga y concatena datos de múltiples períodos.
    
    Args:
        periodos_a_cargar: Lista de números de período a cargar
    
    Returns:
        DataFrame concatenado con todos los períodos
    """
    dfs_periodos = []
    
    for num_periodo in periodos_a_cargar:
        info_periodo = PERIODOS[num_periodo] # Datos asociados al periodo (nombre, fechas, carpeta)
        carpeta = info_periodo['carpeta']
        
        carpeta_path = os.path.join(DATA_DIR, carpeta)
        
        print(f"\nCargando Período {num_periodo}: {info_periodo['nombre']}")
        print(f"Carpeta: {carpeta_path}")
        
        df = cargar_y_etiquetar_periodo(carpeta_path, num_periodo) # Carga y etiqueta el periodo
        
        if df is not None:
            dfs_periodos.append(df) # Añade el df del periodo a la lista
    
    # Concatenar todos los períodos
    df_final = pd.concat(dfs_periodos, ignore_index=True)
    
    print(f"\n✓ Total de filas cargadas: {len(df_final):,}")
    
    return df_final

def limpiar_datos(df):
    """
    Limpia y prepara datos para análisis.
    
    Args:
        df: DataFrame crudo
    
    Returns:
        DataFrame limpio
    """
    df = df.copy()
    
    # Convertir fecha a datetime
    df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
    
    # Extraer componentes de fecha
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['día'] = df['fecha'].dt.day
    df['hora'] = df['fecha'].dt.hour
    df['día_semana'] = df['fecha'].dt.dayofweek  # 0=Lunes, 6=Domingo
    df['es_fin_semana'] = df['día_semana'].isin([5, 6]) # True si es fin de semana
    
    return df

def reclasificar_periodo_por_fecha(df):
    """
    Reclasifica los registros al período correcto basándose en fecha exacta.
    Necesario para el periodo del confinamiento estricto.
    
    Args:
        df: DataFrame con columna 'fecha'
    
    Returns:
        DataFrame con período reclasificado
    """
    def asignar_periodo(fecha):
        if pd.isna(fecha): #Proteccion contra fechas nulas
            return None
        
        for num, info in PERIODOS.items():
            if info['inicio'] <= fecha <= info['fin']: # Verifica si la fecha cae dentro del período
                return num 
        return None
    
    df['periodo_num'] = df['fecha'].apply(asignar_periodo) # Asigna el período correcto
    df['periodo'] = df['periodo_num'].map({
        num: info['nombre'] for num, info in PERIODOS.items() # Mapea el nombre del período
    })
    
    return df


##################################  ANÁLISIS  ##################################

def analisis_intensidad_ocupacion_carga(df):
    """
    Análisis 1: Intensidad, ocupación y carga media por período.
    
    Args:
        df: DataFrame limpio
    
    Returns:
        DataFrame con estadísticas
    """
    analisis = df.groupby('periodo').agg({
        'intensidad': ['mean', 'std', 'max', 'min'],
        'ocupacion': ['mean', 'std', 'max'],
        'carga': ['mean', 'std', 'max']
    }).round(2)
    
    return analisis


def analisis_carga_por_hora(df):
    """
    Análisis 2: Carga por hora del día, por período.
    
    Args:
        df: DataFrame limpio
    
    Returns:
        DataFrame pivotado (filas=horas, columnas=períodos)
    """
    analisis = df.groupby(['periodo', 'hora'])['carga'].agg(['mean', 'std']).reset_index()
    
    # Pivotar para tener períodos como columnas
    #analisis_pivot = analisis.pivot(index='hora', columns='periodo', values='mean')
    
    return analisis


def analisis_laborables_vs_finde(df):
    """
    Análisis 3: Diferencia entre laborables y fin de semana.
    
    Args:
        df: DataFrame limpio
    
    Returns:
        DataFrame con comparativa
    """
    df['tipo_día'] = df['es_fin_semana'].map({True: 'Fin de semana', False: 'Laborable'})
    
    analisis = df.groupby(['periodo', 'tipo_día']).agg({
        'carga': ['mean', 'std'],
        'intensidad': ['mean', 'std'],
        'ocupacion': ['mean', 'std']
    }).round(2)
    
    return analisis


def analisis_tipo_via(df):
    """
    Análisis 4: Comparativa Urbano vs M30.
    
    Args:
        df: DataFrame limpio
    
    Returns:
        DataFrame con comparativa
    """
    analisis = df.groupby(['periodo', 'tipo_elem']).agg({
        'carga': ['mean', 'std'],
        'intensidad': ['mean', 'std'],
        'ocupacion': ['mean', 'std']
    }).round(2)
    
    return analisis

##################################  VISUALIZACIÓN  ##################################

def grafico_intensidad_ocupacion_carga(df):
    """
    Gráfico 1: Comparativa de intensidad, ocupación y carga por período.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Intensidad
    df.groupby('periodo')['intensidad'].mean().plot(ax=axes[0], kind='bar', color='steelblue')
    axes[0].set_title('Intensidad Media por Período', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Intensidad (veh/hora)')
    axes[0].set_xlabel('')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Ocupación
    df.groupby('periodo')['ocupacion'].mean().plot(ax=axes[1], kind='bar', color='coral')
    axes[1].set_title('Ocupación Media por Período', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Ocupación (%)')
    axes[1].set_xlabel('')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Carga
    df.groupby('periodo')['carga'].mean().plot(ax=axes[2], kind='bar', color='darkred')
    axes[2].set_title('Carga Media por Período', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Carga (0-100)')
    axes[2].set_xlabel('')
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_intensidad_ocupacion_carga.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: 01_intensidad_ocupacion_carga.png")
    plt.close()

def grafico_carga_por_hora(df):
    """
    Gráfico 2: Carga por hora, comparando períodos.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for periodo in df['periodo'].unique():
        datos_periodo = df[df['periodo'] == periodo].groupby('hora')['carga'].mean()
        ax.plot(datos_periodo.index, datos_periodo.values, marker='o', label=periodo, linewidth=2)
    
    ax.set_title('Carga del Viario por Hora del Día (por Período)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Hora del día')
    ax.set_ylabel('Carga (0-100)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_carga_por_hora.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: 02_carga_por_hora.png")
    plt.close()


def grafico_laborables_vs_finde(df):
    """
    Gráfico 3: Comparativa laborables vs fin de semana.
    """
    df['tipo_día'] = df['es_fin_semana'].map({True: 'Fin de semana', False: 'Laborable'})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Carga
    comparativa_carga = df.groupby(['periodo', 'tipo_día'])['carga'].mean().unstack()
    comparativa_carga.plot(ax=axes[0], kind='bar', color=['steelblue', 'coral'])
    axes[0].set_title('Carga: Laborables vs Fin de Semana', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Carga media (0-100)')
    axes[0].set_xlabel('')
    axes[0].legend(title='Tipo de día')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Intensidad
    comparativa_intensidad = df.groupby(['periodo', 'tipo_día'])['intensidad'].mean().unstack()
    comparativa_intensidad.plot(ax=axes[1], kind='bar', color=['steelblue', 'coral'])
    axes[1].set_title('Intensidad: Laborables vs Fin de Semana', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Intensidad media (veh/hora)')
    axes[1].set_xlabel('')
    axes[1].legend(title='Tipo de día')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_laborables_vs_finde.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: 03_laborables_vs_finde.png")
    plt.close()


def grafico_tipo_via(df):
    """
    Gráfico 4: Comparativa Urbano vs M30.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Carga
    comparativa_carga = df.groupby(['periodo', 'tipo_elem'])['carga'].mean().unstack()
    comparativa_carga.plot(ax=axes[0], kind='bar', color=['steelblue', 'darkred'])
    axes[0].set_title('Carga: Urbano vs M30', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Carga media (0-100)')
    axes[0].set_xlabel('')
    axes[0].legend(title='Tipo de vía')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Intensidad
    comparativa_intensidad = df.groupby(['periodo', 'tipo_elem'])['intensidad'].mean().unstack()
    comparativa_intensidad.plot(ax=axes[1], kind='bar', color=['steelblue', 'darkred'])
    axes[1].set_title('Intensidad: Urbano vs M30', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Intensidad media (veh/hora)')
    axes[1].set_xlabel('')
    axes[1].legend(title='Tipo de vía')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_tipo_via.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: 04_tipo_via.png")
    plt.close()

##################################  EJECUCIÓN  ##################################

if __name__ == '__main__':
    print("="*80)
print("ANÁLISIS COVID-19 EN TRÁFICO MADRID - PANDAS (LOCAL)")
print("="*80)

# 1. Cargar datos (solo períodos 1 y 2 en versión local)
print("\n[1/5] Cargando datos...")
df = cargar_datos_multiples_periodos(periodos_a_cargar=PERIODOS_A_CARGAR)

# 2. Limpiar datos
print("\n[2/5] Limpiando datos...")
df = limpiar_datos(df)

# 3. Reclasificar por fecha exacta
print("\n[3/5] Reclasificando períodos por fecha...")
df = reclasificar_periodo_por_fecha(df)

# 4. Ejecutar análisis
print("\n[4/5] Ejecutando análisis...")

print("\n  Análisis 1: Intensidad, ocupación y carga")
analisis_1 = analisis_intensidad_ocupacion_carga(df)
print(analisis_1)
analisis_1.to_csv(f'{OUTPUT_DIR}/analisis_01_intensidad_ocupacion_carga.csv')

print("\n  Análisis 2: Carga por hora")
analisis_2 = analisis_carga_por_hora(df)
print(analisis_2)
analisis_2.to_csv(f'{OUTPUT_DIR}/analisis_02_carga_por_hora.csv')

print("\n  Análisis 3: Laborables vs fin de semana")
analisis_3 = analisis_laborables_vs_finde(df)
print(analisis_3)
analisis_3.to_csv(f'{OUTPUT_DIR}/analisis_03_laborables_vs_finde.csv')

print("\n  Análisis 4: Tipo de vía")
analisis_4 = analisis_tipo_via(df)
print(analisis_4)
analisis_4.to_csv(f'{OUTPUT_DIR}/analisis_04_tipo_via.csv')

# 5. Generar gráficos
print("\n[5/5] Generando gráficos...")
grafico_intensidad_ocupacion_carga(df)
grafico_carga_por_hora(df)
grafico_laborables_vs_finde(df)
grafico_tipo_via(df)

print("\n" + "="*80)
print("✓ ANÁLISIS COMPLETADO")
print("="*80)
print(f"\nArchivos guardados en: {OUTPUT_DIR}/")
