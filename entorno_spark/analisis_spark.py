import os
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, year, month, dayofweek, hour, 
    avg, max as spark_max, min as spark_min, stddev, count, when, 
    to_timestamp, lit, unix_timestamp
)
import pandas as pd
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
SPARK_MASTER = os.getenv('SPARK_MASTER', 'spark://10.0.1.14:7077')

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def encontrar_csvs(data_dir):
    """
    Encuentra todos los CSVs en la estructura de carpetas de períodos.
    
    Returns:
        Lista de rutas a todos los archivos CSV
    """
    csvs = list(Path(data_dir).glob('periodo_*/*.csv')) # Buscar todos los CSVs en subcarpetas
    print(f"✓ Encontrados {len(csvs)} archivos CSV")
    return [str(csv) for csv in sorted(csvs)]


def cargar_datos_spark(spark, csvs_paths):
    """
    Carga todos los CSVs en un DataFrame de Spark.
    
    Args:
        spark: SparkSession
        csvs_paths: Lista de rutas a CSVs
    
    Returns:
        DataFrame de Spark con todos los datos
    """
    df = spark.read.csv(
        csvs_paths,
        header=True,
        inferSchema=True,
        sep=';'
    )
    
    print(f"✓ Datos cargados. Total de filas: {df.count():,}")
    
    return df


def limpiar_datos_spark(df):
    """
    Limpia y prepara datos en Spark.
    
    Args:
        df: DataFrame de Spark crudo
    
    Returns:
        DataFrame de Spark limpio
    """
    
    # Convertir fecha a timestamp
    df = df.withColumn(
        'fecha_ts',
        to_timestamp('fecha', 'yyyy-MM-dd HH:mm:ss')
    )
    
    # Extraer componentes de fecha
    df = df \
        .withColumn('año', year('fecha_ts')) \
        .withColumn('mes', month('fecha_ts')) \
        .withColumn('hora', hour('fecha_ts')) \
        .withColumn('día_semana', dayofweek('fecha_ts')) \
        .withColumn('es_fin_semana', when((col('día_semana') == 6) | (col('día_semana') == 7), True).otherwise(False))
    
    return df


def reclasificar_periodo_por_fecha_spark(df):
    """
    Reclasifica registros al período correcto basándose en fecha exacta.
    Necesario para el período del confinamiento estricto.
    
    Args:
        df: DataFrame de Spark con columna 'fecha_ts'
    
    Returns:
        DataFrame con columnas 'periodo_num' y 'periodo'
    """
    
    # Asignar período basándose en fecha exacta
    df = df.withColumn('periodo_num', 
        when((col('fecha_ts') >= lit('2019-01-01').cast('timestamp')) & 
             (col('fecha_ts') <= lit('2020-01-31').cast('timestamp')), 1)
        .when((col('fecha_ts') >= lit('2020-02-01').cast('timestamp')) & 
              (col('fecha_ts') <= lit('2020-03-15').cast('timestamp')), 2)
        .when((col('fecha_ts') >= lit('2020-03-16').cast('timestamp')) & 
              (col('fecha_ts') <= lit('2020-05-02').cast('timestamp')), 3)
        .when((col('fecha_ts') >= lit('2020-05-03').cast('timestamp')) & 
              (col('fecha_ts') <= lit('2020-12-31').cast('timestamp')), 4)
        .otherwise(None)
    )
    
    # Mapear número a nombre de período
    df = df.withColumn('periodo',
        when(col('periodo_num') == 1, 'Pre-pandemia')
        .when(col('periodo_num') == 2, 'Primeros casos')
        .when(col('periodo_num') == 3, 'Confinamiento estricto')
        .when(col('periodo_num') == 4, 'Nueva normalidad')
        .otherwise(None)
    )
    
    # Filtrar registros sin período asignado
    df = df.filter(col('periodo').isNotNull())
    
    return df

##################################  ANÁLISIS  ##################################

def analisis_intensidad_ocupacion_carga_spark(df):
    """
    Análisis 1: Intensidad, ocupación y carga media por período.
    
    Args:
        df: DataFrame de Spark limpio
    
    Returns:
        DataFrame de pandas con estadísticas
    """
    analisis = df.groupby('periodo').agg(
        avg('intensidad').alias('intensidad_mean'),
        stddev('intensidad').alias('intensidad_std'),
        spark_max('intensidad').alias('intensidad_max'),
        spark_min('intensidad').alias('intensidad_min'),
        avg('ocupacion').alias('ocupacion_mean'),
        stddev('ocupacion').alias('ocupacion_std'),
        spark_max('ocupacion').alias('ocupacion_max'),
        avg('carga').alias('carga_mean'),
        stddev('carga').alias('carga_std'),
        spark_max('carga').alias('carga_max'),
    ).orderBy('periodo')
    
    return analisis.toPandas()


def analisis_carga_por_hora_spark(df):
    """
    Análisis 2: Carga por hora del día, por período.
    
    Args:
        df: DataFrame de Spark limpio
    
    Returns:
        DataFrame de pandas pivotado
    """
    analisis = df.groupby('periodo', 'hora').agg(
        avg('carga').alias('mean'),
        stddev('carga').alias('std')
    ).orderBy('periodo', 'hora')
    
    analisis_pd = analisis.toPandas()
    
    return analisis_pd


def analisis_laborables_vs_finde_spark(df):
    """
    Análisis 3: Diferencia entre laborables y fin de semana.
    
    Args:
        df: DataFrame de Spark limpio
    
    Returns:
        DataFrame de pandas con comparativa
    """
    df = df.withColumn('tipo_día',
        when(col('es_fin_semana') == True, 'Fin de semana')
        .otherwise('Laborable')
    )
    
    analisis = df.groupby('periodo', 'tipo_día').agg(
        avg('carga').alias('carga_mean'),
        stddev('carga').alias('carga_std'),
        avg('intensidad').alias('intensidad_mean'),
        stddev('intensidad').alias('intensidad_std'),
        avg('ocupacion').alias('ocupacion_mean'),
        stddev('ocupacion').alias('ocupacion_std')
    ).orderBy('periodo', 'tipo_día')
    
    return analisis.toPandas()


def analisis_tipo_via_spark(df):
    """
    Análisis 4: Comparativa Urbano vs M30.
    
    Args:
        df: DataFrame de Spark limpio
    
    Returns:
        DataFrame de pandas con comparativa
    """
    analisis = df.groupby('periodo', 'tipo_elem').agg(
        avg('carga').alias('carga_mean'),
        stddev('carga').alias('carga_std'),
        avg('intensidad').alias('intensidad_mean'),
        stddev('intensidad').alias('intensidad_std'),
        avg('ocupacion').alias('ocupacion_mean'),
        stddev('ocupacion').alias('ocupacion_std')
    ).orderBy('periodo', 'tipo_elem')
    
    return analisis.toPandas()

##################################  VISUALIZACIÓN  ##################################

def grafico_intensidad_ocupacion_carga(df):
    """
    Gráfico 1: Comparativa de intensidad, ocupación y carga por período.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Intensidad
    df.set_index('periodo')['intensidad_mean'].plot(ax=axes[0], kind='bar', color='steelblue')
    axes[0].set_title('Intensidad Media por Período', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Intensidad (veh/hora)')
    axes[0].set_xlabel('')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Ocupación
    df.set_index('periodo')['ocupacion_mean'].plot(ax=axes[1], kind='bar', color='coral')
    axes[1].set_title('Ocupación Media por Período', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Ocupación (%)')
    axes[1].set_xlabel('')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Carga
    df.set_index('periodo')['carga_mean'].plot(ax=axes[2], kind='bar', color='darkred')
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
        datos_periodo = df[df['periodo'] == periodo].sort_values('hora')
        ax.plot(datos_periodo['hora'], datos_periodo['mean'], marker='o', label=periodo, linewidth=2)
    
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pivotar para tener períodos como índice y tipo_día como columnas
    carga_pivot = df.pivot(index='periodo', columns='tipo_día', values='carga_mean')
    intensidad_pivot = df.pivot(index='periodo', columns='tipo_día', values='intensidad_mean')
    
    # Carga
    carga_pivot.plot(ax=axes[0], kind='bar', color=['steelblue', 'coral'])
    axes[0].set_title('Carga: Laborables vs Fin de Semana', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Carga media (0-100)')
    axes[0].set_xlabel('')
    axes[0].legend(title='Tipo de día')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Intensidad
    intensidad_pivot.plot(ax=axes[1], kind='bar', color=['steelblue', 'coral'])
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
    
    # Pivotar
    carga_pivot = df.pivot(index='periodo', columns='tipo_elem', values='carga_mean')
    intensidad_pivot = df.pivot(index='periodo', columns='tipo_elem', values='intensidad_mean')
    
    # Carga
    carga_pivot.plot(ax=axes[0], kind='bar', color=['steelblue', 'darkred'])
    axes[0].set_title('Carga: Urbano vs M30', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Carga media (0-100)')
    axes[0].set_xlabel('')
    axes[0].legend(title='Tipo de vía')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Intensidad
    intensidad_pivot.plot(ax=axes[1], kind='bar', color=['steelblue', 'darkred'])
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
    print("ANÁLISIS COVID-19 EN TRÁFICO MADRID - SPARK (DISTRIBUIDO)")
    print("="*80)
    
    # 1. Inicializar Spark
    print("\n[1/6] Inicializando Spark...")
    spark = SparkSession.builder \
        .appName("COVID19_Trafico_Madrid") \
        .master(SPARK_MASTER) \
        .config("spark.sql.shuffle.partitions", "100") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    print(f"✓ Spark inicializado en: {SPARK_MASTER}")
    
    # 2. Encontrar y cargar CSVs
    print("\n[2/6] Encontrando archivos CSV...")
    csvs = encontrar_csvs(DATA_DIR)
    
    print("\n[3/6] Cargando datos en Spark...")
    df_spark = cargar_datos_spark(spark, csvs)
    
    # 4. Limpiar datos
    print("\n[4/6] Limpiando datos...")
    df_spark = limpiar_datos_spark(df_spark)
    
    # 5. Reclasificar períodos
    print("\n[5/6] Reclasificando períodos por fecha...")
    df_spark = reclasificar_periodo_por_fecha_spark(df_spark)
    
    # Ejecutar análisis
    print("\n[6/6] Ejecutando análisis...")
    
    print("\n  Instancias por período:")
    df_spark.groupby('periodo').count().orderBy('periodo').show()
    
    print("\n  Análisis 1: Intensidad, ocupación y carga")
    analisis_1 = analisis_intensidad_ocupacion_carga_spark(df_spark)
    print(analisis_1)
    analisis_1.to_csv(f'{OUTPUT_DIR}/analisis_01_intensidad_ocupacion_carga.csv', index=False)
    
    print("\n  Análisis 2: Carga por hora")
    analisis_2 = analisis_carga_por_hora_spark(df_spark)
    print(analisis_2)
    analisis_2.to_csv(f'{OUTPUT_DIR}/analisis_02_carga_por_hora.csv', index=False)
    
    print("\n  Análisis 3: Laborables vs fin de semana")
    analisis_3 = analisis_laborables_vs_finde_spark(df_spark)
    print(analisis_3)
    analisis_3.to_csv(f'{OUTPUT_DIR}/analisis_03_laborables_vs_finde.csv', index=False)
    
    print("\n  Análisis 4: Tipo de vía")
    analisis_4 = analisis_tipo_via_spark(df_spark)
    print(analisis_4)
    analisis_4.to_csv(f'{OUTPUT_DIR}/analisis_04_tipo_via.csv', index=False)
    
    # Generar gráficos
    print("\n  Generando gráficos...")
    grafico_intensidad_ocupacion_carga(analisis_1)
    grafico_carga_por_hora(analisis_2)
    grafico_laborables_vs_finde(analisis_3)
    grafico_tipo_via(analisis_4)
    
    print("\n" + "="*80)
    print("✓ ANÁLISIS COMPLETADO")
    print("="*80)
    print(f"\nArchivos guardados en: {OUTPUT_DIR}/")
    
    spark.stop()
    print("✓ Sesión Spark cerrada")