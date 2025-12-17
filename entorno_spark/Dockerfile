FROM apache/spark:3.5.1

USER root

# Instalar Python 3 y pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Instalar PySpark y librerías necesarias
RUN pip3 install --upgrade pip && \
    pip3 install pyspark==3.5.1 pandas matplotlib seaborn python-dotenv

# Crear directorio de trabajo
WORKDIR /opt/spark/app

# Crear directorios necesarios
RUN mkdir -p /opt/spark/app/datos && \
    mkdir -p /opt/spark/app/outputs

# Copiar archivos
COPY analisis_spark.py /opt/spark/app/
COPY .env /opt/spark/app/
COPY datos /opt/spark/app/datos/

# Cambiar permisos
RUN chown -R spark:spark /opt/spark/app

# Volver al usuario spark
USER spark

# Establecer variables de entorno
ENV PYTHONUNBUFFERED=1
ENV SPARK_LOCAL_IP=0.0.0.0