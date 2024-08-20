# Utilizar la imagen base de Python
FROM python:3.10
# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requisitos e instalar las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el contenido de tu aplicación al contenedor
COPY . .

# Copiar el archivo de credenciales de Google
COPY ./config/google-credentials.json /app/config/google-credentials.json

# Establecer la variable de entorno para las credenciales de Google
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/config/google-credentials.json

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]