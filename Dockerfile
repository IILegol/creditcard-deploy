FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода и модели
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Открытие порта
EXPOSE 5000

# Запуск приложения
CMD ["python", "app/app.py"]