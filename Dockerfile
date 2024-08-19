# Предположим, что у вас уже есть базовый образ
FROM python:3.11-slim

# Установка необходимых пакетов
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    python3-dev \
    cython3  # заменяем cython на cython3

# Устанавливаем зависимости через pip
COPY requirements.txt /app/
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Копируем приложение и запускаем его
COPY . /app
CMD ["python", "app.py"]
