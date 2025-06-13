#!/bin/bash

# Активация виртуального окружения
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Запуск обучения модели
python src/train_model.py

# Запуск Flask приложения
gunicorn --bind 0.0.0.0:5000 src.api.app:app
