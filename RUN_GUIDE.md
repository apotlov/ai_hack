# 🚀 Руководство по запуску антифрод системы

## 📁 Структура данных

Перед запуском убедитесь, что ваши данные размещены в правильной структуре:

```
hackathon/
├── data_train/              # Данные для обучения
│   ├── amplitude/           # Parquet файлы обучения
│   │   ├── train_amplitude_chunk_*.parquet
│   │   ├── train_app_data.parquet
│   │   └── train_target_data.parquet
│   ├── audiofiles/          # Аудиозаписи обучения
│   │   └── *.wav
│   └── svod.csv             # Связки (в корне data_train)
│
├── data/                    # Данные для предсказаний
│   ├── amplitude/           # Parquet файлы предсказаний
│   │   ├── train_amplitude_chunk_*.parquet (без target_data!)
│   │   ├── train_app_data.parquet
│   │   └── svod.csv         # Связки (в amplitude папке)
│   └── audiofiles/          # Аудиозаписи предсказаний
│       └── *.wav
│
├── run.sh                   # Главный скрипт запуска
└── ...
```

## ⚡ Быстрый запуск

### 1. Полный цикл (обучение + предсказания)
```bash
./run.sh
```

### 2. Только обучение модели
```bash
./run.sh --train-only
```

### 3. Только предсказания (требует обученной модели)
```bash
./run.sh --predict-only
```

### 4. Проверка системы
```bash
./run.sh --check
```

## 📋 Предварительные требования

### Python зависимости:
```bash
pip install -r requirements.txt
```

### Ollama (для LLM объяснений):
```bash
# Установка Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Запуск сервера
ollama serve

# Загрузка модели
ollama pull llama3.2:3b
```

## 🔄 Процесс выполнения

### Этап 1: Обучение (data_train/)
1. ✅ Проверка структуры данных
2. 🔧 Извлечение признаков из app_data
3. 🤖 Обучение Random Forest модели
4. 💾 Сохранение модели: `models/real_antifraud_model.joblib`
5. 📄 Создание отчета: `output/training_report.txt`

### Этап 2: Предсказания (data/)
1. 📊 Загрузка обученной модели
2. 🔧 Извлечение признаков из новых данных
3. 🔮 Генерация предсказаний
4. 🤖 LLM анализ и объяснения
5. 📄 Создание отчетов

## 📊 Результаты

После успешного выполнения вы получите:

### Модель:
- `models/real_antifraud_model.joblib` - обученная модель

### Отчеты:
- `output/training_report.txt` - отчет об обучении
- `output/real_data_predictions_with_llm.csv` - предсказания с LLM
- `output/real_data_fraud_analysis.html` - интерактивный HTML отчет
- `output/real_data_fraud_report.txt` - текстовый аналитический отчет

### Логи:
- `logs/training.log` - логи обучения
- `logs/predictions.log` - логи предсказаний

## 🛠️ Устранение неполадок

### Ошибка: "Папка data_train не найдена"
```bash
# Создайте правильную структуру
mkdir -p data_train/amplitude data_train/audiofiles
mkdir -p data/amplitude data/audiofiles
```

### Ошибка: "Нет parquet файлов"
```bash
# Убедитесь что файлы находятся в правильных папках:
ls data_train/amplitude/*.parquet
ls data/amplitude/*.parquet
```

### Ошибка: "Ollama не установлена"
```bash
# Установка и настройка Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull llama3.2:3b
```

### Ошибка: "Python зависимости"
```bash
# Установка зависимостей
pip install pandas numpy scikit-learn tqdm librosa soundfile pyarrow
```

## 🎯 Ключевые особенности

### Упрощенная стратегия
- **Используются только app_data** для совместимости
- **Избегаются amplitude данные** из-за разных типов ID
- **Автоматическая очистка** типов данных

### Автоматизация
- **Проверка зависимостей** перед запуском
- **Валидация структуры данных**
- **Прогресс-бары** для отслеживания
- **Детальное логирование**

### LLM интеграция
- **Локальная обработка** через Ollama
- **Объяснения на русском языке**
- **Рекомендации по действиям**
- **HTML отчеты** с интерактивностью

## 📞 Поддержка

### Проверка системы:
```bash
./run.sh --check
```

### Просмотр логов:
```bash
tail -f logs/training.log
tail -f logs/predictions.log
```

### Справка:
```bash
./run.sh --help
```

## ⚠️ Важные замечания

1. **Размещение svod.csv**:
   - В `data_train/` - файл в корне
   - В `data/` - файл в папке `amplitude/`

2. **Целевые метки**:
   - Нужны только для обучения (`data_train/`)
   - Не нужны для предсказаний (`data/`)

3. **Совместимость данных**:
   - Признаки должны быть одинаковыми в обучении и предсказаниях
   - Используется только `app_data` для стабильности

4. **Производительность**:
   - Обучение: 5-15 минут
   - Предсказания: 10-30 минут (с LLM)
   - Требования: 8GB+ RAM

---

**🎉 Готово к использованию! Запустите `./run.sh` для начала работы.**