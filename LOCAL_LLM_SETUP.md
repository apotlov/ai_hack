# 🤖 Руководство по установке локальной LLM

Подробное руководство по настройке локальной LLM для антифрод системы через Ollama.

## 🎯 Зачем локальная LLM?

### Преимущества:
- ✅ **Бесплатно** - никаких затрат на API
- ✅ **Приватность** - данные не покидают вашу систему
- ✅ **Автономность** - работает без интернета
- ✅ **Контроль** - полный контроль над моделью
- ✅ **Скорость** - нет задержек сети

### Недостатки OpenAI API:
- ❌ Стоимость: $2-60 за 1000 объяснений
- ❌ Зависимость от интернета
- ❌ Отправка данных в OpenAI
- ❌ Лимиты на запросы

## 🚀 Быстрая установка

### Шаг 1: Установка Ollama

#### macOS:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Linux:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Windows:
1. Скачайте установщик: https://ollama.ai/download
2. Запустите .exe файл
3. Следуйте инструкциям

#### Альтернативно (через Docker):
```bash
docker pull ollama/ollama
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### Шаг 2: Запуск Ollama сервера

```bash
ollama serve
```

**Важно**: Оставьте этот терминал открытым - сервер должен работать постоянно.

### Шаг 3: Выбор и загрузка модели

#### Рекомендуемые модели:

1. **Llama 3.2 3B** (Лучший баланс):
```bash
ollama pull llama3.2:3b
```
- Размер: ~6GB
- Требования: 8GB RAM
- Качество: Отличное

2. **Phi-3.5 Mini** (Быстрая):
```bash
ollama pull phi3.5
```
- Размер: ~4GB
- Требования: 6GB RAM
- Качество: Хорошее

3. **Qwen2.5 7B** (Максимальное качество):
```bash
ollama pull qwen2.5:7b
```
- Размер: ~8GB
- Требования: 12GB RAM
- Качество: Превосходное
- Поддержка русского языка

### Шаг 4: Проверка установки

```bash
# Проверить статус
ollama list

# Тест модели
ollama run llama3.2:3b "Привет, как дела?"
```

## 📋 Системные требования

### Минимальные:
- **RAM**: 6GB свободной памяти
- **Диск**: 10GB свободного места
- **CPU**: Любой современный процессор

### Рекомендуемые:
- **RAM**: 12GB+ (для лучшей производительности)
- **Диск**: SSD для быстрой загрузки
- **GPU**: NVIDIA GPU с CUDA (опционально, для ускорения)

### Проверка ресурсов:

#### Проверка RAM:
```bash
# Linux/macOS
free -h

# Windows
wmic memorychip get size
```

#### Проверка диска:
```bash
# Linux/macOS
df -h

# Windows
dir
```

## 🔧 Настройка для антифрод системы

### Установка дополнительных зависимостей:

```bash
pip install requests psutil
```

### Тест интеграции:

```bash
cd hackathon
python3 scripts/predict_local_llm.py
```

### Выбор модели по ресурсам:

```python
# Автоматический выбор в коде
import psutil

ram_gb = psutil.virtual_memory().total / (1024**3)

if ram_gb >= 16:
    model = "qwen2.5:7b"      # Лучшее качество
elif ram_gb >= 8:
    model = "llama3.2:3b"     # Оптимальный
else:
    model = "phi3.5"          # Экономичный
```

## 🎨 Примеры использования

### Базовое использование:

```python
from src.local_llm_enhancer import LocalLLMEnhancer

# Инициализация
llm = LocalLLMEnhancer(model="llama3.2:3b")

# Объяснение решения
explanation = llm.explain_fraud_decision(
    user_data={'user_id': 'user_123'},
    features={'night_activity_ratio': 0.8},
    probability=0.85
)

print(explanation.explanation)
# "Высокая вероятность мошенничества из-за необычной 
#  активности в ночное время (80%)"
```

### Полный пайплайн:

```bash
# 1. Стандартное обучение
python3 scripts/main.py --train

# 2. Предсказания с локальной LLM
python3 scripts/predict_local_llm.py
```

## 📊 Сравнение моделей

| Модель | Размер | RAM | Скорость | Качество | Русский |
|--------|--------|-----|----------|----------|---------|
| phi3.5 | 4GB | 6GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| llama3.2:3b | 6GB | 8GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| qwen2.5:7b | 8GB | 12GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Рекомендации по выбору:

- **Разработка/MVP**: phi3.5 (быстро, достаточно)
- **Продакшен**: llama3.2:3b (баланс качества и скорости)  
- **Максимальное качество**: qwen2.5:7b (если хватает ресурсов)

## ⚡ Оптимизация производительности

### Настройки Ollama:

```bash
# Увеличить количество потоков
export OLLAMA_NUM_PARALLEL=4

# Использовать GPU (если есть NVIDIA)
export OLLAMA_GPU_LAYERS=35

# Увеличить контекст
export OLLAMA_MAX_CONTEXT=4096
```

### Настройки в коде:

```python
# В local_llm_enhancer.py
payload = {
    "model": self.model,
    "prompt": prompt,
    "options": {
        "num_predict": 200,     # Короче ответы = быстрее
        "temperature": 0.3,     # Менее креативно = стабильнее
        "top_p": 0.9,          # Фокус на лучших вариантах
        "num_ctx": 2048        # Контекст для понимания
    }
}
```

### Мониторинг ресурсов:

```bash
# Использование RAM
htop

# Использование GPU (если есть)
nvidia-smi

# Статистика Ollama
curl http://localhost:11434/api/ps
```

## 🐛 Устранение неполадок

### Частые проблемы:

#### 1. "Ollama не установлена"
```bash
# Проверить установку
which ollama

# Переустановить
curl -fsSL https://ollama.ai/install.sh | sh
```

#### 2. "Сервер недоступен"
```bash
# Проверить статус
curl http://localhost:11434/api/tags

# Перезапустить сервер
pkill ollama
ollama serve
```

#### 3. "Модель не найдена"
```bash
# Посмотреть доступные модели
ollama list

# Загрузить нужную модель
ollama pull llama3.2:3b
```

#### 4. "Недостаточно памяти"
```bash
# Проверить использование RAM
free -h

# Использовать более легкую модель
ollama pull phi3.5
```

#### 5. "Медленная работа"
```bash
# Уменьшить размер ответов
export OLLAMA_MAX_TOKENS=150

# Использовать GPU
export OLLAMA_GPU_LAYERS=20

# Меньшая модель
ollama pull phi3.5
```

### Диагностика:

```python
# Тест в коде
from src.local_llm_enhancer import LocalLLMEnhancer, OllamaInstaller

# Проверка установки
print("Ollama установлена:", OllamaInstaller.check_ollama_installed())

# Рекомендуемая модель
print("Рекомендуемая модель:", OllamaInstaller.recommend_model_by_resources())

# Тест соединения
llm = LocalLLMEnhancer("llama3.2:3b")
print("Тест прошел:", llm.test_connection())
```

## 🔄 Обновление и управление

### Обновление Ollama:

```bash
# Обновить Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Обновить модель
ollama pull llama3.2:3b
```

### Управление моделями:

```bash
# Список моделей
ollama list

# Удалить модель
ollama rm old-model

# Информация о модели
ollama show llama3.2:3b
```

### Очистка места:

```bash
# Найти большие файлы Ollama
du -h ~/.ollama

# Удалить неиспользуемые модели
ollama rm unused-model
```

## 📈 Мониторинг и логи

### Логи Ollama:

```bash
# Посмотреть логи
tail -f ~/.ollama/logs/server.log

# Статус сервера
curl http://localhost:11434/api/tags
```

### Мониторинг в коде:

```python
# В вашем коде добавьте
import time
import logging

start_time = time.time()
explanation = llm.explain_fraud_decision(...)
end_time = time.time()

logging.info(f"LLM ответ за {end_time - start_time:.2f} сек")
```

## 🔧 Альтернативные варианты

### 1. LM Studio (GUI интерфейс)
- Скачать: https://lmstudio.ai/
- Графический интерфейс
- Легко управлять моделями

### 2. Text Generation WebUI
```bash
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r requirements.txt
```

### 3. GPT4All (Автономный)
```bash
pip install gpt4all
```

```python
from gpt4all import GPT4All
model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
response = model.generate("Explain fraud detection")
```

## 📋 Чек-лист установки

- [ ] Ollama установлена (`ollama --version`)
- [ ] Сервер запущен (`ollama serve`)
- [ ] Модель загружена (`ollama list`)
- [ ] Тест модели (`ollama run model "test"`)
- [ ] Python зависимости (`pip install requests`)
- [ ] Тест интеграции (`python3 scripts/predict_local_llm.py`)
- [ ] Достаточно RAM (>6GB свободной)
- [ ] Достаточно диска (>10GB)

## 🎯 Результаты работы

После настройки вы получите:

### Файлы результатов:
- `fraud_predictions_local_llm.csv` - предсказания с объяснениями
- `fraud_report_local_llm.txt` - текстовый отчет
- `fraud_report_local_llm.html` - интерактивный HTML отчет

### Пример объяснения:
```
User user_123 - Риск: 85%
Объяснение: Высокая вероятность мошенничества из-за необычной 
активности в ночное время и короткой продолжительности сессий.

Ключевые факторы:
• Активность ночью: 0.80
• Количество сессий: 3.00
• Средняя длительность сессии: 45.00

Рекомендации:
• Немедленно заблокировать подозрительные операции
• Связаться с клиентом для подтверждения
```

## 🆘 Поддержка

### Официальная документация:
- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/README.md)
- [Model Library](https://ollama.ai/library)

### Полезные ссылки:
- [Discord сообщество Ollama](https://discord.gg/ollama)
- [GitHub Issues](https://github.com/ollama/ollama/issues)

### Если ничего не работает:
1. Перезагрузите компьютер
2. Переустановите Ollama
3. Попробуйте модель phi3.5 (самая стабильная)
4. Проверьте антивирус (может блокировать)

---

**Готово!** 🎉 

Ваша антифрод система теперь работает с локальной LLM:

```bash
# Полный пайплайн с локальной LLM
python3 scripts/predict_local_llm.py
```

🔒 Все данные остаются на вашем компьютере!
💰 Никаких затрат на API!
⚡ Работает без интернета!