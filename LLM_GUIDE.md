# 🤖 Руководство по интеграции LLM в антифрод систему

Подробное руководство по добавлению возможностей больших языковых моделей (LLM) в систему детекции мошенничества.

## 📋 Обзор интеграции LLM

### Что добавляет LLM к системе:
- **Объяснения решений** - понятные объяснения почему транзакция подозрительна
- **Анализ паттернов** - выявление сложных трендов мошенничества
- **Генерация отчетов** - автоматические отчеты на естественном языке
- **Рекомендации** - конкретные действия для каждого случая

### Текущая архитектура:
```
Базовая система (без LLM):
Данные → Признаки → Random Forest → Предсказания

Расширенная система (с LLM):
Данные → Признаки → Random Forest → Предсказания → LLM → Объяснения + Отчеты
```

## 🚀 Быстрый старт с LLM

### 1. Установка дополнительных зависимостей

```bash
# Добавить в requirements.txt:
openai>=1.0.0
```

```bash
pip install openai
```

### 2. Настройка API ключа

```bash
# Вариант 1: Переменная окружения
export OPENAI_API_KEY="your-api-key-here"

# Вариант 2: В коде (не рекомендуется для продакшена)
api_key = "your-api-key-here"
```

### 3. Запуск с LLM объяснениями

```bash
# Базовое предсказание (без LLM)
python3 scripts/main.py --predict

# Расширенное предсказание (с LLM)
python3 scripts/predict_with_llm.py
```

## 🔧 Настройка LLM компонентов

### LLMFraudEnhancer - основной класс

```python
from src.llm_enhancer import LLMFraudEnhancer

# С OpenAI API
enhancer = LLMFraudEnhancer(api_key="your-key")

# Демо режим (без API)
enhancer = LLMFraudEnhancer(api_key=None)
```

### Поддерживаемые модели:

```python
# GPT-3.5 (рекомендуется для MVP)
enhancer = LLMFraudEnhancer(api_key="key", model="gpt-3.5-turbo")

# GPT-4 (более качественные объяснения, дороже)
enhancer = LLMFraudEnhancer(api_key="key", model="gpt-4")

# GPT-4-turbo (быстрее и дешевле GPT-4)
enhancer = LLMFraudEnhancer(api_key="key", model="gpt-4-turbo-preview")
```

## 💡 Возможности LLM модуля

### 1. Объяснения решений

```python
# Генерация объяснения для конкретного случая
explanation = enhancer.explain_fraud_decision(
    user_data={'user_id': 'user_123'},
    features={'night_activity_ratio': 0.8, 'session_count': 3},
    probability=0.85
)

print(explanation.explanation)
# "Высокая вероятность мошенничества из-за необычной активности 
#  в ночное время (80%) и малого количества сессий (3)"

print(explanation.recommendations)
# ["Немедленно заблокировать операции", "Связаться с клиентом"]
```

### 2. Анализ паттернов мошенничества

```python
# Анализ группы мошеннических случаев
fraud_cases = [
    {'user_id': 'user_1', 'probability': 0.9, 'risk_level': 'Высокий'},
    {'user_id': 'user_2', 'probability': 0.8, 'risk_level': 'Высокий'}
]

report = enhancer.generate_fraud_report(fraud_cases)
print(report)
# "Обнаружены паттерны мошенничества: активность в ночное время,
#  множественные мелкие транзакции, новые получатели..."
```

### 3. Расширенные предсказания

```python
# Добавление объяснений к предсказаниям
enhanced_predictions = enhancer.enhance_predictions_with_explanations(
    predictions_df=predictions,
    features_df=features
)

# Новые колонки:
# - llm_explanation: объяснение решения
# - key_factors: ключевые факторы
# - recommendations: рекомендуемые действия
# - explanation_confidence: уверенность в объяснении
```

## 📊 Типы выходных данных с LLM

### 1. CSV с объяснениями
```csv
user_id,fraud_probability,risk_level,llm_explanation,key_factors,recommendations
user_1,0.85,Высокий,"Подозрительная активность в ночное время...","Ночная активность: 80%","Блокировать операции | Связаться с клиентом"
```

### 2. Детальный текстовый отчет
```
=== ДЕТАЛЬНЫЙ ОТЧЕТ АНТИФРОД СИСТЕМЫ С LLM ===
Дата: 2024-01-15 14:30:00

СТАТИСТИКА:
- Всего проанализировано: 100
- Высокий риск (≥70%): 15
- Средний риск (30-70%): 25

АНАЛИЗ ПАТТЕРНОВ (LLM):
Обнаружены следующие тренды мошенничества:
1. Увеличение активности в ночное время
2. Множественные мелкие транзакции
3. Операции с новыми получателями
...
```

### 3. HTML интерактивный отчет
- Цветовая кодировка по уровням риска
- Детальная информация по каждому случаю
- Объяснения и рекомендации
- Статистические графики

## ⚙️ Конфигурация промптов

### Настройка промптов для объяснений:

```python
# Кастомный промпт для объяснений
custom_prompt = """
Проанализируй данные банковского клиента и объясни решение антифрод системы.

Данные клиента: {user_data}
Признаки модели: {features}
Вероятность мошенничества: {probability:.1%}

Требуется:
1. Краткое объяснение (до 100 слов)
2. 3 главных подозрительных фактора
3. Конкретные рекомендации
4. Оценка уверенности

Стиль: профессиональный, для банковского аналитика.
"""

# Применение кастомного промпта
enhancer.explanation_prompts["high_risk"] = custom_prompt
```

### Промпты для разных сценариев:

```python
# Для высокого риска - детальный анализ
# Для среднего риска - основные факторы
# Для низкого риска - краткое подтверждение

# Настройка через конфигурацию
prompts_config = {
    "high_risk_detail": "Детальный анализ для высокого риска...",
    "medium_risk_summary": "Краткий анализ для среднего риска...",
    "pattern_analysis": "Анализ паттернов мошенничества..."
}
```

## 💰 Управление затратами

### Стоимость использования OpenAI API:

```python
# GPT-3.5-turbo (рекомендуется)
# Input: $0.0015 / 1K tokens
# Output: $0.002 / 1K tokens

# Примерная стоимость на 1000 объяснений: $2-5

# GPT-4 (более качественно, но дороже)
# Input: $0.03 / 1K tokens  
# Output: $0.06 / 1K tokens

# Примерная стоимость на 1000 объяснений: $30-60
```

### Оптимизация затрат:

```python
# 1. Используйте GPT-3.5 для MVP
enhancer = LLMFraudEnhancer(model="gpt-3.5-turbo")

# 2. Ограничивайте длину ответов
max_tokens = 200  # Вместо 1000 по умолчанию

# 3. Кэшируйте похожие запросы
cache_explanations = True

# 4. Обрабатывайте только высокий риск
if probability > 0.7:
    explanation = enhancer.explain_fraud_decision(...)
```

## 🔒 Безопасность и приватность

### Защита данных:

```python
# 1. Не отправляйте чувствительные данные
sensitive_fields = ['phone', 'email', 'card_number', 'account_id']

# 2. Анонимизируйте user_id
user_data = {'user_id': hash_user_id(original_id)}

# 3. Используйте агрегированные признаки
features = aggregate_features(raw_features)

# 4. Логируйте все LLM запросы
logger.info(f"LLM request for user {anonymized_id}")
```

### Соответствие требованиям:

- **GDPR**: Анонимизация персональных данных
- **PCI DSS**: Не передавайте платежные данные
- **Банковские требования**: Логирование всех решений

## 🚨 Режимы работы

### 1. Демо режим (без API ключа)
```python
# Использует правила и шаблоны
enhancer = MockLLMEnhancer()
explanation = enhancer.explain_fraud_decision(...)
# Результат: базовые объяснения без LLM
```

### 2. Гибридный режим
```python
# LLM только для высокого риска
if probability > 0.7:
    explanation = llm_enhancer.explain_fraud_decision(...)
else:
    explanation = rule_based_explanation(...)
```

### 3. Полный LLM режим
```python
# LLM для всех случаев
for case in all_cases:
    explanation = llm_enhancer.explain_fraud_decision(...)
```

## 📈 Метрики качества LLM

### Автоматическая оценка:

```python
# Уверенность в объяснении (0-1)
confidence = explanation.confidence

# Полнота объяснения (количество факторов)
completeness = len(explanation.key_factors)

# Релевантность рекомендаций
relevance = evaluate_recommendations(explanation.recommendations)
```

### Ручная оценка:

- **Понятность** - объяснения понятны аналитику
- **Точность** - объяснения соответствуют данным  
- **Полезность** - рекомендации практичны
- **Консистентность** - похожие случаи объясняются похоже

## 🔧 Интеграция с существующей системой

### Обновление основного скрипта:

```python
# В main.py добавить опцию LLM
parser.add_argument("--with-llm", action="store_true", 
                   help="Использовать LLM объяснения")

if args.with_llm:
    # Запуск с LLM
    result = run_prediction_with_llm()
else:
    # Обычный запуск
    result = run_prediction()
```

### Обновление пайплайна:

```bash
# Стандартный пайплайн
python3 scripts/main.py --full

# Пайплайн с LLM
python3 scripts/main.py --full --with-llm
```

## 🐛 Устранение неполадок

### Частые проблемы:

1. **API ключ не работает**
```bash
# Проверьте ключ
echo $OPENAI_API_KEY

# Проверьте баланс на OpenAI
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

2. **Превышение лимитов**
```python
# Добавьте retry логику
import time
try:
    response = openai.ChatCompletion.create(...)
except openai.error.RateLimitError:
    time.sleep(60)  # Подождать минуту
    response = openai.ChatCompletion.create(...)
```

3. **Качество объяснений низкое**
```python
# Улучшите промпты
# Добавьте примеры в промпт
# Используйте GPT-4 вместо GPT-3.5
# Увеличьте max_tokens
```

4. **Медленная работа**
```python
# Обрабатывайте батчами
# Используйте асинхронные запросы
# Кэшируйте результаты
```

## 📋 Чек-лист интеграции LLM

- [ ] Установлен пакет `openai`
- [ ] Настроен API ключ OpenAI
- [ ] Протестирован демо режим
- [ ] Настроены промпты под ваш домен
- [ ] Реализовано логирование LLM запросов
- [ ] Настроена анонимизация данных
- [ ] Определен бюджет на API вызовы
- [ ] Протестирована производительность
- [ ] Настроена обработка ошибок
- [ ] Документированы новые возможности

## 🎯 Рекомендации по использованию

### Для MVP (минимальная версия):
- Используйте демо режим без API
- Фокус на правилах и шаблонах
- Основные объяснения без LLM

### Для рабочей версии:
- OpenAI API с GPT-3.5-turbo
- Объяснения для случаев высокого риска
- Базовые отчеты с LLM анализом

### Для продвинутой версии:
- GPT-4 для качественных объяснений
- Полный анализ всех случаев
- Интерактивные отчеты
- Кастомные промпты для домена

## 🔮 Будущие возможности

### Планируемые улучшения:
- **Тонкая настройка** модели на банковских данных
- **Многоязычные** объяснения
- **Голосовые** объяснения для операторов
- **Интеграция с внешними** системами
- **Real-time** объяснения

### Альтернативные LLM:
- **Claude** от Anthropic
- **LLaMA** локальное развертывание
- **Azure OpenAI** для корпоративных клиентов

---

## 🆘 Поддержка и документация

**Официальная документация:**
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)

**Примеры использования:**
```bash
# Базовый пример
python3 scripts/predict_with_llm.py

# С кастомным API ключом
OPENAI_API_KEY="your-key" python3 scripts/predict_with_llm.py

# Только демо режим
python3 scripts/predict_with_llm.py --demo-mode
```

**Готово!** 🚀 Ваша антифрод система теперь может объяснять свои решения с помощью LLM.