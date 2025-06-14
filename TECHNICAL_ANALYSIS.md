# 🔧 Техническая документация антифрод системы

> **Статус**: Критическая проблема с объединением данных требует немедленного исправления

## 📊 Текущая диагностика системы

### 🚨 Критическая проблема: Нулевое пересечение данных

```
📈 Загруженные данные:
├── Target Data: 18,386 записей (✅ загружено)
├── App Data: 18,386 записей (✅ загружено) 
├── Amplitude Data: 13,439 записей (✅ загружено)
└── Audio Files: 4,218 файлов (✅ найдено)

❌ Результат объединения: 0 записей
🎯 Целевых меток после merge: 0
```

**Причина**: Полное отсутствие пересечения ключей при объединении по `APPLICATIONID`

### 🔍 Анализ структуры ключей

#### Escape-символы в APPLICATIONID:
```python
# Target Data примеры
"Д\286\011639474"  # Содержит \286\ и \011
"Д\286\011668478" 
"Д\286\011681466"

# Amplitude Data примеры  
"Д\286\011221568"  # Другие ID, тоже с escape
"Д\286\011221568"
```

#### Проблемы кодировки:
1. **Unicode escape sequences**: `\286\011` может интерпретироваться как специальные символы
2. **String vs bytes**: pandas может по-разному обрабатывать эти последовательности
3. **Encoding issues**: latin1 vs utf-8 декодирование
4. **Case sensitivity**: `applicationid` vs `APPLICATIONID`

## 🔧 Архитектура компонентов

### 📊 RealDataLoader - Загрузчик данных

**Класс**: `hackathon/src/real_data_loader.py`

```python
class RealDataLoader:
    def __init__(self, data_dir: str):
        self.amplitude_dir = data_dir / "amplitude" 
        self.audio_dir = data_dir / "audiofiles"
    
    # Ключевые методы:
    def load_amplitude_chunks() -> pd.DataFrame    # ✅ 13,439 записей
    def load_app_data() -> pd.DataFrame           # ✅ 18,386 записей  
    def load_target_data() -> pd.DataFrame        # ✅ 18,386 записей
    def get_audio_files_metadata() -> pd.DataFrame # ✅ 4,218 файлов
```

**Статус**: ✅ Все методы работают корректно, данные загружаются

### 🔧 RealFeaturesProcessor - Процессор признаков

**Класс**: `hackathon/src/real_features_processor.py`

```python
class RealFeaturesProcessor:
    def combine_all_features() -> Tuple[pd.DataFrame, pd.Series]:
        # ❌ ПРОБЛЕМА ЗДЕСЬ: merge возвращает пустой результат
        
        # Шаги обработки:
        amplitude_features = extract_amplitude_features()  # ✅ (13,439, 116)
        app_features = extract_app_features()             # ✅ (18,386, 64) 
        audio_features = extract_audio_features()         # ✅ Работает
        temporal_features = extract_temporal_features()   # ✅ Работает
        
        # ❌ ПРОБЛЕМА: Объединение по applicationid
        combined = merge(amplitude_features, app_features, on='applicationid')
        
        # ❌ ПРОБЛЕМА: Связывание с target_data
        final_data = merge(combined, target_data, 
                          left_on='applicationid', 
                          right_on='APPLICATIONID')  # → Пустой результат
```

**Проблемные методы**:
- `combine_all_features()` - возвращает пустые DataFrame и Series
- `_find_group_column()` - находит колонки, но merge не работает
- `_normalize_keys()` - добавлена отладка, но нужно доработать

### 🎵 AudioProcessor - Обработка аудио

**Класс**: `hackathon/src/audio_processor.py`

```python  
class AudioProcessor:
    def extract_audio_features(file_path: str) -> Dict:
        # MFCC признаки (13 коэффициентов)
        # Спектральные признаки (centroid, bandwidth, rolloff)
        # Временные признаки (RMS, zero_crossing_rate)
        # Harmony и Percussive компоненты
```

**Статус**: ✅ Работает корректно, признаки извлекаются

### 📈 ModelTrainer - Обучение модели

**Класс**: `hackathon/src/model_trainer.py`

```python
class ModelTrainer:
    def train_model(X: pd.DataFrame, y: pd.Series):
        # ❌ НЕ МОЖЕТ ОБУЧИТЬСЯ: X и y пустые из-за проблемы выше
        
        # Конфигурация модели:
        RandomForestClassifier(
            n_estimators=100,
            max_depth=10, 
            class_weight='balanced',  # Для дисбаланса классов
            random_state=42
        )
```

**Статус**: ❌ Блокирован пустыми данными

### 🤖 LocalLLMEnhancer - LLM интеграция

**Класс**: `hackathon/src/local_llm_enhancer.py`

```python
class LocalLLMEnhancer:
    def __init__(self, model="llama3.2:3b", base_url="http://localhost:11434"):
        # Ollama интеграция для локальной LLM
    
    def explain_prediction(user_id, probability, features) -> FraudExplanation:
        # Генерация объяснений на русском языке
        # Анализ ключевых факторов
        # Рекомендации по действиям
```

**Статус**: ✅ Готов к работе (ожидает исправления данных)

## 🔍 Детальная диагностика проблемы

### Добавленная отладочная информация:

```python
# В real_features_processor.py добавлено:
def combine_all_features():
    # ... existing code ...
    
    logger.info(f"🔍 Целевая колонка: {target_col}")
    logger.info(f"🔍 Колонка для слияния: {merge_col}")
    logger.info(f"🔍 Доступные колонки в target_data: {list(target_data.columns)}")
    logger.info(f"🔍 Доступные колонки в combined_features: {list(combined_features.columns)}")
    
    # Нормализация ключей для обработки escape символов
    def normalize_key(key):
        if pd.isna(key):
            return ""
        key_str = str(key).strip()
        try:
            key_str = key_str.encode('latin1').decode('unicode_escape')
        except:
            pass
        return key_str.upper()
    
    # Проверка пересечения ключей
    features_keys = set(combined_features['applicationid_normalized'])
    target_keys = set(target_data_normalized[merge_col + '_normalized'])
    intersection = features_keys.intersection(target_keys)
    
    logger.info(f"🔍 Пересечение ключей: {len(intersection)} из {len(features_keys)} features и {len(target_keys)} target")
```

### Ожидаемые результаты отладки:

После запуска `python scripts/train_real_data.py` должны появиться логи:

```
🔍 Целевая колонка: target
🔍 Колонка для слияния: APPLICATIONID
🔍 Доступные колонки в target_data: ['APPLICATIONID', 'CREATE_DATE', 'DEL_F1PD_CNT', ...]
🔍 Доступные колонки в combined_features: ['applicationid', 'amplitude_mean', 'app_age', ...]
🔍 Уникальных applicationid в features: XXXXX
🔍 Примеры applicationid в features: ['Д\286\011221568', ...]
🔍 Уникальных APPLICATIONID в target_data: XXXXX  
🔍 Примеры APPLICATIONID в target_data: ['Д\286\011639474', ...]
🔍 Пересечение ключей: 0 из XXXXX features и XXXXX target
```

## 🛠️ План исправления

### Этап 1: Диагностика (✅ В процессе)

1. **Запуск с отладкой**:
   ```bash
   python scripts/train_real_data.py 2>&1 | tee debug_log.txt
   ```

2. **Анализ ключей**: Сравнить конкретные значения APPLICATIONID из разных источников

3. **Проверка кодировки**: Убедиться что escape символы обрабатываются одинаково

### Этап 2: Исправление нормализации ключей

**Проблема**: Текущий `normalize_key()` может работать некорректно

**Решение**: Улучшенная нормализация
```python
def normalize_applicationid(app_id):
    """Улучшенная нормализация APPLICATIONID"""
    if pd.isna(app_id) or app_id == '':
        return None
        
    # Приводим к строке
    app_id_str = str(app_id).strip()
    
    # Обрабатываем escape последовательности
    # \286 → соответствующий unicode символ
    # \011 → соответствующий unicode символ
    
    # Вариант 1: Интерпретировать как octal escape
    try:
        app_id_str = app_id_str.encode().decode('unicode_escape')
    except:
        pass
    
    # Вариант 2: Заменить на сырые символы
    # app_id_str = app_id_str.replace('\\', '')
    
    # Вариант 3: Использовать как есть но нормализовать регистр
    return app_id_str.upper().strip()
```

### Этап 3: Fallback стратегии

**Если нормализация не поможет**:

1. **Частичное совпадение**: Поиск похожих ID
2. **Временная привязка**: Связывание по датам
3. **Статистический анализ**: Работа с доступными данными
4. **Синтетические метки**: Создание псевдо-меток для тестирования

### Этап 4: Валидация исправления

```python
# Проверочный скрипт
def validate_data_merge():
    processor = RealFeaturesProcessor("data")
    X, y = processor.combine_all_features()
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {y.value_counts()}")
    
    if len(y) > 0:
        print("✅ Проблема исправлена!")
        return True
    else:
        print("❌ Проблема остается")
        return False
```

## 📋 Технические спецификации

### Структура данных:

#### Target Data Schema:
```python
{
    'APPLICATIONID': str,      # Ключ связывания  
    'CREATE_DATE': datetime,   # Дата создания заявки
    'target': int,            # 0/1 метка мошенничества
    'DEL_F1PD_CNT': int,      # Счетчики просрочек
    'DEL_SPD_CNT': int,
    'DEL_TPD_CNT': int, 
    'DEL_F4PD_CNT': int,
    'tag': object             # Дополнительные теги
}
```

#### App Data Schema:
```python
{
    'APPLICATIONID': str,     # Ключ связывания
    'CREATE_DATE': date,      # Дата заявки  
    'TOTALAMOUNT': float,     # Сумма кредита
    'CREDITTERM_RBL0': int,   # Срок кредита
    'CLI_AGE': int,          # Возраст клиента
    'GENDER': str,           # Пол
    'MARITALSTATUS': str,    # Семейное положение
    'PRODUCT_GROUP': str,    # Группа продукта
    # ... + ~60 других полей
}
```

#### Amplitude Data Schema:
```python
{
    'applicationid': str,     # Ключ связывания (lowercase!)
    'event_time': datetime,   # Время события
    'event_type': str,       # Тип события (session_start, main_page, etc)
    'session_id': float,     # ID сессии
    'device_brand': str,     # Бренд устройства
    'os_name': str,         # ОС (ios, android)
    'ip_address': str,      # IP адрес
    'user_id': str,         # ID пользователя
    # ... + технические поля
}
```

#### Audio Files Metadata:
```python
{
    'file_path': str,           # Путь к файлу
    'applicationid': str,       # Извлечен из svod.csv
    'duration': float,          # Длительность в секундах
    'sample_rate': int,         # Частота дискретизации
    'channels': int,            # Количество каналов
    'file_size': int           # Размер файла
}
```

### Извлекаемые признаки:

#### Amplitude признаки (116 штук):
```python
amplitude_features = {
    # Статистические по событиям
    'event_count': int,              # Общее количество событий
    'unique_event_types': int,       # Уникальные типы событий
    'session_duration_mean': float,  # Средняя длительность сессии
    'session_duration_std': float,   # Стд. отклонение длительности
    
    # Временные паттерны  
    'events_per_hour': float,        # Интенсивность событий
    'peak_activity_hour': int,       # Час пиковой активности
    'activity_variance': float,      # Вариативность активности
    
    # Поведенческие
    'device_changes': int,           # Смены устройства
    'ip_changes': int,              # Смены IP
    'location_changes': int,        # Смены локации
    
    # ... + ~100 других признаков
}
```

#### App признаки (64 штуки):
```python
app_features = {
    # Демографические
    'age': int,                     # Возраст
    'gender_encoded': int,          # Пол (закодированный)
    'marital_status_encoded': int,  # Семейное положение
    
    # Финансовые
    'credit_amount': float,         # Сумма кредита
    'credit_term': int,            # Срок кредита  
    'amount_to_income_ratio': float, # Отношение суммы к доходу
    
    # Кредитная история
    'bki_score': float,            # Скоринг БКИ
    'delinquency_history': int,    # История просрочек
    
    # ... + ~50 других признаков
}
```

#### Audio признаки (~40 штук):
```python
audio_features = {
    # MFCC (Mel-frequency cepstral coefficients)
    'mfcc_1_mean': float,          # Среднее 1-го MFCC коэффициента
    'mfcc_1_std': float,           # Стд. отклонение 1-го MFCC
    # ... до mfcc_13_mean/std
    
    # Спектральные признаки
    'spectral_centroid_mean': float,    # Центроид спектра
    'spectral_bandwidth_mean': float,   # Полоса пропускания
    'spectral_rolloff_mean': float,     # Частота спада
    'zero_crossing_rate_mean': float,   # Частота пересечения нуля
    
    # Энергетические
    'rms_energy_mean': float,           # RMS энергия
    'harmonic_mean': float,             # Гармоническая компонента
    'percussive_mean': float,           # Ударная компонента
    
    # Ритмические  
    'tempo': float,                     # Темп
    'beat_strength': float,             # Сила ритма
}
```

### ML модель конфигурация:

```python
model_config = {
    'algorithm': 'RandomForestClassifier',
    'parameters': {
        'n_estimators': 100,           # Количество деревьев
        'max_depth': 10,               # Максимальная глубина
        'min_samples_split': 5,        # Минимум сэмплов для разбиения
        'min_samples_leaf': 2,         # Минимум сэмплов в листе
        'class_weight': 'balanced',    # Балансировка классов
        'random_state': 42,            # Воспроизводимость
        'n_jobs': -1                   # Параллелизм
    },
    
    'target_metrics': {
        'auc_roc': 0.85,              # Целевая AUC-ROC
        'precision': 0.60,            # Целевая точность
        'recall': 0.40,               # Целевая полнота
        'f1_score': 0.48              # Целевая F1
    },
    
    'class_distribution': {
        'negative_class': 18046,       # Не мошенники (98.15%)
        'positive_class': 340,         # Мошенники (1.85%)
        'imbalance_ratio': 53.1        # Соотношение дисбаланса
    }
}
```

## 🔬 Отладочные инструменты

### Диагностический скрипт:

```python
# hackathon/debug_data_merge.py
import logging
from src.real_data_loader import RealDataLoader
from src.real_features_processor import RealFeaturesProcessor

logging.basicConfig(level=logging.INFO)

def debug_applicationid_matching():
    """Детальная диагностика проблемы с APPLICATIONID"""
    
    # Загружаем данные
    loader = RealDataLoader("data")
    
    # Получаем сырые данные
    amplitude_data = loader.load_amplitude_chunks()
    app_data = loader.load_app_data()  
    target_data = loader.load_target_data()
    
    # Анализируем ключи
    amp_ids = set(amplitude_data['applicationid'].astype(str))
    app_ids = set(app_data['APPLICATIONID'].astype(str))
    target_ids = set(target_data['APPLICATIONID'].astype(str))
    
    print("=== АНАЛИЗ КЛЮЧЕЙ ===")
    print(f"Amplitude IDs: {len(amp_ids)} уникальных")
    print(f"App IDs: {len(app_ids)} уникальных")  
    print(f"Target IDs: {len(target_ids)} уникальных")
    
    print("\n=== ПРИМЕРЫ КЛЮЧЕЙ ===")
    print("Amplitude (первые 5):", list(amp_ids)[:5])
    print("App (первые 5):", list(app_ids)[:5])
    print("Target (первые 5):", list(target_ids)[:5])
    
    print("\n=== ПЕРЕСЕЧЕНИЯ ===")
    amp_app = amp_ids.intersection(app_ids)
    app_target = app_ids.intersection(target_ids)  
    amp_target = amp_ids.intersection(target_ids)
    all_three = amp_ids.intersection(app_ids).intersection(target_ids)
    
    print(f"Amplitude ∩ App: {len(amp_app)}")
    print(f"App ∩ Target: {len(app_target)}") 
    print(f"Amplitude ∩ Target: {len(amp_target)}")
    print(f"Все три: {len(all_three)}")
    
    if len(all_three) > 0:
        print("Примеры общих ID:", list(all_three)[:5])
    
    print("\n=== АНАЛИЗ СИМВОЛОВ ===")
    # Проверяем конкретные символы в ID
    for source, ids in [("Amplitude", amp_ids), ("App", app_ids), ("Target", target_ids)]:
        sample_id = list(ids)[0]
        print(f"{source} sample: {repr(sample_id)}")
        print(f"  Bytes: {sample_id.encode('utf-8')}")
        print(f"  Length: {len(sample_id)}")

if __name__ == "__main__":
    debug_applicationid_matching()
```

### Проверочный скрипт для исправления:

```python
# hackathon/validate_fix.py
def validate_data_pipeline():
    """Проверка всего пайплайна после исправления"""
    
    try:
        # Шаг 1: Загрузка данных
        processor = RealFeaturesProcessor("data")
        
        # Шаг 2: Извлечение признаков  
        X, y = processor.combine_all_features()
        
        if X.empty or y.empty:
            print("❌ Данные все еще пустые")
            return False
            
        print(f"✅ Данные объединены: {X.shape[0]} записей, {X.shape[1]} признаков")
        print(f"✅ Целевые метки: {len(y)} записей")
        print(f"✅ Распределение классов: {y.value_counts().to_dict()}")
        
        # Шаг 3: Проверка качества признаков
        print(f"✅ Пропущенные значения: {X.isnull().sum().sum()}")
        print(f"✅ Бесконечные значения: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")
        
        # Шаг 4: Пробное обучение
        from src.model_trainer import ModelTrainer
        trainer = ModelTrainer("models")
        
        model, metrics = trainer.train_model(X, y)
        print(f"✅ Модель обучена, AUC-ROC: {metrics.get('auc_roc', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в пайплайне: {e}")
        return False

if __name__ == "__main__":
    success = validate_data_pipeline()
    print("🎯 Результат:", "УСПЕХ" if success else "НЕУДАЧА")
```

## ⚡ Оптимизация производительности

### Текущие узкие места:

1. **Загрузка данных**: Parquet файлы читаются последовательно
2. **Извлечение признаков**: Аудио обработка для 4,218 файлов
3. **Объединение данных**: Множественные merge операции
4. **LLM генерация**: Последовательная обработка каждой записи

### Планы оптимизации:

```python
# Параллельная обработка аудио
from concurrent.futures import ProcessPoolExecutor

def parallel_audio_processing(audio_files, n_workers=4):
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        features = list(executor.map(extract_audio_features, audio_files))
    return features

# Батчевая LLM генерация  
def batch_llm_analysis(predictions, batch_size=10):
    for i in range(0, len(predictions), batch_size):
        batch = predictions[i:i+batch_size]
        yield process_llm_batch(batch)

# Кэширование промежуточных результатов
import joblib

@joblib.memory.cache
def cached_feature_extraction(data_hash):
    # Кэшируем извлеченные признаки
    pass
```

## 📚 Дополнительная документация

### Связанные файлы:
- `LOCAL_LLM_SETUP.md` - Установка и настройка Ollama
- `FINAL_SETUP.md` - Полное руководство по установке
- `DATA_LINKING_GUIDE.md` - Руководство по связыванию данных
- `QUICK_START.md` - Быстрый старт

### Логи и отладка:
- Все компоненты используют Python `logging` модуль
- Уровень логирования: `INFO` для основных операций, `DEBUG` для детальной диагностики
- Критические ошибки логируются как `ERROR`

### Тестирование:
- Модульные тесты для каждого компонента
- Интеграционные тесты для пайплайна
- Проверки качества данных
- Валидация модели

---

**📊 Статус**: Система готова к исправлению критической проблемы с данными. После устранения проблемы объединения APPLICATIONID, все остальные компоненты готовы к работе.

**🎯 Цель**: Получить рабочий датасет из 18,386 записей с признаками и метками для обучения антифрод модели.