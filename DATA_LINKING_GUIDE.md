# 🔗 Руководство по связыванию данных для антифрод системы

## 📋 Обзор структуры данных

Система работает с тремя основными источниками данных, которые необходимо правильно связать для эффективной работы антифрода:

### 1. Amplitude данные (поведенческие метрики)
```
data/amplitude/
├── train_amplitude_chunk_*.parquet  # Технические данные звонков (чанки)
├── train_app_data.parquet          # Справочные данные по заявкам  
└── train_target_data.parquet       # Целевые метки (fraud/not_fraud)
```

### 2. Аудиофайлы звонков
```
data/audiofiles/
└── YYYYMMDDHHMMSS_CallID_Phone,_Code1,_Code2.wav
    Пример: 20241130170025_503121_77014990040,_500217,_500218.wav
```

### 3. Сводная таблица связей (svod.csv)
```
data/svod.csv  # Ключевая таблица для связывания всех данных
```

## 🔑 Основные ключи связывания

### Главный ключ: **APPLICATIONID**
- **Назначение**: Уникальный идентификатор заявки клиента
- **Формат**: Строковый (например: "Д\286\012039196")
- **Источники**: 
  - `svod.csv` → колонка "APPLICATIONID"
  - `train_app_data.parquet` → может содержать как "APPLICATIONID" или схожие поля
  - `train_target_data.parquet` → ключ для целевых меток

### Вспомогательные ключи:

#### 1. Call ID (из имени аудиофайла)
- **Назначение**: Идентификатор звонка/записи
- **Формат**: Числовой (например: "503121")
- **Извлечение**: Второй элемент в имени файла после разделения по "_"
- **Пример**: `20241130170025_503121_77014990040.wav` → Call ID = "503121"

#### 2. Phone Number
- **Назначение**: Номер телефона клиента
- **Формат**: Строковый (например: "77014990040")
- **Источники**: 
  - Имя аудиофайла (третий элемент)
  - `svod.csv` → "Абонент 2 прив2" или "Абонент 2"

#### 3. Filename
- **Назначение**: Прямая связь между svod.csv и аудиофайлами
- **Формат**: Полное имя файла
- **Источник**: `svod.csv` → колонка "Файлы"

## 📊 Структура svod.csv (основная таблица связей)

### Ключевые колонки для связывания:
```csv
ИИН,APPLICATIONID,Дата заявки,Абонент 2 прив2,Дата и время звонка,Дата звонка,Длительность фонограммы,Абонент 1,Направление вызова,Абонент 2,Состояние аудиозаписи,Фонограмма прослушана,Важная фонограмма,Помечена на удаление,Группа,Ключевые слова,Имя станции записи,Комментарий,Подразделение,Имя канала записи,Файлы,Дата время инцидента
```

### Критически важные колонки:
- **APPLICATIONID** - главный ключ связи
- **Файлы** - имя аудиофайла для прямой связи
- **Абонент 2 прив2** - номер телефона для доп. проверки
- **ИИН** - идентификатор клиента
- **Дата и время звонка** - временная метка для валидации

## 🔗 Алгоритм связывания данных

### Этап 1: Парсинг аудиофайлов
```python
def parse_audio_filename(filename):
    # Пример: "20241130170025_503121_77014990040,_500217,_500218.wav"
    parts = filename.replace('.wav', '').split('_')
    
    return {
        'datetime': datetime.strptime(parts[0], '%Y%m%d%H%M%S'),
        'call_id': parts[1],                    # 503121
        'phone': parts[2].split(',')[0],        # 77014990040
        'codes': parts[2:],                     # [500217, 500218]
        'original_filename': filename
    }
```

### Этап 2: Связывание через svod.csv
```python
def link_audio_to_applicationid(audio_metadata, svod_data):
    # Метод 1: Прямая связь по имени файла
    file_mapping = svod_data.set_index('Файлы')['APPLICATIONID'].to_dict()
    audio_metadata['applicationid'] = audio_metadata['original_filename'].map(file_mapping)
    
    # Метод 2: Связь по номеру телефона (резервный)
    phone_mapping = svod_data.set_index('Абонент 2 прив2')['APPLICATIONID'].to_dict()
    missing_apps = audio_metadata['applicationid'].isna()
    audio_metadata.loc[missing_apps, 'applicationid'] = \
        audio_metadata.loc[missing_apps, 'phone'].map(phone_mapping)
    
    return audio_metadata
```

### Этап 3: Объединение с amplitude данными
```python
def merge_all_data(amplitude_data, target_data, audio_metadata):
    # Основное объединение по APPLICATIONID
    merged_data = amplitude_data.merge(
        target_data, 
        on='APPLICATIONID', 
        how='left'
    )
    
    # Добавление аудио признаков
    merged_data = merged_data.merge(
        audio_metadata,
        on='applicationid',
        how='left'
    )
    
    return merged_data
```

## 🎯 Практическая реализация

### 1. Валидация связей
```python
def validate_data_links(merged_data):
    total_records = len(merged_data)
    
    # Проверка наличия APPLICATIONID
    app_id_coverage = merged_data['APPLICATIONID'].notna().sum()
    print(f"APPLICATIONID покрытие: {app_id_coverage}/{total_records} ({app_id_coverage/total_records:.1%})")
    
    # Проверка связи с аудио
    audio_coverage = merged_data['original_filename'].notna().sum()
    print(f"Аудио покрытие: {audio_coverage}/{total_records} ({audio_coverage/total_records:.1%})")
    
    # Проверка целевых меток
    target_coverage = merged_data['is_fraud'].notna().sum()
    print(f"Целевые метки: {target_coverage}/{total_records} ({target_coverage/total_records:.1%})")
```

### 2. Обработка конфликтов связывания
```python
def resolve_linking_conflicts(merged_data):
    # Дубликаты APPLICATIONID
    duplicates = merged_data.duplicated(subset=['APPLICATIONID'], keep=False)
    if duplicates.any():
        print(f"⚠️ Найдено дубликатов APPLICATIONID: {duplicates.sum()}")
        # Стратегия: берем последнюю запись по времени
        merged_data = merged_data.sort_values('datetime').drop_duplicates(
            subset=['APPLICATIONID'], 
            keep='last'
        )
    
    # Аудиофайлы без APPLICATIONID
    orphaned_audio = merged_data['APPLICATIONID'].isna() & merged_data['original_filename'].notna()
    if orphaned_audio.any():
        print(f"⚠️ Аудиофайлы без APPLICATIONID: {orphaned_audio.sum()}")
    
    return merged_data
```

## 📈 Качество связывания - метрики

### Ключевые показатели:
- **Полнота связывания**: % записей с APPLICATIONID
- **Аудио покрытие**: % записей с аудиофайлами  
- **Целевое покрытие**: % записей с метками fraud/not_fraud
- **Временная согласованность**: соответствие временных меток

### Целевые значения:
- ✅ **Отлично**: >90% полное связывание
- ⚠️ **Приемлемо**: 70-90% связывание
- ❌ **Критично**: <70% связывание

## 🛠️ Инструменты диагностики

### 1. Проверка структуры данных
```bash
# Проверка наличия всех файлов
python scripts/validate_data_structure.py

# Анализ качества связывания  
python scripts/analyze_data_links.py

# Генерация отчета по данным
python scripts/generate_data_report.py
```

### 2. Автоматическое исправление
```python
# Исправление кодировок в svod.csv
python scripts/fix_encoding.py

# Поиск и восстановление потерянных связей
python scripts/recover_missing_links.py

# Валидация временных меток
python scripts/validate_timestamps.py
```

## 🚨 Частые проблемы и решения

### Проблема 1: Несоответствие имен файлов
**Симптом**: Аудиофайлы не связываются с APPLICATIONID
**Причина**: Различия в именах файлов между svod.csv и директорией audiofiles
**Решение**: 
```python
# Нормализация имен файлов
def normalize_filename(filename):
    return filename.strip().replace('\\', '/').split('/')[-1]
```

### Проблема 2: Кодировка svod.csv
**Симптом**: Ошибки при чтении svod.csv
**Причина**: Неправильная кодировка (обычно cp1251 вместо utf-8)
**Решение**:
```python
# Автоматическое определение кодировки
for encoding in ['utf-8', 'cp1251', 'windows-1251']:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        break
    except UnicodeDecodeError:
        continue
```

### Проблема 3: Дубликаты APPLICATIONID
**Симптом**: Один APPLICATIONID связан с несколькими записями
**Причина**: Несколько звонков по одной заявке
**Решение**: Агрегация или выбор репрезентативной записи

### Проблема 4: Отсутствующие целевые метки
**Симптом**: Нет меток fraud/not_fraud для обучения
**Причина**: Несоответствие ключей между данными и метками
**Решение**: Расширенный поиск по альтернативным ключам

## 📋 Чек-лист для проверки связывания

### Перед обучением модели:
- [ ] svod.csv загружается без ошибок кодировки
- [ ] Все аудиофайлы найдены и доступны
- [ ] >80% аудиофайлов связаны с APPLICATIONID
- [ ] Amplitude данные содержат APPLICATIONID
- [ ] Целевые метки соответствуют APPLICATIONID
- [ ] Временные метки корректны и согласованы
- [ ] Нет критических дубликатов данных

### После объединения данных:
- [ ] Итоговый датасет содержит все типы признаков
- [ ] Распределение классов (fraud/not_fraud) сбалансировано
- [ ] Нет утечек данных между train/test
- [ ] Качество признаков валидировано

## 🎯 Рекомендации по оптимизации

### 1. Повышение качества связывания:
- Использовать множественные ключи (APPLICATIONID + phone + timestamp)
- Реализовать fuzzy matching для имен файлов
- Добавить валидацию временных интервалов

### 2. Обработка edge cases:
- Один APPLICATIONID → множество аудиозаписей (выбор лучшей)
- Аудиозапись без APPLICATIONID (попытка восстановления)
- Разные форматы идентификаторов (нормализация)

### 3. Мониторинг качества:
- Автоматические проверки при загрузке новых данных
- Алерты при снижении процента связывания
- Логирование всех конфликтов и их разрешений

---

## 💡 Итоговая схема связывания

```
[svod.csv] ←→ [APPLICATIONID] ←→ [amplitude_data.parquet]
     ↕                                      ↕
[Файлы] ←→ [audiofiles/*.wav]    [target_data.parquet]
     ↓                                      ↓
[Audio Features] ←→ [FINAL DATASET] ←→ [Fraud Labels]
```

**Результат**: Единый датасет с поведенческими метриками amplitude, аудио-признаками и целевыми метками для обучения антифрод модели.