# 🔧 Руководство по устранению критических проблем

> **Статус**: Активное исправление проблемы с объединением данных по APPLICATIONID

## 🚨 Критическая проблема: Нулевое пересечение данных

### Симптомы
```
INFO:real_features_processor:✅ Извлечено amplitude признаков: (13439, 116)
INFO:real_features_processor:✅ Извлечено признаков заявок: (18386, 64)
INFO:real_features_processor:🎯 Целевых меток: 0
ERROR:__main__:❌ Критическая ошибка при обучении: ❌ Не удалось подготовить данные для обучения
```

**Диагноз**: Данные загружаются корректно, но при объединении по `APPLICATIONID` получается пустой результат.

## 🔍 Пошаговая диагностика

### Шаг 1: Запуск диагностического скрипта

Создайте файл `hackathon/debug_merge.py`:

```python
#!/usr/bin/env python3
"""Диагностика проблемы с объединением данных"""

import pandas as pd
from pathlib import Path
from src.real_data_loader import RealDataLoader

def analyze_applicationid_keys():
    """Детальный анализ ключей APPLICATIONID"""
    
    print("🔍 АНАЛИЗ APPLICATIONID КЛЮЧЕЙ")
    print("=" * 50)
    
    # Загружаем данные
    loader = RealDataLoader("data")
    
    # Получаем сырые данные
    print("📊 Загрузка данных...")
    amplitude_data = loader.load_amplitude_chunks()
    app_data = loader.load_app_data()
    target_data = loader.load_target_data()
    
    print(f"✅ Amplitude: {len(amplitude_data)} записей")
    print(f"✅ App: {len(app_data)} записей")
    print(f"✅ Target: {len(target_data)} записей")
    
    # Анализируем ключи
    print("\n🔑 АНАЛИЗ КЛЮЧЕЙ")
    print("-" * 30)
    
    # Получаем уникальные ID
    amp_ids = amplitude_data['applicationid'].dropna().astype(str).unique()
    app_ids = app_data['APPLICATIONID'].dropna().astype(str).unique()
    target_ids = target_data['APPLICATIONID'].dropna().astype(str).unique()
    
    print(f"📈 Amplitude уникальных ID: {len(amp_ids)}")
    print(f"📱 App уникальных ID: {len(app_ids)}")
    print(f"🎯 Target уникальных ID: {len(target_ids)}")
    
    # Показываем примеры
    print("\n📋 ПРИМЕРЫ КЛЮЧЕЙ")
    print("-" * 20)
    print("Amplitude примеры:")
    for i, app_id in enumerate(amp_ids[:3]):
        print(f"  {i+1}. {repr(app_id)} (длина: {len(app_id)})")
        
    print("App примеры:")
    for i, app_id in enumerate(app_ids[:3]):
        print(f"  {i+1}. {repr(app_id)} (длина: {len(app_id)})")
        
    print("Target примеры:")
    for i, app_id in enumerate(target_ids[:3]):
        print(f"  {i+1}. {repr(app_id)} (длина: {len(app_id)})")
    
    # Проверяем пересечения
    print("\n🔄 АНАЛИЗ ПЕРЕСЕЧЕНИЙ")
    print("-" * 25)
    
    amp_set = set(amp_ids)
    app_set = set(app_ids)
    target_set = set(target_ids)
    
    amp_app = amp_set.intersection(app_set)
    app_target = app_set.intersection(target_set)
    amp_target = amp_set.intersection(target_set)
    all_three = amp_set.intersection(app_set).intersection(target_set)
    
    print(f"📊 Amplitude ∩ App: {len(amp_app)} ID")
    print(f"📊 App ∩ Target: {len(app_target)} ID")
    print(f"📊 Amplitude ∩ Target: {len(amp_target)} ID")
    print(f"📊 Все три источника: {len(all_three)} ID")
    
    if len(all_three) > 0:
        print("\n✅ НАЙДЕНЫ ОБЩИЕ ID:")
        for app_id in list(all_three)[:5]:
            print(f"  - {repr(app_id)}")
    else:
        print("\n❌ НЕТ ОБЩИХ ID!")
        
        # Ищем похожие ID
        print("\n🔍 ПОИСК ПОХОЖИХ ID:")
        sample_amp = amp_ids[0] if len(amp_ids) > 0 else ""
        sample_app = app_ids[0] if len(app_ids) > 0 else ""
        sample_target = target_ids[0] if len(target_ids) > 0 else ""
        
        print(f"Amplitude sample: {repr(sample_amp)}")
        print(f"App sample:       {repr(sample_app)}")  
        print(f"Target sample:    {repr(sample_target)}")
        
        # Анализируем символы
        print("\n🔤 АНАЛИЗ СИМВОЛОВ:")
        for name, sample in [("Amplitude", sample_amp), ("App", sample_app), ("Target", sample_target)]:
            if sample:
                print(f"{name}:")
                print(f"  Raw: {repr(sample)}")
                print(f"  Bytes: {sample.encode('utf-8')}")
                print(f"  Hex: {sample.encode('utf-8').hex()}")
    
    return len(all_three) > 0

def test_normalization_approaches():
    """Тестируем разные подходы к нормализации"""
    
    print("\n🧪 ТЕСТИРОВАНИЕ НОРМАЛИЗАЦИИ")
    print("=" * 40)
    
    # Примеры проблемных ID
    test_ids = [
        "Д\\286\\011639474",
        "Д\286\011639474", 
        "Д\\u00be\\u0009639474"
    ]
    
    for test_id in test_ids:
        print(f"\nТестируем: {repr(test_id)}")
        
        # Подход 1: unicode_escape
        try:
            normalized1 = test_id.encode('utf-8').decode('unicode_escape')
            print(f"  unicode_escape: {repr(normalized1)}")
        except Exception as e:
            print(f"  unicode_escape: ОШИБКА - {e}")
            
        # Подход 2: raw string
        try:
            normalized2 = test_id.replace('\\', '')
            print(f"  remove_backslash: {repr(normalized2)}")
        except Exception as e:
            print(f"  remove_backslash: ОШИБКА - {e}")
            
        # Подход 3: latin1 decode
        try:
            normalized3 = test_id.encode('latin1').decode('unicode_escape')
            print(f"  latin1_decode: {repr(normalized3)}")
        except Exception as e:
            print(f"  latin1_decode: ОШИБКА - {e}")

if __name__ == "__main__":
    print("🚀 НАЧАЛО ДИАГНОСТИКИ")
    print("=" * 50)
    
    success = analyze_applicationid_keys()
    
    if not success:
        test_normalization_approaches()
    
    print("\n🏁 ДИАГНОСТИКА ЗАВЕРШЕНА")
    if success:
        print("✅ Найдены общие ключи - проблема может быть в другом месте")
    else:
        print("❌ Нет пересечения ключей - нужна нормализация")
```

Запустите диагностику:
```bash
cd hackathon
python debug_merge.py
```

### Шаг 2: Анализ результатов диагностики

**Ожидаемые результаты:**

#### Сценарий A: Нет пересечения ключей
```
❌ НЕТ ОБЩИХ ID!
🔍 ПОИСК ПОХОЖИХ ID:
Amplitude sample: 'Д\\286\\011221568'
App sample:       'Д\\286\\011639474'
Target sample:    'Д\\286\\011639474'
```

**→ Решение**: Нужна нормализация escape-символов

#### Сценарий B: Есть общие ключи, но merge не работает
```
✅ НАЙДЕНЫ ОБЩИЕ ID:
  - 'Д\\286\\011639474'
📊 Все три источника: 1500 ID
```

**→ Решение**: Проблема в логике merge, проверить регистр столбцов

### Шаг 3: Исправление нормализации ключей

Если диагностика показала проблему с escape-символами, примените исправление:

```python
# В файле src/real_features_processor.py
# Замените функцию normalize_key на улучшенную версию:

def normalize_key(key):
    """Улучшенная нормализация APPLICATIONID"""
    if pd.isna(key) or key == '':
        return None
        
    # Приводим к строке
    key_str = str(key).strip()
    
    # ИСПРАВЛЕНИЕ: Обрабатываем escape последовательности
    # Проблема: 'Д\\286\\011639474' нужно преобразовать в читаемый вид
    
    try:
        # Метод 1: Интерпретировать как unicode escape
        if '\\' in key_str:
            # Заменяем двойные слеши на одинарные для правильной интерпретации
            key_str = key_str.replace('\\\\', '\\')
            key_str = key_str.encode('utf-8', errors='ignore').decode('unicode_escape', errors='ignore')
    except Exception:
        # Если не получается - используем как есть
        pass
    
    # Приводим к верхнему регистру для единообразия
    return key_str.upper().strip()
```

### Шаг 4: Альтернативное исправление merge логики

Если проблема в самом merge, замените логику объединения:

```python
# В файле src/real_features_processor.py
# В методе combine_all_features(), замените блок merge:

# СТАРЫЙ КОД (проблемный):
final_data = pd.merge(
    combined_features, target_data,
    left_on='applicationid', right_on=merge_col,
    how='inner'
)

# НОВЫЙ КОД (исправленный):
# Нормализуем ключи в обеих таблицах
combined_features['app_id_normalized'] = combined_features['applicationid'].apply(normalize_key)
target_data['app_id_normalized'] = target_data[merge_col].apply(normalize_key)

# Убираем None значения
combined_features = combined_features[combined_features['app_id_normalized'].notna()]
target_data = target_data[target_data['app_id_normalized'].notna()]

# Выполняем merge по нормализованным ключам
final_data = pd.merge(
    combined_features, target_data,
    left_on='app_id_normalized', right_on='app_id_normalized',
    how='inner'
)

# Очищаем вспомогательные колонки
final_data = final_data.drop(columns=['app_id_normalized'], errors='ignore')

logger.info(f"🔍 Записей после нормализованного merge: {len(final_data)}")
```

### Шаг 5: Валидация исправления

Создайте проверочный скрипт `hackathon/validate_fix.py`:

```python
#!/usr/bin/env python3
"""Проверка исправления проблемы"""

from src.real_features_processor import RealFeaturesProcessor
import pandas as pd

def validate_merge_fix():
    """Проверяем что исправление работает"""
    
    print("🧪 ВАЛИДАЦИЯ ИСПРАВЛЕНИЯ")
    print("=" * 40)
    
    try:
        # Инициализируем процессор
        processor = RealFeaturesProcessor("data")
        
        # Пробуем объединить данные
        print("📊 Выполняем combine_all_features()...")
        X, y = processor.combine_all_features()
        
        # Проверяем результат
        print(f"📈 Форма признаков X: {X.shape}")
        print(f"🎯 Размер целевых меток y: {len(y)}")
        
        if X.empty or y.empty:
            print("❌ ИСПРАВЛЕНИЕ НЕ СРАБОТАЛО - данные все еще пустые")
            return False
            
        # Проверяем распределение классов
        class_dist = y.value_counts()
        print(f"📊 Распределение классов: {class_dist.to_dict()}")
        
        fraud_rate = y.mean()
        print(f"📈 Доля мошенничества: {fraud_rate:.2%}")
        
        # Проверяем качество признаков
        null_counts = X.isnull().sum().sum()
        inf_counts = X.select_dtypes(include=['number']).apply(lambda x: x.isin([float('inf'), float('-inf')]).sum()).sum()
        
        print(f"🔍 Пропущенные значения: {null_counts}")
        print(f"🔍 Бесконечные значения: {inf_counts}")
        
        # Показываем примеры признаков
        print(f"📋 Первые 5 колонок признаков: {list(X.columns[:5])}")
        print(f"📋 Последние 5 колонок признаков: {list(X.columns[-5:])}")
        
        print("\n✅ ИСПРАВЛЕНИЕ УСПЕШНО!")
        print(f"✅ Готово {len(y)} записей для обучения модели")
        
        return True
        
    except Exception as e:
        print(f"❌ ОШИБКА ПРИ ВАЛИДАЦИИ: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_training():
    """Тестируем обучение модели на исправленных данных"""
    
    print("\n🤖 ТЕСТ ОБУЧЕНИЯ МОДЕЛИ")
    print("=" * 30)
    
    try:
        # Получаем данные
        processor = RealFeaturesProcessor("data")
        X, y = processor.combine_all_features()
        
        if X.empty or y.empty:
            print("❌ Нельзя обучить модель - нет данных")
            return False
        
        # Пробуем обучить простую модель
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, roc_auc_score
        
        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Обучающая выборка: {X_train.shape}")
        print(f"📊 Тестовая выборка: {X_test.shape}")
        
        # Обучаем модель
        print("🏋️ Обучение Random Forest...")
        model = RandomForestClassifier(
            n_estimators=50,  # Уменьшили для быстроты
            max_depth=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Оцениваем качество
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"📈 AUC-ROC: {auc_score:.3f}")
        
        print("\n📊 Отчет по классификации:")
        print(classification_report(y_test, y_pred))
        
        # Важность признаков
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 Топ-10 важных признаков:")
        print(feature_importance.head(10))
        
        print("\n✅ МОДЕЛЬ УСПЕШНО ОБУЧЕНА!")
        return True
        
    except Exception as e:
        print(f"❌ ОШИБКА ПРИ ОБУЧЕНИИ: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 НАЧАЛО ВАЛИДАЦИИ")
    print("=" * 50)
    
    # Валидируем исправление
    merge_ok = validate_merge_fix()
    
    if merge_ok:
        # Тестируем обучение
        model_ok = test_model_training()
        
        if model_ok:
            print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
            print("✅ Система готова к полноценному использованию")
        else:
            print("\n⚠️ Данные исправлены, но есть проблемы с моделью")
    else:
        print("\n❌ ИСПРАВЛЕНИЕ НЕ ПОМОГЛО")
        print("Нужна дополнительная диагностика")
    
    print("\n🏁 ВАЛИДАЦИЯ ЗАВЕРШЕНА")
```

Запустите валидацию:
```bash
python validate_fix.py
```

## 🛠️ Резервные планы исправления

### План B: Частичное совпадение ключей

Если полное совпадение не работает, попробуйте частичное:

```python
def fuzzy_merge_applicationid(df1, df2, key1, key2, threshold=0.8):
    """Нечеткое объединение по APPLICATIONID"""
    from difflib import SequenceMatcher
    
    # Получаем уникальные ключи
    keys1 = df1[key1].dropna().unique()
    keys2 = df2[key2].dropna().unique()
    
    # Ищем похожие ключи
    matches = []
    for k1 in keys1:
        for k2 in keys2:
            similarity = SequenceMatcher(None, str(k1), str(k2)).ratio()
            if similarity >= threshold:
                matches.append((k1, k2, similarity))
    
    print(f"🔍 Найдено {len(matches)} нечетких совпадений")
    
    # Создаем маппинг
    key_mapping = {k1: k2 for k1, k2, _ in matches}
    
    # Применяем маппинг
    df1_mapped = df1.copy()
    df1_mapped[key1 + '_mapped'] = df1_mapped[key1].map(key_mapping)
    
    # Выполняем merge
    result = pd.merge(
        df1_mapped, df2,
        left_on=key1 + '_mapped', right_on=key2,
        how='inner'
    )
    
    return result.drop(columns=[key1 + '_mapped'])
```

### План C: Временная привязка

Если ID не совпадают, попробуйте связать по времени:

```python
def temporal_merge(amplitude_data, app_data, time_window_hours=24):
    """Объединение по временной близости"""
    
    # Конвертируем даты
    amplitude_data['event_date'] = pd.to_datetime(amplitude_data['event_time']).dt.date
    app_data['create_date'] = pd.to_datetime(app_data['CREATE_DATE']).dt.date
    
    # Объединяем по датам с окном
    merged_records = []
    
    for _, app_row in app_data.iterrows():
        create_date = app_row['create_date']
        
        # Ищем amplitude записи в окне времени
        date_range = pd.date_range(
            start=create_date - pd.Timedelta(days=1),
            end=create_date + pd.Timedelta(days=1)
        ).date
        
        matching_amp = amplitude_data[
            amplitude_data['event_date'].isin(date_range)
        ]
        
        if not matching_amp.empty:
            # Берем первую подходящую запись
            merged_record = {**app_row.to_dict(), **matching_amp.iloc[0].to_dict()}
            merged_records.append(merged_record)
    
    return pd.DataFrame(merged_records)
```

### План D: Создание синтетических меток

Для тестирования системы можно создать синтетические целевые метки:

```python
def create_synthetic_targets(features_df, fraud_rate=0.02):
    """Создаем синтетические целевые метки для тестирования"""
    import numpy as np
    
    n_samples = len(features_df)
    n_fraud = int(n_samples * fraud_rate)
    
    # Создаем случайные метки с заданной долей мошенничества
    targets = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, size=n_fraud, replace=False)
    targets[fraud_indices] = 1
    
    print(f"🎯 Создано {n_fraud} синтетических меток мошенничества из {n_samples}")
    
    return pd.Series(targets, index=features_df.index)
```

## 📋 Чек-лист исправления

### Перед исправлением:
- [ ] Создан backup исходных файлов
- [ ] Запущена диагностика `debug_merge.py`
- [ ] Определена конкретная причина проблемы
- [ ] Выбран подходящий план исправления

### Во время исправления:
- [ ] Применены изменения в `normalize_key()` функции
- [ ] Обновлена логика merge в `combine_all_features()`
- [ ] Добавлена дополнительная отладочная информация
- [ ] Протестированы промежуточные результаты

### После исправления:
- [ ] Запущена валидация `validate_fix.py`
- [ ] Проверено что `X.shape[0] > 0` и `len(y) > 0`
- [ ] Проверено распределение классов в целевых метках
- [ ] Успешно обучена тестовая модель
- [ ] Запущен полный пайплайн `train_real_data.py`

## 🔔 Контрольные точки

### Успешное исправление:
```
✅ Данные объединены: (15000, 200) записей
✅ Целевые метки: 15000 записей  
✅ Распределение классов: {0: 14700, 1: 300}
✅ Доля мошенничества: 2.00%
✅ Модель обучена, AUC-ROC: 0.857
```

### Неполное исправление:
```
⚠️ Данные объединены: (500, 200) записей  ← Слишком мало
⚠️ Целевые метки: 500 записей
⚠️ Доля мошенничества: 0.20%  ← Слишком низкая доля
```

### Неудачное исправление:
```
❌ Форма признаков X: (0, 0)
❌ Размер целевых меток y: 0
❌ Данные все еще пустые
```

## 📞 Поддержка и помощь

### Если ничего не помогает:

1. **Создайте подробный отчет**:
   ```bash
   python debug_merge.py > debug_report.txt 2>&1
   python validate_fix.py > validation_report.txt 2>&1
   ```

2. **Соберите образцы данных**:
   ```python
   # Создайте samples.py
   from src.real_data_loader import RealDataLoader
   
   loader = RealDataLoader("data")
   
   # Сохраните образцы
   amplitude_sample = loader.load_amplitude_chunks().head(10)
   app_sample = loader.load_app_data().head(10)
   target_sample = loader.load_target_data().head(10)
   
   amplitude_sample.to_csv("amplitude_sample.csv")
   app_sample.to_csv("app_sample.csv")  
   target_sample.to_csv("target_sample.csv")
   ```

3. **Проверьте системные требования**:
   - Python >= 3.8
   - pandas >= 1.5.0
   - Достаточно RAM (8GB+)
   - Корректные права доступа к файлам

---

**🎯 Цель**: Получить рабочий датасет с объединенными признаками и целевыми метками для успешного обучения антифрод модели.

**✅ Критерий успеха**: `X.shape[0] > 10000` и `len(y) > 10000` с осмысленным распределением классов.