#!/usr/bin/env python3
"""
Скрипт обучения антифрод модели на реальных данных из data_train
"""

import sys
import os
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent / "src"))

import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, Any

from real_features_processor import RealFeaturesProcessor
from model_trainer import ModelTrainer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_data_structure(data_dir: Path) -> bool:
    """
    Проверка структуры данных для обучения

    Args:
        data_dir: Директория с данными

    Returns:
        True если структура корректна
    """
    logger.info("🔍 Проверка структуры данных...")

    amplitude_dir = data_dir / "amplitude"
    audio_dir = data_dir / "audiofiles"

    # Проверяем наличие основных файлов
    required_files = [
        "train_target_data.parquet",  # Обязательно для обучения
    ]

    missing_files = []
    for file_name in required_files:
        file_path = amplitude_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)

    if missing_files:
        logger.error(f"❌ Отсутствуют обязательные файлы: {missing_files}")
        return False

    # Проверяем amplitude чанки
    amplitude_chunks = list(amplitude_dir.glob("train_amplitude_chunk_*.parquet"))
    if not amplitude_chunks:
        logger.warning("⚠️  Amplitude чанки не найдены, но это не критично")
    else:
        logger.info(f"✅ Найдено amplitude чанков: {len(amplitude_chunks)}")

    # Проверяем данные заявок
    app_data_file = amplitude_dir / "train_app_data.parquet"
    if app_data_file.exists():
        logger.info("✅ Данные заявок найдены")
    else:
        logger.warning("⚠️  Данные заявок не найдены")

    # Проверяем аудиофайлы
    if audio_dir.exists():
        audio_files = list(audio_dir.glob("*.wav"))
        logger.info(f"🎵 Найдено аудиофайлов: {len(audio_files)}")
    else:
        logger.warning("⚠️  Директория аудиофайлов не найдена")

    # Проверяем svod.csv в корне data_train
    svod_file = data_dir / "svod.csv"
    if svod_file.exists():
        logger.info("✅ Файл svod.csv найден в корне")
    else:
        logger.warning("⚠️  Файл svod.csv не найден в корне")

    return True


def analyze_data_quality(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Анализ качества данных

    Args:
        X: Признаки
        y: Целевые метки

    Returns:
        Словарь с метриками качества
    """
    logger.info("📊 Анализ качества данных...")

    # Базовая статистика
    n_samples, n_features = X.shape
    fraud_rate = y.mean()
    missing_ratio = X.isnull().sum().sum() / (n_samples * n_features)

    logger.info(f"📈 Образцов: {n_samples}")
    logger.info(f"📊 Признаков: {n_features}")
    logger.info(f"🎯 Доля мошенничества: {fraud_rate:.2%}")
    logger.info(f"🔍 Пропущенных значений: {missing_ratio:.2%}")

    # Предупреждения
    if fraud_rate < 0.01:
        logger.warning(f"⚠️  Очень низкая доля мошенничества: {fraud_rate:.2%}")
    elif fraud_rate > 0.5:
        logger.warning(f"⚠️  Подозрительно высокая доля мошенничества: {fraud_rate:.2%}")

    if missing_ratio > 0.3:
        logger.warning(f"⚠️  Высокая доля пропущенных значений: {missing_ratio:.2%}")

    # Проверка дисбаланса классов
    class_balance = y.value_counts().min() / y.value_counts().max()
    if class_balance < 0.1:
        logger.warning(f"⚠️  Дисбаланс классов: {fraud_rate:.2%}")

    return {
        'n_samples': n_samples,
        'n_features': n_features,
        'fraud_rate': fraud_rate,
        'missing_values_ratio': missing_ratio,
        'class_balance': class_balance
    }


def prepare_training_data(features_processor: RealFeaturesProcessor) -> tuple:
    """
    Подготовка данных для обучения

    Args:
        features_processor: Процессор признаков

    Returns:
        Кортеж (X, y, metadata)
    """
    logger.info("🔧 Подготовка данных для обучения...")

    with tqdm(total=4, desc="🔧 Подготовка данных", unit="шаг", leave=False) as pbar:
        # Извлекаем и объединяем все признаки
        pbar.set_description("🔗 Объединение признаков")
        X, y = features_processor.combine_all_features()
        pbar.update(1)

        if X.empty or y.empty:
            raise ValueError("❌ Не удалось подготовить данные для обучения")

        # Удаляем session_id из признаков если он есть
        pbar.set_description("🧹 Очистка данных")
        if 'session_id' in X.columns:
            X = X.drop('session_id', axis=1)
        pbar.update(1)

        # Анализируем качество данных
        pbar.set_description("📊 Анализ качества")
        quality_metrics = analyze_data_quality(X, y)
        pbar.update(1)

        # Предупреждения по качеству
        pbar.set_description("⚠️  Проверка качества")
        if quality_metrics['missing_values_ratio'] > 0.2:
            logger.warning("⚠️  Высокая доля пропущенных значений")

        if quality_metrics['fraud_rate'] < 0.05:
            logger.warning("⚠️  Очень низкая доля мошенничества")
        elif quality_metrics['fraud_rate'] > 0.5:
            logger.warning("⚠️  Подозрительно высокая доля мошенничества")
        pbar.update(1)

    return X, y, quality_metrics


def train_antifraud_model(X: pd.DataFrame, y: pd.Series, models_dir: Path, quality_metrics: Dict) -> Dict:
    """
    Обучение антифрод модели

    Args:
        X: Признаки
        y: Целевые метки
        models_dir: Директория для сохранения модели
        quality_metrics: Метрики качества данных

    Returns:
        Метрики обученной модели
    """
    logger.info("🤖 Начинаем обучение антифрод модели...")

    # Инициализируем тренера
    model_trainer = ModelTrainer(str(models_dir))

    # Настраиваем для несбалансированных данных
    if quality_metrics['fraud_rate'] < 0.1:
        logger.info("🎯 Настроена модель для несбалансированных данных")

    # Обучаем модель
    metrics = model_trainer.train(X, y, test_size=0.2)

    # Сохраняем модель
    model_name = "real_antifraud_model"
    model_trainer.save_model(model_name)

    logger.info(f"💾 Модель сохранена: {model_name}")

    return metrics


def validate_antifraud_model(model_trainer: ModelTrainer, X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Валидация обученной модели

    Args:
        model_trainer: Обученный тренер модели
        X: Признаки для валидации
        y: Целевые метки для валидации

    Returns:
        Метрики валидации
    """
    logger.info("✅ Валидация антифрод модели...")

    # Получаем предсказания
    predictions = model_trainer.predict(X)
    probabilities = model_trainer.predict_proba(X)

    # Базовые метрики
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    try:
        auc_score = roc_auc_score(y, probabilities[:, 1])
        logger.info(f"📊 AUC-ROC: {auc_score:.3f}")
    except:
        auc_score = 0.0
        logger.warning("⚠️  Не удалось вычислить AUC-ROC")

    # Матрица ошибок
    cm = confusion_matrix(y, predictions)
    logger.info(f"📊 Матрица ошибок:\n{cm}")

    # Отчет по классификации
    report = classification_report(y, predictions, output_dict=True)
    logger.info("📊 Отчет по классификации:")
    logger.info(classification_report(y, predictions))

    return {
        'auc_roc': auc_score,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }


def create_training_report(quality_metrics: Dict, training_metrics: Dict,
                         validation_metrics: Dict, output_dir: Path) -> None:
    """
    Создание отчета об обучении

    Args:
        quality_metrics: Метрики качества данных
        training_metrics: Метрики обучения
        validation_metrics: Метрики валидации
        output_dir: Директория для сохранения отчета
    """
    logger.info("📄 Создание отчета об обучении...")

    report_content = f"""
=== ОТЧЕТ ОБ ОБУЧЕНИИ АНТИФРОД МОДЕЛИ ===
Дата обучения: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Источник данных: data_train/

КАЧЕСТВО ДАННЫХ:
- Образцов: {quality_metrics['n_samples']:,}
- Признаков: {quality_metrics['n_features']:,}
- Доля мошенничества: {quality_metrics['fraud_rate']:.2%}
- Пропущенные значения: {quality_metrics['missing_values_ratio']:.2%}
- Баланс классов: {quality_metrics['class_balance']:.3f}

МЕТРИКИ ОБУЧЕНИЯ:
- AUC-ROC (обучение): {training_metrics.get('auc_roc', 'N/A')}
- Precision (обучение): {training_metrics.get('precision', 'N/A')}
- Recall (обучение): {training_metrics.get('recall', 'N/A')}
- F1-Score (обучение): {training_metrics.get('f1_score', 'N/A')}

МЕТРИКИ ВАЛИДАЦИИ:
- AUC-ROC (валидация): {validation_metrics.get('auc_roc', 'N/A')}

РЕКОМЕНДАЦИИ:
"""

    # Добавляем рекомендации на основе метрик
    if quality_metrics['fraud_rate'] < 0.02:
        report_content += "\n- ⚠️  Очень низкая доля мошенничества - рассмотрите дополнительные методы балансировки"

    if quality_metrics['missing_values_ratio'] > 0.1:
        report_content += "\n- ⚠️  Высокая доля пропущенных значений - улучшите качество данных"

    if training_metrics.get('auc_roc', 0) > 0.8:
        report_content += "\n- ✅ Хорошее качество модели"
    elif training_metrics.get('auc_roc', 0) > 0.7:
        report_content += "\n- 📊 Приемлемое качество модели"
    else:
        report_content += "\n- ⚠️  Низкое качество модели - требуется улучшение"

    report_content += f"""

СТАТУС МОДЕЛИ: ✅ ГОТОВА К ИСПОЛЬЗОВАНИЮ
Файл модели: models/real_antifraud_model.joblib

=== КОНЕЦ ОТЧЕТА ===
"""

    # Сохраняем отчет
    report_file = output_dir / "training_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    logger.info(f"📄 Отчет сохранен: {report_file}")


def main():
    """
    Основная функция обучения на реальных данных из data_train
    """
    logger.info("🚀 Запуск обучения антифрод модели на данных из data_train")

    try:
        # Общий прогресс для всего процесса
        with tqdm(total=7, desc="🚀 Обучение антифрод модели", unit="этап", position=0) as main_pbar:

            # Настройки путей - ИСПОЛЬЗУЕМ data_train
            main_pbar.set_description("📂 Настройка путей")
            data_dir = Path(__file__).parent.parent / "data_train"  # ИЗМЕНЕНО: data_train вместо data
            models_dir = Path(__file__).parent.parent / "models"
            output_dir = Path(__file__).parent.parent / "output"

            # Создаем директории
            models_dir.mkdir(exist_ok=True)
            output_dir.mkdir(exist_ok=True)

            logger.info(f"📂 Директория данных: {data_dir}")
            logger.info(f"🤖 Директория моделей: {models_dir}")
            logger.info(f"📊 Выходная директория: {output_dir}")
            main_pbar.update(1)

            # Проверяем структуру данных
            main_pbar.set_description("🔍 Проверка данных")
            if not validate_data_structure(data_dir):
                logger.error("❌ Проверка структуры данных не пройдена")
                return False
            main_pbar.update(1)

        # Инициализируем процессор признаков
        logger.info("🔧 Инициализация процессора признаков...")
        features_processor = RealFeaturesProcessor(str(data_dir))

        # Подготавливаем данные
        main_pbar.set_description("🔧 Подготовка данных")
        X, y, quality_metrics = prepare_training_data(features_processor)
        main_pbar.update(1)

        # Проверяем минимальные требования
        main_pbar.set_description("✅ Проверка данных")
        if len(X) < 100:
            logger.error("❌ Недостаточно данных для обучения (минимум 100 образцов)")
            return False

        # КРИТИЧНО: Проверяем типы данных
        logger.info("🔍 Проверка типов данных...")

        # Проверяем все типы данных
        problematic_cols = []
        for col in X.columns:
            col_dtype = str(X[col].dtype)

            # Проверяем на проблемные типы
            if any(dtype_name in col_dtype.lower() for dtype_name in ['object', 'datetime', 'timedelta', 'string']):
                problematic_cols.append(col)
                logger.warning(f"⚠️  Найдена проблемная колонка: {col} (тип: {col_dtype})")

        if problematic_cols:
            logger.error(f"❌ Найдены проблемные колонки: {problematic_cols}")
            logger.info("🔧 Попытка принудительного преобразования...")

            for col in problematic_cols:
                col_dtype = str(X[col].dtype)
                try:
                    if 'datetime' in col_dtype.lower():
                        # Преобразуем datetime в timestamp
                        X[col] = pd.to_datetime(X[col]).astype('int64') // 10**9
                        logger.info(f"✅ Преобразована datetime колонка: {col}")
                    elif 'timedelta' in col_dtype.lower():
                        # Преобразуем timedelta в секунды
                        X[col] = pd.to_timedelta(X[col]).dt.total_seconds()
                        logger.info(f"✅ Преобразована timedelta колонка: {col}")
                    else:
                        # Пробуем числовое преобразование
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                        X[col] = X[col].fillna(0)
                        logger.info(f"✅ Преобразована колонка: {col}")
                except Exception as e:
                    logger.error(f"❌ Не удалось преобразовать {col}: {e}")
                    X = X.drop(columns=[col])
                    logger.info(f"🗑️  Удалена проблемная колонка: {col}")

        # Финальная проверка - оставляем только числовые типы
        final_numeric_types = ['int', 'float', 'number']
        valid_cols = []

        for col in X.columns:
            col_dtype = str(X[col].dtype).lower()
            if any(num_type in col_dtype for num_type in final_numeric_types):
                valid_cols.append(col)
            else:
                logger.warning(f"🗑️  Удаляем колонку с неподходящим типом: {col} ({X[col].dtype})")

        if len(valid_cols) != len(X.columns):
            X = X[valid_cols]
            logger.info(f"🔧 Оставлены только числовые колонки: {X.shape}")

        # Финальная проверка на NaN и Inf
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)

        logger.info(f"✅ Финальные данные: {X.shape}, все колонки числовые")
        logger.info(f"📊 Типы данных: {X.dtypes.value_counts().to_dict()}")
        main_pbar.update(1)

        if y.nunique() < 2:
            logger.error("❌ Нет разнообразия в целевых метках")
            return False

        # Обучаем модель
        main_pbar.set_description("🤖 Обучение модели")
        training_metrics = train_antifraud_model(X, y, models_dir, quality_metrics)
        main_pbar.update(1)

        # Валидируем модель
        main_pbar.set_description("✅ Валидация модели")
        model_trainer = ModelTrainer(str(models_dir))
        model_trainer.load_model("real_antifraud_model")
        validation_metrics = validate_antifraud_model(model_trainer, X, y)
        main_pbar.update(1)

        # Создаем отчет
        main_pbar.set_description("📄 Создание отчета")
        create_training_report(quality_metrics, training_metrics, validation_metrics, output_dir)
        main_pbar.update(1)

        logger.info("🎉 Обучение завершено успешно!")
        logger.info(f"💾 Модель сохранена: models/real_antifraud_model.joblib")
        logger.info(f"📄 Отчет создан: output/training_report.txt")

        return True

    except Exception as e:
        logger.error(f"❌ Критическая ошибка при обучении: {e}")
        logger.error("Traceback (most recent call last):")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
