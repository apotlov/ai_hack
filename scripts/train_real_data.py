#!/usr/bin/env python3
"""
Скрипт обучения антифрод модели на реальных данных
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

    quality_metrics = {}

    # Основные метрики
    quality_metrics.update({
        'samples_count': len(X),
        'features_count': X.shape[1],
        'target_distribution': y.value_counts().to_dict(),
        'fraud_rate': y.mean(),
        'missing_values_ratio': X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
    })

    # Анализ признаков
    numeric_features = X.select_dtypes(include=[np.number]).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    quality_metrics.update({
        'numeric_features_count': len(numeric_features),
        'categorical_features_count': len(categorical_features)
    })

    # Проверка на константные признаки
    constant_features = X.columns[X.nunique() <= 1]
    quality_metrics['constant_features_count'] = len(constant_features)

    # Признаки с высоким количеством пропусков
    high_missing_features = X.columns[X.isnull().sum() / len(X) > 0.5]
    quality_metrics['high_missing_features_count'] = len(high_missing_features)

    # Проверка баланса классов
    if len(y.value_counts()) == 2:
        minority_class_ratio = y.value_counts().min() / len(y)
        quality_metrics['minority_class_ratio'] = minority_class_ratio
        quality_metrics['is_balanced'] = minority_class_ratio >= 0.3

    # Логируем основные метрики
    logger.info(f"📈 Образцов: {quality_metrics['samples_count']}")
    logger.info(f"📊 Признаков: {quality_metrics['features_count']}")
    logger.info(f"🎯 Доля мошенничества: {quality_metrics['fraud_rate']:.2%}")
    logger.info(f"🔍 Пропущенных значений: {quality_metrics['missing_values_ratio']:.2%}")

    if quality_metrics.get('is_balanced', True):
        logger.info("✅ Классы относительно сбалансированы")
    else:
        logger.warning(f"⚠️  Дисбаланс классов: {quality_metrics['minority_class_ratio']:.2%}")

    return quality_metrics


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


def train_antifraud_model(X: pd.DataFrame, y: pd.Series,
                         models_dir: Path, quality_metrics: Dict) -> Dict[str, Any]:
    """
    Обучение антифрод модели

    Args:
        X: Признаки
        y: Целевые метки
        models_dir: Директория для сохранения моделей
        quality_metrics: Метрики качества данных

    Returns:
        Словарь с результатами обучения
    """
    logger.info("🤖 Начинаем обучение антифрод модели...")

    # Инициализируем тренера модели
    model_trainer = ModelTrainer(str(models_dir))

    # Настраиваем параметры модели в зависимости от данных
    if quality_metrics['fraud_rate'] < 0.1:
        # Для несбалансированных данных
        from sklearn.ensemble import RandomForestClassifier
        model_trainer.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',  # Важно для несбалансированных данных
            random_state=42,
            n_jobs=-1
        )
        logger.info("🎯 Настроена модель для несбалансированных данных")

    # Обучаем модель
    metrics = model_trainer.train(X, y, test_size=0.2)

    # Дополнительная валидация для антифрод модели
    additional_metrics = validate_antifraud_model(model_trainer, X, y, quality_metrics)
    metrics.update(additional_metrics)

    # Сохраняем модель
    model_name = "real_antifraud_model"
    model_trainer.save_model(model_name)

    logger.info(f"💾 Модель сохранена: {model_name}")

    return metrics


def validate_antifraud_model(model_trainer: ModelTrainer, X: pd.DataFrame,
                           y: pd.Series, quality_metrics: Dict) -> Dict[str, Any]:
    """
    Дополнительная валидация антифрод модели

    Args:
        model_trainer: Обученная модель
        X: Признаки
        y: Целевые метки
        quality_metrics: Метрики качества данных

    Returns:
        Дополнительные метрики
    """
    logger.info("🔍 Дополнительная валидация модели...")

    # Предсказания на полном датасете
    y_pred = model_trainer.predict(X)
    y_pred_proba = model_trainer.predict_proba(X)[:, 1]

    # Специфичные для антифрод метрики
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
    avg_precision = average_precision_score(y, y_pred_proba)

    # Анализ по разным порогам
    high_precision_threshold = None
    for i, (p, r, t) in enumerate(zip(precision, recall, thresholds)):
        if p >= 0.8:  # Ищем порог для точности 80%+
            high_precision_threshold = t
            break

    additional_metrics = {
        'avg_precision_score': avg_precision,
        'high_precision_threshold': high_precision_threshold,
        'fraud_detection_rate_at_80_precision': recall[precision >= 0.8][0] if any(precision >= 0.8) else 0
    }

    # Анализ важности признаков
    feature_importance = model_trainer.get_feature_importance(top_n=20)
    if not feature_importance.empty:
        additional_metrics['top_features'] = feature_importance.head(10)['feature'].tolist()

    return additional_metrics


def create_training_report(metrics: Dict[str, Any], quality_metrics: Dict[str, Any],
                         output_dir: Path) -> None:
    """
    Создание отчета об обучении

    Args:
        metrics: Метрики модели
        quality_metrics: Метрики качества данных
        output_dir: Директория для сохранения отчета
    """
    logger.info("📄 Создание отчета об обучении...")

    report_content = f"""
=== ОТЧЕТ ОБ ОБУЧЕНИИ АНТИФРОД МОДЕЛИ ===
Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

КАЧЕСТВО ДАННЫХ:
- Образцов для обучения: {quality_metrics['samples_count']:,}
- Количество признаков: {quality_metrics['features_count']:,}
- Доля мошенничества: {quality_metrics['fraud_rate']:.2%}
- Пропущенных значений: {quality_metrics['missing_values_ratio']:.2%}
- Числовых признаков: {quality_metrics['numeric_features_count']:,}
- Категориальных признаков: {quality_metrics['categorical_features_count']:,}

РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:
- Размер обучающей выборки: {metrics.get('train_size', 'N/A'):,}
- Размер тестовой выборки: {metrics.get('test_size', 'N/A'):,}

МЕТРИКИ КАЧЕСТВА МОДЕЛИ:
- Test Accuracy: {metrics.get('test_accuracy', 0):.4f}
- Test Precision: {metrics.get('test_precision', 0):.4f}
- Test Recall: {metrics.get('test_recall', 0):.4f}
- Test F1-Score: {metrics.get('test_f1', 0):.4f}
- Test AUC-ROC: {metrics.get('test_auc', 0):.4f}
- Average Precision: {metrics.get('avg_precision_score', 0):.4f}

КРОСС-ВАЛИДАЦИЯ:
- CV AUC (среднее): {metrics.get('cv_auc_mean', 0):.4f}
- CV AUC (стд. откл.): {metrics.get('cv_auc_std', 0):.4f}

АНТИФРОД СПЕЦИФИЧНЫЕ МЕТРИКИ:
- Порог для точности 80%: {metrics.get('high_precision_threshold', 'N/A')}
- Recall при точности 80%: {metrics.get('fraud_detection_rate_at_80_precision', 0):.4f}

ВАЖНЕЙШИЕ ПРИЗНАКИ:
"""

    # Добавляем топ признаки
    if 'top_features' in metrics:
        for i, feature in enumerate(metrics['top_features'], 1):
            report_content += f"{i:2d}. {feature}\n"

    report_content += f"""

РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ:
"""

    # Добавляем рекомендации
    if metrics.get('test_auc', 0) >= 0.85:
        report_content += "✅ Модель показывает отличное качество (AUC ≥ 0.85)\n"
    elif metrics.get('test_auc', 0) >= 0.75:
        report_content += "⚠️  Модель показывает хорошее качество (AUC ≥ 0.75)\n"
    else:
        report_content += "❌ Модель требует улучшения (AUC < 0.75)\n"

    if quality_metrics.get('fraud_rate', 0) < 0.05:
        report_content += "⚠️  Низкая доля мошенничества - рассмотрите методы балансировки\n"

    if quality_metrics.get('missing_values_ratio', 0) > 0.2:
        report_content += "⚠️  Высокая доля пропусков - улучшите качество данных\n"

    report_content += """
СЛЕДУЮЩИЕ ШАГИ:
1. Протестируйте модель на новых данных
2. Настройте пороги в зависимости от бизнес-требований
3. Мониторьте качество модели в продакшене
4. Регулярно переобучайте на новых данных

=== КОНЕЦ ОТЧЕТА ===
"""

    # Сохраняем отчет
    report_file = output_dir / "training_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    logger.info(f"📄 Отчет сохранен: {report_file}")


def main():
    """
    Основная функция обучения на реальных данных
    """
    logger.info("🚀 Запуск обучения антифрод модели на реальных данных")

    try:
        # Общий прогресс для всего процесса
        with tqdm(total=7, desc="🚀 Обучение антифрод модели", unit="этап", position=0) as main_pbar:

            # Настройки путей
            main_pbar.set_description("📂 Настройка путей")
            data_dir = Path(__file__).parent.parent / "data"
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

        # Проверяем что все колонки числовые
        non_numeric_cols = []
        for col in X.columns:
            if X[col].dtype == 'object':
                non_numeric_cols.append(col)
                logger.warning(f"⚠️  Найдена нечисловая колонка: {col} (тип: {X[col].dtype})")

        if non_numeric_cols:
            logger.error(f"❌ Найдены нечисловые колонки: {non_numeric_cols}")
            logger.info("🔧 Попытка принудительного преобразования...")

            for col in non_numeric_cols:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    X[col] = X[col].fillna(0)
                    logger.info(f"✅ Преобразована колонка: {col}")
                except Exception as e:
                    logger.error(f"❌ Не удалось преобразовать {col}: {e}")
                    X = X.drop(columns=[col])
                    logger.info(f"🗑️  Удалена проблемная колонка: {col}")

        # Финальная проверка
        remaining_object_cols = X.select_dtypes(include=['object']).columns
        if len(remaining_object_cols) > 0:
            logger.error(f"❌ Остались нечисловые колонки: {list(remaining_object_cols)}")
            X = X.select_dtypes(exclude=['object'])
            logger.info(f"🔧 Удалены все нечисловые колонки, осталось: {X.shape}")

        logger.info(f"✅ Финальные данные: {X.shape}, все колонки числовые")
        main_pbar.update(1)

        if y.nunique() < 2:
            logger.error("❌ Нет разнообразия в целевых метках")
            return False

        # Обучаем модель
        training_metrics = train_antifraud_model(X, y, models_dir, quality_metrics)

        # Создаем отчет
        create_training_report(training_metrics, quality_metrics, output_dir)

        # Сохраняем обучающие данные для анализа
        training_data_path = output_dir / "training_data_summary.csv"
        X_summary = X.describe()
        X_summary.to_csv(training_data_path)

        # Выводим итоговые результаты
        logger.info("\n" + "="*60)
        logger.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        logger.info("="*60)
        logger.info(f"📊 Обучено на {len(X):,} образцах")
        logger.info(f"🎯 Доля мошенничества: {y.mean():.2%}")
        logger.info(f"📈 Test AUC: {training_metrics.get('test_auc', 0):.4f}")
        logger.info(f"🎯 Test F1: {training_metrics.get('test_f1', 0):.4f}")
        logger.info(f"💾 Модель: real_antifraud_model")
        logger.info("="*60)

        # Рекомендации по следующим шагам
        logger.info("\n💡 СЛЕДУЮЩИЕ ШАГИ:")
        logger.info("1. Протестируйте модель:")
        logger.info("   python scripts/predict_real_data.py")
        logger.info("2. Добавьте LLM объяснения:")
        logger.info("   python scripts/predict_local_llm.py")
        logger.info("3. Просмотрите отчет:")
        logger.info(f"   cat {output_dir}/training_report.txt")

        return True

    except Exception as e:
        logger.error(f"❌ Критическая ошибка при обучении: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
