#!/usr/bin/env python3
"""
Скрипт предсказания для реальных данных антифрод системы
"""

import sys
import os
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent / "src"))

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from real_features_processor import RealFeaturesProcessor
from model_trainer import ModelTrainer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_prediction_data(data_dir: Path) -> bool:
    """
    Проверка данных для предсказания

    Args:
        data_dir: Директория с данными для предсказания

    Returns:
        True если данные готовы для предсказания
    """
    logger.info("🔍 Проверка данных для предсказания...")

    amplitude_dir = data_dir / "amplitude"

    # Ищем файлы для предсказания (могут называться test_* или predict_*)
    data_patterns = [
        "test_amplitude_chunk_*.parquet",
        "predict_amplitude_chunk_*.parquet",
        "amplitude_chunk_*.parquet"
    ]

    found_files = []
    for pattern in data_patterns:
        found_files.extend(list(amplitude_dir.glob(pattern)))

    if found_files:
        logger.info(f"✅ Найдено файлов данных: {len(found_files)}")
        return True

    # Проверяем стандартные тренировочные файлы (для демо)
    train_files = list(amplitude_dir.glob("train_amplitude_chunk_*.parquet"))
    if train_files:
        logger.warning("⚠️  Найдены только тренировочные файлы, используем их для демонстрации")
        return True

    logger.error("❌ Не найдены данные для предсказания")
    return False


def load_trained_model(models_dir: Path) -> ModelTrainer:
    """
    Загрузка обученной модели

    Args:
        models_dir: Директория с моделями

    Returns:
        Загруженная модель
    """
    logger.info("📂 Загрузка обученной модели...")

    model_trainer = ModelTrainer(str(models_dir))

    # Пробуем загрузить реальную модель
    model_names = [
        "real_antifraud_model",
        "antifraud_model_v1",
        "antifraud_model"
    ]

    for model_name in model_names:
        try:
            model_trainer.load_model(model_name)
            logger.info(f"✅ Модель загружена: {model_name}")
            return model_trainer
        except FileNotFoundError:
            continue

    raise FileNotFoundError("❌ Не найдена обученная модель. Сначала запустите обучение.")


def prepare_prediction_features(features_processor: RealFeaturesProcessor,
                               prediction_data_dir: str) -> pd.DataFrame:
    """
    Подготовка признаков для предсказания

    Args:
        features_processor: Процессор признаков
        prediction_data_dir: Директория с данными для предсказания

    Returns:
        DataFrame с признаками для предсказания
    """
    logger.info("🔧 Подготовка признаков для предсказания...")

    # Создаем признаки из данных для предсказания
    features_df = features_processor.create_prediction_features(prediction_data_dir)

    if features_df.empty:
        raise ValueError("❌ Не удалось создать признаки для предсказания")

    # Удаляем session_id из признаков (сохраняем отдельно)
    session_ids = None
    if 'session_id' in features_df.columns:
        session_ids = features_df['session_id'].copy()
        features_df = features_df.drop('session_id', axis=1)

    logger.info(f"📊 Подготовлено признаков: {features_df.shape}")

    return features_df, session_ids


def make_predictions(model_trainer: ModelTrainer, X: pd.DataFrame) -> Dict[str, Any]:
    """
    Выполнение предсказаний

    Args:
        model_trainer: Обученная модель
        X: Признаки для предсказания

    Returns:
        Словарь с результатами предсказаний
    """
    logger.info("🔮 Выполнение предсказаний...")

    # Предсказание классов
    predictions = model_trainer.predict(X)

    # Предсказание вероятностей
    probabilities = model_trainer.predict_proba(X)
    fraud_probabilities = probabilities[:, 1]  # Вероятность мошенничества

    # Статистика результатов
    results = {
        'predictions': predictions,
        'fraud_probabilities': fraud_probabilities,
        'total_samples': len(X),
        'predicted_fraud_count': int(predictions.sum()),
        'predicted_fraud_rate': float(predictions.mean()),
        'avg_fraud_probability': float(fraud_probabilities.mean()),
        'max_fraud_probability': float(fraud_probabilities.max()),
        'min_fraud_probability': float(fraud_probabilities.min()),
        'high_risk_count': int((fraud_probabilities >= 0.7).sum()),
        'medium_risk_count': int(((fraud_probabilities >= 0.3) & (fraud_probabilities < 0.7)).sum()),
        'low_risk_count': int((fraud_probabilities < 0.3).sum())
    }

    return results


def create_prediction_report(results: Dict[str, Any], X: pd.DataFrame,
                           session_ids: Optional[pd.Series], output_dir: Path) -> pd.DataFrame:
    """
    Создание отчета с предсказаниями

    Args:
        results: Результаты предсказаний
        X: Исходные признаки
        session_ids: ID сессий
        output_dir: Директория для сохранения

    Returns:
        DataFrame с отчетом
    """
    logger.info("📊 Создание отчета с предсказаниями...")

    # Создаем основной DataFrame с результатами
    report_df = pd.DataFrame({
        'sample_id': range(len(X)),
        'prediction': results['predictions'],
        'fraud_probability': results['fraud_probabilities'],
        'risk_level': pd.cut(
            results['fraud_probabilities'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Низкий', 'Средний', 'Высокий']
        )
    })

    # Добавляем session_ids если есть
    if session_ids is not None:
        report_df.insert(0, 'session_id', session_ids.values)

    # Добавляем временные метки для анализа
    report_df['prediction_datetime'] = pd.Timestamp.now()

    # Добавляем некоторые ключевые признаки для анализа
    feature_cols_to_include = []

    # Ищем важные признаки
    important_patterns = ['amplitude_', 'audio_', 'app_', 'temporal_']
    for pattern in important_patterns:
        matching_cols = [col for col in X.columns if col.startswith(pattern)]
        if matching_cols:
            # Берем первые несколько признаков каждого типа
            feature_cols_to_include.extend(matching_cols[:3])

    # Добавляем выбранные признаки в отчет
    for col in feature_cols_to_include[:10]:  # Ограничиваем 10 колонками
        if col in X.columns:
            report_df[f'feature_{col}'] = X[col].values

    # Сортируем по вероятности мошенничества (от высокой к низкой)
    report_df = report_df.sort_values('fraud_probability', ascending=False)

    # Сохраняем детальный отчет
    detailed_report_path = output_dir / "fraud_predictions_detailed.csv"
    report_df.to_csv(detailed_report_path, index=False, encoding='utf-8')
    logger.info(f"💾 Детальный отчет сохранен: {detailed_report_path}")

    return report_df


def analyze_predictions(results: Dict[str, Any], report_df: pd.DataFrame) -> str:
    """
    Анализ результатов предсказаний

    Args:
        results: Результаты предсказаний
        report_df: Отчет с предсказаниями

    Returns:
        Текстовый анализ
    """
    analysis = f"""
=== АНАЛИЗ РЕЗУЛЬТАТОВ ПРЕДСКАЗАНИЙ ===
Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

ОБЩАЯ СТАТИСТИКА:
- Всего проанализировано образцов: {results['total_samples']:,}
- Предсказано случаев мошенничества: {results['predicted_fraud_count']:,}
- Общая доля мошенничества: {results['predicted_fraud_rate']:.2%}

РАСПРЕДЕЛЕНИЕ ПО УРОВНЯМ РИСКА:
- 🔴 Высокий риск (≥70%): {results['high_risk_count']:,} ({results['high_risk_count']/results['total_samples']:.1%})
- 🟡 Средний риск (30-70%): {results['medium_risk_count']:,} ({results['medium_risk_count']/results['total_samples']:.1%})
- 🟢 Низкий риск (<30%): {results['low_risk_count']:,} ({results['low_risk_count']/results['total_samples']:.1%})

ВЕРОЯТНОСТИ МОШЕННИЧЕСТВА:
- Средняя вероятность: {results['avg_fraud_probability']:.3f} ({results['avg_fraud_probability']:.1%})
- Максимальная вероятность: {results['max_fraud_probability']:.3f} ({results['max_fraud_probability']:.1%})
- Минимальная вероятность: {results['min_fraud_probability']:.3f} ({results['min_fraud_probability']:.1%})

ТОП-10 САМЫХ ПОДОЗРИТЕЛЬНЫХ СЛУЧАЕВ:
"""

    # Добавляем топ случаи
    top_cases = report_df.head(10)
    for i, (_, case) in enumerate(top_cases.iterrows(), 1):
        session_info = f" (ID: {case['session_id']})" if 'session_id' in case else ""
        analysis += f"""
{i:2d}. Образец {case['sample_id']}{session_info}
    Вероятность мошенничества: {case['fraud_probability']:.1%}
    Уровень риска: {case['risk_level']}
    Предсказание: {'МОШЕННИЧЕСТВО' if case['prediction'] == 1 else 'Легитимно'}
"""

    analysis += f"""

РЕКОМЕНДАЦИИ ПО ДЕЙСТВИЯМ:

НЕМЕДЛЕННЫЕ ДЕЙСТВИЯ (Высокий риск):
"""

    if results['high_risk_count'] > 0:
        analysis += f"""- Проверить {results['high_risk_count']} случаев высокого риска
- Заблокировать подозрительные операции
- Связаться с клиентами для подтверждения операций
- Уведомить службу безопасности
"""
    else:
        analysis += "- Случаев высокого риска не обнаружено\n"

    analysis += "\nМОНИТОРИНГ (Средний риск):\n"

    if results['medium_risk_count'] > 0:
        analysis += f"""- Усилить мониторинг {results['medium_risk_count']} случаев среднего риска
- Настроить дополнительные алерты
- Запросить дополнительную аутентификацию при необходимости
"""
    else:
        analysis += "- Случаев среднего риска не обнаружено\n"

    analysis += f"""
ОБЩИЕ РЕКОМЕНДАЦИИ:
- Регулярно переобучать модель на новых данных
- Мониторить изменения в паттернах мошенничества
- Анализировать ложноположительные срабатывания
- Настраивать пороги в зависимости от бизнес-требований

"""

    # Предупреждения
    if results['predicted_fraud_rate'] > 0.3:
        analysis += "⚠️  ВНИМАНИЕ: Подозрительно высокая доля предсказанного мошенничества\n"
        analysis += "   Возможно, модель требует донастройки или данные содержат аномалии\n\n"

    if results['avg_fraud_probability'] < 0.1:
        analysis += "ℹ️  ИНФОРМАЦИЯ: Низкий средний уровень риска в выборке\n"
        analysis += "   Это может указывать на качественную предварительную фильтрацию\n\n"

    analysis += "=== КОНЕЦ АНАЛИЗА ==="

    return analysis


def create_summary_files(results: Dict[str, Any], analysis: str,
                        report_df: pd.DataFrame, output_dir: Path):
    """
    Создание итоговых файлов

    Args:
        results: Результаты предсказаний
        analysis: Текстовый анализ
        report_df: Отчет с предсказаниями
        output_dir: Директория для сохранения
    """
    logger.info("📄 Создание итоговых файлов...")

    # Сохраняем анализ
    analysis_path = output_dir / "prediction_analysis.txt"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(analysis)
    logger.info(f"📄 Анализ сохранен: {analysis_path}")

    # Создаем краткий отчет только для высокого риска
    high_risk_cases = report_df[report_df['fraud_probability'] >= 0.7]
    if not high_risk_cases.empty:
        high_risk_path = output_dir / "high_risk_cases.csv"
        high_risk_cases.to_csv(high_risk_path, index=False, encoding='utf-8')
        logger.info(f"🚨 Случаи высокого риска сохранены: {high_risk_path}")

    # Создаем JSON с основной статистикой
    import json
    stats_path = output_dir / "prediction_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"📊 Статистика сохранена: {stats_path}")


def main():
    """
    Основная функция предсказания на реальных данных
    """
    logger.info("🚀 Запуск предсказания на реальных данных")

    try:
        # Настройки путей
        project_dir = Path(__file__).parent.parent
        models_dir = project_dir / "models"
        output_dir = project_dir / "output"

        # Директория с данными для предсказания
        # Можно передать как аргумент или использовать стандартную
        prediction_data_dir = project_dir / "data"

        if len(sys.argv) > 1:
            # Если передан путь к данным как аргумент
            prediction_data_dir = Path(sys.argv[1])
            logger.info(f"📂 Использую данные из: {prediction_data_dir}")

        # Создаем выходную директорию
        output_dir.mkdir(exist_ok=True)

        logger.info(f"🤖 Директория моделей: {models_dir}")
        logger.info(f"📊 Выходная директория: {output_dir}")

        # Проверяем данные для предсказания
        if not validate_prediction_data(prediction_data_dir):
            return False

        # Загружаем обученную модель
        model_trainer = load_trained_model(models_dir)

        # Получаем информацию о модели
        model_info = model_trainer.get_model_info()
        logger.info(f"ℹ️  Модель: {model_info['model_type']}")
        logger.info(f"ℹ️  Признаков: {model_info['feature_count']}")
        if 'test_auc' in model_info['metrics']:
            logger.info(f"ℹ️  Test AUC: {model_info['metrics']['test_auc']:.4f}")

        # Инициализируем процессор признаков
        logger.info("🔧 Инициализация процессора признаков...")
        features_processor = RealFeaturesProcessor(str(project_dir / "data"))

        # Подготавливаем признаки для предсказания
        X, session_ids = prepare_prediction_features(features_processor, str(prediction_data_dir))

        # Выполняем предсказания
        results = make_predictions(model_trainer, X)

        # Создаем отчет
        report_df = create_prediction_report(results, X, session_ids, output_dir)

        # Анализируем результаты
        analysis = analyze_predictions(results, report_df)

        # Создаем итоговые файлы
        create_summary_files(results, analysis, report_df, output_dir)

        # Выводим основные результаты
        logger.info("\n" + "="*60)
        logger.info("✅ ПРЕДСКАЗАНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        logger.info("="*60)
        logger.info(f"📊 Обработано образцов: {results['total_samples']:,}")
        logger.info(f"🎯 Предсказано мошенничества: {results['predicted_fraud_count']:,} ({results['predicted_fraud_rate']:.1%})")
        logger.info(f"🔴 Высокий риск: {results['high_risk_count']:,}")
        logger.info(f"🟡 Средний риск: {results['medium_risk_count']:,}")
        logger.info(f"🟢 Низкий риск: {results['low_risk_count']:,}")
        logger.info("="*60)

        # Показываем топ-3 случая
        logger.info("\n🚨 ТОП-3 ПОДОЗРИТЕЛЬНЫХ СЛУЧАЕВ:")
        top_3 = report_df.head(3)
        for i, (_, case) in enumerate(top_3.iterrows(), 1):
            session_info = f" (ID: {case['session_id']})" if 'session_id' in case else ""
            logger.info(f"{i}. Образец {case['sample_id']}{session_info}: {case['fraud_probability']:.1%} риск")

        # Показываем созданные файлы
        logger.info(f"\n📁 СОЗДАННЫЕ ФАЙЛЫ:")
        logger.info(f"   📊 Детальный отчет: fraud_predictions_detailed.csv")
        logger.info(f"   📄 Анализ: prediction_analysis.txt")
        logger.info(f"   📈 Статистика: prediction_stats.json")
        if results['high_risk_count'] > 0:
            logger.info(f"   🚨 Высокий риск: high_risk_cases.csv")

        logger.info(f"\n💡 СЛЕДУЮЩИЕ ШАГИ:")
        if results['high_risk_count'] > 0:
            logger.info(f"1. СРОЧНО проверьте {results['high_risk_count']} случаев высокого риска")
            logger.info(f"2. Просмотрите файл: {output_dir}/high_risk_cases.csv")
        logger.info(f"3. Изучите анализ: {output_dir}/prediction_analysis.txt")
        logger.info(f"4. Для объяснений запустите: python scripts/predict_local_llm.py")

        return True

    except Exception as e:
        logger.error(f"❌ Критическая ошибка при предсказании: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
