#!/usr/bin/env python3
"""
Скрипт предсказания с помощью антифрод модели
"""

import sys
import os
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent / "src"))

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(models_dir: Path, model_name: str = "antifraud_model_v1") -> ModelTrainer:
    """
    Загрузка обученной модели

    Args:
        models_dir: Директория с моделями
        model_name: Имя модели

    Returns:
        Загруженный ModelTrainer
    """
    logger.info(f"📂 Загрузка модели: {model_name}")

    model_trainer = ModelTrainer(str(models_dir))

    try:
        model_trainer.load_model(model_name)
        logger.info("✅ Модель успешно загружена")
        return model_trainer
    except FileNotFoundError:
        logger.error(f"❌ Модель не найдена: {model_name}")
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке модели: {e}")
        raise


def make_predictions(model_trainer: ModelTrainer, X: pd.DataFrame) -> Dict[str, Any]:
    """
    Выполнение предсказаний

    Args:
        model_trainer: Обученная модель
        X: Признаки для предсказания

    Returns:
        Словарь с результатами предсказаний
    """
    logger.info(f"🔮 Делаем предсказания для {len(X)} образцов...")

    # Предсказание классов
    predictions = model_trainer.predict(X)

    # Предсказание вероятностей
    probabilities = model_trainer.predict_proba(X)
    fraud_probabilities = probabilities[:, 1]  # Вероятность класса "мошенничество"

    # Формируем результат
    results = {
        'predictions': predictions,
        'fraud_probabilities': fraud_probabilities,
        'total_samples': len(X),
        'predicted_fraud_count': int(predictions.sum()),
        'predicted_fraud_rate': float(predictions.mean()),
        'avg_fraud_probability': float(fraud_probabilities.mean()),
        'max_fraud_probability': float(fraud_probabilities.max()),
        'min_fraud_probability': float(fraud_probabilities.min())
    }

    return results


def create_prediction_report(results: Dict[str, Any], X: pd.DataFrame,
                           output_path: Path) -> pd.DataFrame:
    """
    Создание отчета с предсказаниями

    Args:
        results: Результаты предсказаний
        X: Исходные признаки
        output_path: Путь для сохранения отчета

    Returns:
        DataFrame с отчетом
    """
    logger.info("📊 Создание отчета с предсказаниями...")

    # Создаем DataFrame с результатами
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

    # Добавляем топ признаки для анализа (первые несколько колонок)
    feature_cols = X.columns[:5] if len(X.columns) >= 5 else X.columns
    for col in feature_cols:
        report_df[f'feature_{col}'] = X[col].values

    # Сортируем по вероятности мошенничества (от высокой к низкой)
    report_df = report_df.sort_values('fraud_probability', ascending=False)

    # Сохраняем отчет
    report_df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"💾 Отчет сохранен: {output_path}")

    return report_df


def analyze_high_risk_cases(report_df: pd.DataFrame, threshold: float = 0.7) -> None:
    """
    Анализ случаев высокого риска

    Args:
        report_df: DataFrame с предсказаниями
        threshold: Пороговое значение для высокого риска
    """
    high_risk_cases = report_df[report_df['fraud_probability'] >= threshold]

    logger.info(f"🚨 Анализ случаев высокого риска (вероятность >= {threshold}):")
    logger.info(f"   Найдено случаев: {len(high_risk_cases)}")

    if len(high_risk_cases) > 0:
        logger.info(f"   Средняя вероятность: {high_risk_cases['fraud_probability'].mean():.4f}")
        logger.info(f"   Максимальная вероятность: {high_risk_cases['fraud_probability'].max():.4f}")

        logger.info("   Топ-5 самых подозрительных случаев:")
        for idx, row in high_risk_cases.head(5).iterrows():
            logger.info(f"     ID {row['sample_id']}: {row['fraud_probability']:.4f} ({row['risk_level']})")


def main():
    """
    Основная функция предсказания
    """
    logger.info("🎯 Запуск предсказания антифрод модели")

    try:
        # Настройки
        data_dir = Path(__file__).parent.parent / "data"
        models_dir = Path(__file__).parent.parent / "models"
        output_dir = Path(__file__).parent.parent / "output"

        # Создаем выходную директорию
        output_dir.mkdir(exist_ok=True)

        logger.info(f"Директория данных: {data_dir}")
        logger.info(f"Директория моделей: {models_dir}")
        logger.info(f"Выходная директория: {output_dir}")

        # Загрузка модели
        model_trainer = load_model(models_dir)

        # Получение информации о модели
        model_info = model_trainer.get_model_info()
        logger.info(f"ℹ️  Информация о модели:")
        logger.info(f"   Тип: {model_info['model_type']}")
        logger.info(f"   Признаков: {model_info['feature_count']}")
        logger.info(f"   Test AUC: {model_info['metrics'].get('test_auc', 'N/A')}")

        # Инициализация извлекателя признаков
        logger.info("📊 Инициализация извлекателя признаков...")
        feature_extractor = FeatureExtractor(str(data_dir))

        # Проверяем наличие новых данных для предсказания
        data_summary = feature_extractor.data_loader.get_data_summary()
        logger.info(f"Сводка по данным: {data_summary}")

        # Извлечение признаков для предсказания
        logger.info("🔧 Извлечение признаков для предсказания...")

        # Пробуем загрузить реальные данные
        X_predict = feature_extractor.extract_features()

        if X_predict.empty:
            logger.warning("⚠️  Новые данные не найдены, используем тестовые данные...")

            # Создаем тестовые данные для демонстрации
            X_predict = feature_extractor.create_sample_features(n_samples=50)

            # Убираем user_id если есть
            if 'user_id' in X_predict.columns:
                user_ids = X_predict['user_id'].copy()
                X_predict = X_predict.drop('user_id', axis=1)
            else:
                user_ids = pd.Series([f'test_user_{i}' for i in range(len(X_predict))])
        else:
            # Сохраняем user_id если есть
            if 'user_id' in X_predict.columns:
                user_ids = X_predict['user_id'].copy()
                X_predict = X_predict.drop('user_id', axis=1)
            else:
                user_ids = pd.Series([f'user_{i}' for i in range(len(X_predict))])

        logger.info(f"📈 Подготовлено для предсказания: {X_predict.shape[0]} образцов, {X_predict.shape[1]} признаков")

        # Выполнение предсказаний
        results = make_predictions(model_trainer, X_predict)

        # Вывод общей статистики
        logger.info("📊 Результаты предсказаний:")
        logger.info(f"   Всего образцов: {results['total_samples']}")
        logger.info(f"   Предсказано мошенничество: {results['predicted_fraud_count']}")
        logger.info(f"   Доля мошенничества: {results['predicted_fraud_rate']:.2%}")
        logger.info(f"   Средняя вероятность мошенничества: {results['avg_fraud_probability']:.4f}")
        logger.info(f"   Максимальная вероятность: {results['max_fraud_probability']:.4f}")

        # Создание подробного отчета
        output_path = output_dir / "fraud_predictions.csv"
        report_df = create_prediction_report(results, X_predict, output_path)

        # Добавляем user_id в отчет
        report_df.insert(0, 'user_id', user_ids.values)
        report_df.to_csv(output_path, index=False, encoding='utf-8')

        # Анализ случаев высокого риска
        analyze_high_risk_cases(report_df, threshold=0.7)

        # Создание сводного отчета
        summary_path = output_dir / "prediction_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== Сводный отчет предсказаний антифрод модели ===\n\n")
            f.write(f"Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Модель: {model_info['model_type']}\n")
            f.write(f"Количество признаков: {model_info['feature_count']}\n\n")

            f.write("=== Общая статистика ===\n")
            f.write(f"Всего образцов: {results['total_samples']}\n")
            f.write(f"Предсказано мошенничество: {results['predicted_fraud_count']}\n")
            f.write(f"Доля мошенничества: {results['predicted_fraud_rate']:.2%}\n")
            f.write(f"Средняя вероятность: {results['avg_fraud_probability']:.4f}\n")
            f.write(f"Максимальная вероятность: {results['max_fraud_probability']:.4f}\n\n")

            f.write("=== Распределение по уровням риска ===\n")
            risk_counts = report_df['risk_level'].value_counts()
            for risk_level, count in risk_counts.items():
                f.write(f"{risk_level}: {count} ({count/len(report_df):.1%})\n")

            f.write(f"\n=== Топ-10 самых подозрительных случаев ===\n")
            top_cases = report_df.head(10)
            for _, row in top_cases.iterrows():
                f.write(f"User {row['user_id']}: {row['fraud_probability']:.4f} ({row['risk_level']})\n")

        logger.info(f"📄 Сводный отчет сохранен: {summary_path}")

        # Создание файла с рекомендациями
        recommendations_path = output_dir / "recommendations.txt"
        with open(recommendations_path, 'w', encoding='utf-8') as f:
            f.write("=== Рекомендации по результатам анализа ===\n\n")

            high_risk_count = len(report_df[report_df['fraud_probability'] >= 0.7])
            medium_risk_count = len(report_df[
                (report_df['fraud_probability'] >= 0.3) &
                (report_df['fraud_probability'] < 0.7)
            ])

            f.write("1. НЕМЕДЛЕННЫЕ ДЕЙСТВИЯ:\n")
            if high_risk_count > 0:
                f.write(f"   - Проверить {high_risk_count} случаев с высоким риском (≥70%)\n")
                f.write("   - Рассмотреть блокировку подозрительных транзакций\n")
                f.write("   - Связаться с клиентами для подтверждения операций\n\n")
            else:
                f.write("   - Случаев высокого риска не обнаружено\n\n")

            f.write("2. МОНИТОРИНГ:\n")
            if medium_risk_count > 0:
                f.write(f"   - Усилить мониторинг {medium_risk_count} случаев среднего риска\n")
                f.write("   - Настроить дополнительные алерты\n\n")
            else:
                f.write("   - Продолжить стандартный мониторинг\n\n")

            f.write("3. АНАЛИЗ МОДЕЛИ:\n")
            if results['avg_fraud_probability'] > 0.5:
                f.write("   - Высокий средний уровень риска - проверить качество данных\n")
            else:
                f.write("   - Нормальный уровень риска в выборке\n")

            f.write(f"   - Модель показала Test AUC: {model_info['metrics'].get('test_auc', 'N/A')}\n")

        logger.info(f"💡 Рекомендации сохранены: {recommendations_path}")

        logger.info("✅ Предсказание успешно завершено!")
        logger.info(f"🎯 Основные файлы созданы:")
        logger.info(f"   - Детальный отчет: {output_path}")
        logger.info(f"   - Сводка: {summary_path}")
        logger.info(f"   - Рекомендации: {recommendations_path}")

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка при выполнении предсказания: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
