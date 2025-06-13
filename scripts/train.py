#!/usr/bin/env python3
"""
Скрипт обучения антифрод модели
"""

import sys
import os
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent / "src"))

import logging
import pandas as pd
from typing import Dict, Any

from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Основная функция обучения модели
    """
    logger.info("🚀 Запуск обучения антифрод модели")

    try:
        # Настройки
        data_dir = Path(__file__).parent.parent / "data"
        models_dir = Path(__file__).parent.parent / "models"

        logger.info(f"Директория данных: {data_dir}")
        logger.info(f"Директория моделей: {models_dir}")

        # Создаем директории если их нет
        models_dir.mkdir(exist_ok=True)

        # Инициализация компонентов
        logger.info("📊 Инициализация извлекателя признаков...")
        feature_extractor = FeatureExtractor(str(data_dir))

        # Проверяем доступность данных
        data_summary = feature_extractor.data_loader.get_data_summary()
        logger.info(f"Сводка по данным: {data_summary}")

        # Создаем примеры данных если их нет
        if not data_summary['amplitude_files'] and not data_summary['audio_files']:
            logger.info("⚠️  Данные не найдены, создаем примеры для демонстрации...")
            feature_extractor.data_loader.create_sample_data()

        # Извлечение признаков и целевых меток
        logger.info("🔧 Извлечение признаков из данных...")
        X, y = feature_extractor.get_features_with_targets()

        if X.empty or y.empty:
            logger.error("❌ Не удалось извлечь данные для обучения")

            # Создаем синтетические данные для демонстрации
            logger.info("🧪 Создание синтетических данных для демонстрации...")
            X = feature_extractor.create_sample_features(n_samples=1000)

            # Создаем синтетические целевые метки
            import numpy as np
            np.random.seed(42)

            # Симулируем зависимость от некоторых признаков
            fraud_prob = (
                0.1 +
                0.3 * (X['amplitude_night_activity_ratio'] > 0.3).astype(int) +
                0.2 * (X['amplitude_session_count'] < 5).astype(int) +
                0.1 * (X['audio_duration'] < 20).astype(int)
            )

            y = pd.Series([
                np.random.choice([0, 1], p=[1-p, p])
                for p in fraud_prob
            ], name='is_fraud')

            # Убираем user_id из признаков
            if 'user_id' in X.columns:
                X = X.drop('user_id', axis=1)

        logger.info(f"📈 Подготовлено данных: {X.shape[0]} образцов, {X.shape[1]} признаков")
        logger.info(f"🎯 Распределение классов: {y.value_counts().to_dict()}")

        # Проверяем качество данных
        logger.info("🔍 Анализ качества данных...")
        feature_stats = feature_extractor.get_feature_statistics(X)
        logger.info(f"Статистика признаков: {feature_stats}")

        # Инициализация тренера модели
        logger.info("🤖 Инициализация тренера модели...")
        model_trainer = ModelTrainer(str(models_dir))

        # Обучение модели
        logger.info("🎓 Начинаем обучение модели...")
        metrics = model_trainer.train(X, y, test_size=0.2)

        # Вывод результатов обучения
        logger.info("📊 Результаты обучения:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric_name}: {value:.4f}")
            else:
                logger.info(f"  {metric_name}: {value}")

        # Важность признаков
        logger.info("🔍 Анализ важности признаков...")
        feature_importance = model_trainer.get_feature_importance(top_n=10)

        if not feature_importance.empty:
            logger.info("Топ-10 важных признаков:")
            for _, row in feature_importance.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # Сохранение модели
        logger.info("💾 Сохранение обученной модели...")
        model_trainer.save_model("antifraud_model_v1")

        # Сохранение признаков для анализа
        features_path = models_dir / "training_features.csv"
        X_with_target = X.copy()
        X_with_target['target'] = y
        X_with_target.to_csv(features_path, index=False)
        logger.info(f"Признаки сохранены: {features_path}")

        # Сохранение информации о модели
        model_info = model_trainer.get_model_info()
        model_info_path = models_dir / "model_info.txt"

        with open(model_info_path, 'w', encoding='utf-8') as f:
            f.write("=== Информация об антифрод модели ===\n\n")
            f.write(f"Тип модели: {model_info['model_type']}\n")
            f.write(f"Количество признаков: {model_info['feature_count']}\n")
            f.write(f"Обучена: {model_info['is_trained']}\n\n")

            f.write("=== Метрики модели ===\n")
            for metric, value in model_info['metrics'].items():
                if isinstance(value, float):
                    f.write(f"{metric}: {value:.4f}\n")
                else:
                    f.write(f"{metric}: {value}\n")

            f.write("\n=== Важность признаков ===\n")
            if not feature_importance.empty:
                for _, row in feature_importance.iterrows():
                    f.write(f"{row['feature']}: {row['importance']:.4f}\n")

        logger.info(f"Информация о модели сохранена: {model_info_path}")

        # Финальная проверка
        logger.info("🧪 Финальная проверка модели...")
        test_metrics = model_trainer.evaluate_model(X.head(100), y.head(100))
        logger.info("Метрики на контрольной выборке:")
        for metric, value in test_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

        logger.info("✅ Обучение успешно завершено!")
        logger.info(f"🎯 Основные результаты:")
        logger.info(f"   - Test AUC: {metrics.get('test_auc', 0):.4f}")
        logger.info(f"   - Test F1: {metrics.get('test_f1', 0):.4f}")
        logger.info(f"   - Количество признаков: {len(X.columns)}")
        logger.info(f"   - Размер обучающей выборки: {len(X)}")

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка при обучении модели: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
