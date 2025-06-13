#!/usr/bin/env python3
"""
Главный скрипт запуска антифрод системы MVP
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from typing import Optional

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """
    Создание необходимых директорий
    """
    base_dir = Path(__file__).parent.parent
    directories = [
        base_dir / "data" / "amplitude",
        base_dir / "data" / "audio",
        base_dir / "models",
        base_dir / "output"
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Директория готова: {directory}")


def run_training():
    """
    Запуск обучения модели
    """
    logger.info("🎓 Запуск процесса обучения...")

    try:
        # Импортируем и запускаем скрипт обучения
        import subprocess
        import sys

        train_script = Path(__file__).parent / "train.py"
        result = subprocess.run([sys.executable, str(train_script)],
                              capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✅ Обучение завершено успешно")
            logger.info(result.stdout)
            return True
        else:
            logger.error("❌ Ошибка при обучении")
            logger.error(result.stderr)
            return False

    except Exception as e:
        logger.error(f"❌ Критическая ошибка при обучении: {e}")
        return False


def run_prediction():
    """
    Запуск предсказаний
    """
    logger.info("🔮 Запуск процесса предсказания...")

    try:
        import subprocess
        import sys

        predict_script = Path(__file__).parent / "predict.py"
        result = subprocess.run([sys.executable, str(predict_script)],
                              capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✅ Предсказание завершено успешно")
            logger.info(result.stdout)
            return True
        else:
            logger.error("❌ Ошибка при предсказании")
            logger.error(result.stderr)
            return False

    except Exception as e:
        logger.error(f"❌ Критическая ошибка при предсказании: {e}")
        return False


def run_full_pipeline():
    """
    Запуск полного пайплайна: обучение + предсказание
    """
    logger.info("🚀 Запуск полного пайплайна антифрод системы")

    # Подготовка среды
    setup_directories()

    # Обучение модели
    if not run_training():
        logger.error("❌ Пайплайн остановлен из-за ошибки обучения")
        return False

    # Предсказания
    if not run_prediction():
        logger.error("❌ Пайплайн остановлен из-за ошибки предсказания")
        return False

    logger.info("🎉 Полный пайплайн завершен успешно!")
    return True


def check_system():
    """
    Проверка системы и зависимостей
    """
    logger.info("🔍 Проверка системы...")

    try:
        # Проверка основных директорий
        base_dir = Path(__file__).parent.parent
        required_dirs = [
            base_dir / "src",
            base_dir / "scripts",
            base_dir / "data"
        ]

        for directory in required_dirs:
            if not directory.exists():
                logger.error(f"❌ Отсутствует директория: {directory}")
                return False
            logger.info(f"✅ Директория найдена: {directory}")

        # Проверка основных модулей
        try:
            import pandas
            import numpy
            import sklearn
            import joblib
            logger.info("✅ Основные зависимости установлены")
        except ImportError as e:
            logger.error(f"❌ Отсутствует зависимость: {e}")
            return False

        # Проверка модулей проекта
        try:
            from feature_extractor import FeatureExtractor
            from model_trainer import ModelTrainer
            from data_loader import DataLoader
            logger.info("✅ Модули проекта загружены успешно")
        except ImportError as e:
            logger.error(f"❌ Ошибка загрузки модуля проекта: {e}")
            return False

        logger.info("✅ Система готова к работе")
        return True

    except Exception as e:
        logger.error(f"❌ Ошибка при проверке системы: {e}")
        return False


def show_status():
    """
    Показать статус системы
    """
    logger.info("📊 Статус антифрод системы")

    base_dir = Path(__file__).parent.parent

    # Проверка моделей
    models_dir = base_dir / "models"
    if models_dir.exists():
        models = list(models_dir.glob("*.joblib"))
        logger.info(f"🤖 Найдено моделей: {len(models)}")
        for model in models:
            logger.info(f"   - {model.name}")
    else:
        logger.info("🤖 Модели не найдены")

    # Проверка данных
    data_dir = base_dir / "data"
    if data_dir.exists():
        amplitude_files = list((data_dir / "amplitude").glob("*.parquet")) if (data_dir / "amplitude").exists() else []
        audio_files = list((data_dir / "audio").glob("*.wav")) if (data_dir / "audio").exists() else []
        targets_file = data_dir / "targets.csv"

        logger.info(f"📊 Amplitude файлов: {len(amplitude_files)}")
        logger.info(f"🎵 Аудио файлов: {len(audio_files)}")
        logger.info(f"🎯 Целевые метки: {'✅' if targets_file.exists() else '❌'}")

    # Проверка результатов
    output_dir = base_dir / "output"
    if output_dir.exists():
        results = list(output_dir.glob("*.csv")) + list(output_dir.glob("*.txt"))
        logger.info(f"📄 Файлов результатов: {len(results)}")
        for result in results:
            logger.info(f"   - {result.name}")


def main():
    """
    Главная функция
    """
    parser = argparse.ArgumentParser(
        description="Антифрод система MVP - Главный скрипт запуска",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --full              # Полный пайплайн (обучение + предсказание)
  python main.py --train             # Только обучение модели
  python main.py --predict           # Только предсказания
  python main.py --check             # Проверка системы
  python main.py --status            # Показать статус
        """
    )

    parser.add_argument("--full", action="store_true",
                       help="Запустить полный пайплайн (обучение + предсказание)")
    parser.add_argument("--train", action="store_true",
                       help="Запустить только обучение модели")
    parser.add_argument("--predict", action="store_true",
                       help="Запустить только предсказания")
    parser.add_argument("--check", action="store_true",
                       help="Проверить систему и зависимости")
    parser.add_argument("--status", action="store_true",
                       help="Показать статус системы")
    parser.add_argument("--setup", action="store_true",
                       help="Создать необходимые директории")

    args = parser.parse_args()

    # Приветствие
    print("=" * 60)
    print("🛡️  АНТИФРОД СИСТЕМА MVP")
    print("   Банковская система детекции мошенничества")
    print("=" * 60)
    print()

    success = True

    try:
        if args.setup:
            setup_directories()

        elif args.check:
            success = check_system()

        elif args.status:
            show_status()

        elif args.train:
            success = run_training()

        elif args.predict:
            success = run_prediction()

        elif args.full:
            success = run_full_pipeline()

        else:
            # По умолчанию показываем статус и меню
            show_status()
            print()
            print("🔧 Доступные команды:")
            print("   --full      Полный пайплайн")
            print("   --train     Обучение модели")
            print("   --predict   Предсказания")
            print("   --check     Проверка системы")
            print("   --status    Статус системы")
            print("   --setup     Создать директории")
            print()
            print("Пример: python main.py --full")

    except KeyboardInterrupt:
        logger.info("\n⏸️  Работа прервана пользователем")
        success = False

    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        logger.error(traceback.format_exc())
        success = False

    finally:
        print()
        if success:
            print("✅ Работа завершена успешно!")
        else:
            print("❌ Работа завершена с ошибками")
        print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
