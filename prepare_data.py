#!/usr/bin/env python3
"""
Скрипт подготовки данных для антифрод системы
Поможет подключить ваши Parquet и WAV файлы
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import logging
from typing import Dict, List, Optional

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreparer:
    """
    Класс для подготовки и проверки данных
    """

    def __init__(self, base_dir: str = "."):
        """
        Инициализация

        Args:
            base_dir: Базовая директория проекта
        """
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.amplitude_dir = self.data_dir / "amplitude"
        self.audio_dir = self.data_dir / "audio"

        # Создаем директории
        self.amplitude_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def check_data_structure(self) -> Dict:
        """
        Проверка структуры данных

        Returns:
            Словарь с информацией о данных
        """
        logger.info("🔍 Проверка структуры данных...")

        # Проверяем Parquet файлы
        parquet_files = list(self.amplitude_dir.glob("*.parquet"))

        # Проверяем аудио файлы
        audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.m4a"]
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(self.audio_dir.glob(ext)))

        # Проверяем targets
        targets_file = self.data_dir / "targets.csv"

        info = {
            "parquet_files": len(parquet_files),
            "parquet_list": [f.name for f in parquet_files],
            "audio_files": len(audio_files),
            "audio_list": [f.name for f in audio_files[:5]],  # Показываем первые 5
            "has_targets": targets_file.exists(),
            "targets_path": str(targets_file)
        }

        return info

    def validate_parquet_files(self) -> bool:
        """
        Валидация Parquet файлов

        Returns:
            True если файлы валидны
        """
        logger.info("📊 Валидация Parquet файлов...")

        parquet_files = list(self.amplitude_dir.glob("*.parquet"))

        if not parquet_files:
            logger.warning("⚠️  Parquet файлы не найдены")
            return False

        required_columns = ["user_id"]
        recommended_columns = ["event_time", "event_type"]

        all_valid = True

        for file_path in parquet_files:
            try:
                logger.info(f"Проверяем файл: {file_path.name}")

                # Читаем первые несколько строк
                df = pd.read_parquet(file_path, nrows=100)

                logger.info(f"  Размер: {len(df)} строк (первые 100)")
                logger.info(f"  Колонки: {list(df.columns)}")

                # Проверяем обязательные колонки
                missing_required = [col for col in required_columns if col not in df.columns]
                if missing_required:
                    logger.error(f"  ❌ Отсутствуют обязательные колонки: {missing_required}")
                    all_valid = False
                else:
                    logger.info("  ✅ Обязательные колонки найдены")

                # Проверяем рекомендуемые колонки
                missing_recommended = [col for col in recommended_columns if col not in df.columns]
                if missing_recommended:
                    logger.warning(f"  ⚠️  Отсутствуют рекомендуемые колонки: {missing_recommended}")

                # Проверяем данные
                if df.empty:
                    logger.error(f"  ❌ Файл пустой")
                    all_valid = False

                # Проверяем user_id
                if "user_id" in df.columns:
                    unique_users = df["user_id"].nunique()
                    logger.info(f"  👥 Уникальных пользователей: {unique_users}")

                # Показываем пример данных
                logger.info("  📋 Пример данных:")
                logger.info(f"     {df.head(2).to_dict('records')}")

            except Exception as e:
                logger.error(f"  ❌ Ошибка при чтении {file_path.name}: {e}")
                all_valid = False

        return all_valid

    def validate_audio_files(self) -> bool:
        """
        Валидация аудио файлов

        Returns:
            True если файлы валидны
        """
        logger.info("🎵 Валидация аудио файлов...")

        audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.m4a"]
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(self.audio_dir.glob(ext)))

        if not audio_files:
            logger.warning("⚠️  Аудио файлы не найдены")
            return False

        logger.info(f"Найдено {len(audio_files)} аудио файлов")

        # Проверяем несколько файлов
        sample_files = audio_files[:5]  # Проверяем первые 5

        try:
            import librosa

            for audio_file in sample_files:
                try:
                    logger.info(f"Проверяем файл: {audio_file.name}")

                    # Пробуем загрузить файл
                    y, sr = librosa.load(str(audio_file), sr=None, duration=1.0)

                    logger.info(f"  📏 Длительность: ~{len(y)/sr:.1f} сек")
                    logger.info(f"  🔊 Частота дискретизации: {sr} Hz")
                    logger.info(f"  📊 Размер файла: {audio_file.stat().st_size / 1024:.1f} KB")

                    # Извлекаем user_id из имени файла
                    user_id = self._extract_user_id_from_filename(audio_file.stem)
                    logger.info(f"  👤 Извлеченный user_id: {user_id}")

                except Exception as e:
                    logger.error(f"  ❌ Ошибка при обработке {audio_file.name}: {e}")
                    return False

            logger.info("✅ Аудио файлы прошли валидацию")
            return True

        except ImportError:
            logger.error("❌ Библиотека librosa не установлена")
            return False

    def _extract_user_id_from_filename(self, filename: str) -> str:
        """
        Извлечение user_id из имени файла

        Args:
            filename: Имя файла без расширения

        Returns:
            Извлеченный user_id
        """
        # Пробуем разные паттерны
        if '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 2:
                return parts[0]  # Предполагаем что user_id в начале

        # Если не получилось, возвращаем как есть
        return filename

    def validate_targets(self) -> bool:
        """
        Валидация файла с целевыми метками

        Returns:
            True если файл валиден
        """
        logger.info("🎯 Валидация файла с целевыми метками...")

        targets_file = self.data_dir / "targets.csv"

        if not targets_file.exists():
            logger.warning("⚠️  Файл targets.csv не найден")
            return False

        try:
            df = pd.read_csv(targets_file)

            logger.info(f"📏 Размер: {len(df)} записей")
            logger.info(f"📊 Колонки: {list(df.columns)}")

            # Проверяем обязательные колонки
            required_columns = ["user_id", "is_fraud"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.error(f"❌ Отсутствуют колонки: {missing_columns}")
                return False

            # Проверяем значения is_fraud
            fraud_values = df["is_fraud"].unique()
            if not all(val in [0, 1] for val in fraud_values):
                logger.error(f"❌ Некорректные значения is_fraud: {fraud_values}")
                return False

            # Статистика
            fraud_count = df["is_fraud"].sum()
            fraud_rate = fraud_count / len(df)

            logger.info(f"📈 Статистика мошенничества:")
            logger.info(f"  Всего случаев: {len(df)}")
            logger.info(f"  Мошенничество: {fraud_count} ({fraud_rate:.1%})")
            logger.info(f"  Легитимные: {len(df) - fraud_count} ({1-fraud_rate:.1%})")

            # Показываем пример
            logger.info("📋 Пример данных:")
            logger.info(f"   {df.head(3).to_dict('records')}")

            logger.info("✅ Файл с целевыми метками валиден")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка при чтении targets.csv: {e}")
            return False

    def create_sample_data(self, n_users: int = 100):
        """
        Создание примеров данных для тестирования

        Args:
            n_users: Количество пользователей
        """
        logger.info(f"🧪 Создание примеров данных для {n_users} пользователей...")

        np.random.seed(42)

        # Создаем пример Amplitude данных
        sample_amplitude = []

        for user_id in range(1, n_users + 1):
            n_events = np.random.randint(5, 50)  # 5-50 событий на пользователя

            for event_id in range(n_events):
                event_data = {
                    "user_id": f"user_{user_id}",
                    "event_time": pd.Timestamp("2024-01-01") + pd.Timedelta(
                        days=np.random.randint(0, 90),
                        hours=np.random.randint(0, 24),
                        minutes=np.random.randint(0, 60)
                    ),
                    "event_type": np.random.choice([
                        "login", "transaction", "logout", "view_balance",
                        "transfer", "payment", "profile_update"
                    ]),
                    "session_duration": np.random.exponential(300),  # Среднее 5 минут
                    "click_count": np.random.poisson(10),
                    "page_views": np.random.poisson(5),
                    "device_type": np.random.choice(["mobile", "desktop", "tablet"]),
                    "amount": np.random.lognormal(3, 1) if np.random.random() < 0.3 else None
                }
                sample_amplitude.append(event_data)

        # Сохраняем Amplitude данные
        df_amplitude = pd.DataFrame(sample_amplitude)
        amplitude_file = self.amplitude_dir / "sample_amplitude_data.parquet"
        df_amplitude.to_parquet(amplitude_file, index=False)

        logger.info(f"💾 Создан файл: {amplitude_file}")
        logger.info(f"   Записей: {len(df_amplitude)}")
        logger.info(f"   Пользователей: {df_amplitude['user_id'].nunique()}")

        # Создаем целевые метки
        fraud_rate = 0.15  # 15% мошенничества

        targets_data = []
        for user_id in range(1, n_users + 1):
            is_fraud = 1 if np.random.random() < fraud_rate else 0
            targets_data.append({
                "user_id": f"user_{user_id}",
                "is_fraud": is_fraud
            })

        df_targets = pd.DataFrame(targets_data)
        targets_file = self.data_dir / "targets.csv"
        df_targets.to_csv(targets_file, index=False)

        logger.info(f"💾 Создан файл: {targets_file}")
        logger.info(f"   Записей: {len(df_targets)}")
        logger.info(f"   Мошенничество: {df_targets['is_fraud'].sum()} ({df_targets['is_fraud'].mean():.1%})")

        logger.info("✅ Примеры данных созданы успешно!")

    def copy_external_data(self, external_data_path: str):
        """
        Копирование внешних данных в проект

        Args:
            external_data_path: Путь к внешним данным
        """
        external_path = Path(external_data_path)

        if not external_path.exists():
            logger.error(f"❌ Путь не существует: {external_data_path}")
            return

        logger.info(f"📂 Копирование данных из: {external_data_path}")

        # Копируем Parquet файлы
        parquet_files = list(external_path.glob("**/*.parquet"))
        for file_path in parquet_files:
            dest_path = self.amplitude_dir / file_path.name
            shutil.copy2(file_path, dest_path)
            logger.info(f"✅ Скопирован: {file_path.name}")

        # Копируем аудио файлы
        audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.m4a"]
        for ext in audio_extensions:
            audio_files = list(external_path.glob(f"**/{ext}"))
            for file_path in audio_files:
                dest_path = self.audio_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                logger.info(f"✅ Скопирован: {file_path.name}")

        # Копируем targets.csv если есть
        targets_files = list(external_path.glob("**/targets.csv"))
        if targets_files:
            dest_path = self.data_dir / "targets.csv"
            shutil.copy2(targets_files[0], dest_path)
            logger.info(f"✅ Скопирован: targets.csv")

    def generate_report(self) -> str:
        """
        Генерация отчета о данных

        Returns:
            Текст отчета
        """
        info = self.check_data_structure()

        report = ["=" * 60]
        report.append("📊 ОТЧЕТ О ДАННЫХ АНТИФРОД СИСТЕМЫ")
        report.append("=" * 60)
        report.append("")

        # Parquet данные
        report.append("📈 AMPLITUDE ДАННЫЕ (Parquet):")
        if info["parquet_files"] > 0:
            report.append(f"  ✅ Найдено файлов: {info['parquet_files']}")
            for filename in info["parquet_list"]:
                report.append(f"     - {filename}")
        else:
            report.append("  ❌ Parquet файлы не найдены")
        report.append("")

        # Аудио данные
        report.append("🎵 АУДИО ДАННЫЕ:")
        if info["audio_files"] > 0:
            report.append(f"  ✅ Найдено файлов: {info['audio_files']}")
            for filename in info["audio_list"]:
                report.append(f"     - {filename}")
            if info["audio_files"] > 5:
                report.append(f"     ... и еще {info['audio_files'] - 5} файлов")
        else:
            report.append("  ❌ Аудио файлы не найдены")
        report.append("")

        # Целевые метки
        report.append("🎯 ЦЕЛЕВЫЕ МЕТКИ:")
        if info["has_targets"]:
            report.append(f"  ✅ Файл найден: {info['targets_path']}")
        else:
            report.append("  ❌ Файл targets.csv не найден")
        report.append("")

        # Рекомендации
        report.append("💡 РЕКОМЕНДАЦИИ:")
        if info["parquet_files"] == 0:
            report.append("  • Добавьте Parquet файлы в data/amplitude/")
        if info["audio_files"] == 0:
            report.append("  • Добавьте аудио файлы в data/audio/")
        if not info["has_targets"]:
            report.append("  • Создайте файл data/targets.csv с колонками: user_id, is_fraud")

        if info["parquet_files"] > 0 or info["audio_files"] > 0:
            report.append("  • Запустите валидацию: python prepare_data.py --validate")
            report.append("  • Начните обучение: python scripts/main.py --train")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)


def main():
    """
    Главная функция
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Подготовка данных для антифрод системы",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python prepare_data.py --check               # Проверить данные
  python prepare_data.py --validate            # Валидация всех данных
  python prepare_data.py --create-sample       # Создать примеры данных
  python prepare_data.py --copy /path/to/data  # Копировать внешние данные
  python prepare_data.py --report              # Сгенерировать отчет
        """
    )

    parser.add_argument("--check", action="store_true", help="Проверить структуру данных")
    parser.add_argument("--validate", action="store_true", help="Валидировать все данные")
    parser.add_argument("--create-sample", action="store_true", help="Создать примеры данных")
    parser.add_argument("--copy", type=str, help="Скопировать данные из внешнего источника")
    parser.add_argument("--report", action="store_true", help="Сгенерировать отчет")
    parser.add_argument("--users", type=int, default=100, help="Количество пользователей для примеров")

    args = parser.parse_args()

    # Приветствие
    print("=" * 60)
    print("📊 ПОДГОТОВКА ДАННЫХ ДЛЯ АНТИФРОД СИСТЕМЫ")
    print("=" * 60)
    print()

    preparer = DataPreparer()

    try:
        if args.check:
            info = preparer.check_data_structure()
            print("🔍 Структура данных:")
            print(f"  Parquet файлов: {info['parquet_files']}")
            print(f"  Аудио файлов: {info['audio_files']}")
            print(f"  Целевые метки: {'✅' if info['has_targets'] else '❌'}")

        elif args.validate:
            print("🔍 Запуск полной валидации...")

            parquet_valid = preparer.validate_parquet_files()
            audio_valid = preparer.validate_audio_files()
            targets_valid = preparer.validate_targets()

            print("\n📊 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ:")
            print(f"  Parquet файлы: {'✅' if parquet_valid else '❌'}")
            print(f"  Аудио файлы: {'✅' if audio_valid else '❌'}")
            print(f"  Целевые метки: {'✅' if targets_valid else '❌'}")

            if all([parquet_valid, audio_valid, targets_valid]):
                print("\n🎉 Все данные готовы для обучения!")
                print("Запустите: python scripts/main.py --train")
            else:
                print("\n⚠️  Есть проблемы с данными, исправьте их перед обучением")

        elif args.create_sample:
            preparer.create_sample_data(n_users=args.users)

        elif args.copy:
            preparer.copy_external_data(args.copy)

        elif args.report:
            report = preparer.generate_report()
            print(report)

            # Сохраняем отчет в файл
            report_file = Path("data_report.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n💾 Отчет сохранен в: {report_file}")

        else:
            # По умолчанию показываем отчет
            report = preparer.generate_report()
            print(report)

    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
