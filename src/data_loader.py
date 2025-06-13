"""
Модуль загрузки данных для MVP антифрод системы
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Класс для загрузки данных из различных источников:
    - Amplitude данные (parquet файлы)
    - Аудио файлы (wav)
    - Целевые метки
    """

    def __init__(self, data_dir: str = "data"):
        """
        Инициализация загрузчика данных

        Args:
            data_dir: Путь к директории с данными
        """
        self.data_dir = Path(data_dir)
        self.amplitude_dir = self.data_dir / "amplitude"
        self.audio_dir = self.data_dir / "audio"
        self.targets_file = self.data_dir / "targets.csv"

        # Проверяем существование директорий
        self._validate_directories()

    def _validate_directories(self):
        """Проверка существования необходимых директорий"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Директория данных не найдена: {self.data_dir}")

        if not self.amplitude_dir.exists():
            logger.warning(f"Директория Amplitude данных не найдена: {self.amplitude_dir}")

        if not self.audio_dir.exists():
            logger.warning(f"Директория аудио данных не найдена: {self.audio_dir}")

    def load_amplitude_chunks(self, chunk_size: int = 10000) -> pd.DataFrame:
        """
        Загрузка Amplitude данных из parquet файлов по частям

        Args:
            chunk_size: Размер чанка для обработки

        Returns:
            DataFrame с объединенными данными
        """
        if not self.amplitude_dir.exists():
            logger.error("Директория с Amplitude данными не существует")
            return pd.DataFrame()

        parquet_files = list(self.amplitude_dir.glob("*.parquet"))

        if not parquet_files:
            logger.warning("Не найдено parquet файлов в директории Amplitude")
            return pd.DataFrame()

        logger.info(f"Найдено {len(parquet_files)} parquet файлов")

        all_data = []

        for file_path in parquet_files:
            try:
                logger.info(f"Загружаем файл: {file_path.name}")

                # Загружаем данные по частям
                df = pd.read_parquet(file_path)

                # Добавляем источник данных
                df['source_file'] = file_path.name

                all_data.append(df)

            except Exception as e:
                logger.error(f"Ошибка при загрузке {file_path}: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        # Объединяем все данные
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Загружено {len(combined_df)} записей из Amplitude")

        return combined_df

    def load_app_data(self) -> Dict[str, pd.DataFrame]:
        """
        Загрузка данных приложения (если есть дополнительные файлы)

        Returns:
            Словарь с данными по типам
        """
        app_data = {}

        # Ищем CSV файлы в основной директории
        csv_files = list(self.data_dir.glob("*.csv"))

        for csv_file in csv_files:
            if csv_file.name != "targets.csv":
                try:
                    df = pd.read_csv(csv_file)
                    app_data[csv_file.stem] = df
                    logger.info(f"Загружен файл {csv_file.name}: {len(df)} записей")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке {csv_file}: {e}")

        return app_data

    def load_target_data(self) -> pd.DataFrame:
        """
        Загрузка целевых меток (fraud/not fraud)

        Returns:
            DataFrame с целевыми метками
        """
        if not self.targets_file.exists():
            logger.warning(f"Файл с целевыми метками не найден: {self.targets_file}")
            # Создаем пустой DataFrame с нужными колонками
            return pd.DataFrame(columns=['user_id', 'is_fraud'])

        try:
            targets_df = pd.read_csv(self.targets_file)
            logger.info(f"Загружено {len(targets_df)} целевых меток")

            # Проверяем наличие необходимых колонок
            required_columns = ['user_id', 'is_fraud']
            missing_columns = [col for col in required_columns if col not in targets_df.columns]

            if missing_columns:
                logger.error(f"Отсутствуют колонки: {missing_columns}")
                return pd.DataFrame(columns=required_columns)

            return targets_df

        except Exception as e:
            logger.error(f"Ошибка при загрузке целевых меток: {e}")
            return pd.DataFrame(columns=['user_id', 'is_fraud'])

    def get_audio_files_list(self) -> List[Path]:
        """
        Получение списка аудио файлов

        Returns:
            Список путей к аудио файлам
        """
        if not self.audio_dir.exists():
            logger.warning("Директория с аудио файлами не существует")
            return []

        # Поддерживаемые форматы
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
        audio_files = []

        for extension in audio_extensions:
            audio_files.extend(list(self.audio_dir.glob(extension)))

        logger.info(f"Найдено {len(audio_files)} аудио файлов")
        return audio_files

    def create_sample_data(self):
        """
        Создание примеров данных для тестирования (если данных нет)
        """
        logger.info("Создание примеров данных...")

        # Создаем пример целевых меток
        if not self.targets_file.exists():
            sample_targets = pd.DataFrame({
                'user_id': [f'user_{i}' for i in range(100)],
                'is_fraud': np.random.choice([0, 1], 100, p=[0.8, 0.2])
            })
            sample_targets.to_csv(self.targets_file, index=False)
            logger.info(f"Создан файл с примером целевых меток: {self.targets_file}")

        # Создаем пример Amplitude данных
        if not list(self.amplitude_dir.glob("*.parquet")):
            sample_amplitude = pd.DataFrame({
                'user_id': [f'user_{i}' for i in range(100)],
                'event_time': pd.date_range('2024-01-01', periods=100, freq='1H'),
                'event_type': np.random.choice(['login', 'transaction', 'logout'], 100),
                'session_duration': np.random.normal(300, 100, 100),
                'click_count': np.random.poisson(10, 100),
                'page_views': np.random.poisson(5, 100)
            })

            sample_file = self.amplitude_dir / "sample_data.parquet"
            sample_amplitude.to_parquet(sample_file)
            logger.info(f"Создан файл с примером Amplitude данных: {sample_file}")

    def get_data_summary(self) -> Dict:
        """
        Получение сводки по доступным данным

        Returns:
            Словарь с информацией о данных
        """
        summary = {
            'amplitude_files': len(list(self.amplitude_dir.glob("*.parquet"))) if self.amplitude_dir.exists() else 0,
            'audio_files': len(self.get_audio_files_list()),
            'targets_available': self.targets_file.exists(),
            'data_dir_exists': self.data_dir.exists()
        }

        return summary
