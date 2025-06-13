"""
Модуль объединения признаков из различных источников данных
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from data_loader import DataLoader
from amplitude_processor import AmplitudeProcessor
from audio_processor import AudioProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Класс для объединения признаков из разных источников:
    - Amplitude поведенческие данные
    - Аудио признаки
    """

    def __init__(self, data_dir: str = "data"):
        """
        Инициализация извлекателя признаков

        Args:
            data_dir: Директория с данными
        """
        self.data_dir = data_dir
        self.data_loader = DataLoader(data_dir)
        self.amplitude_processor = AmplitudeProcessor()
        self.audio_processor = AudioProcessor()

    def extract_features(self, include_audio: bool = True, include_amplitude: bool = True) -> pd.DataFrame:
        """
        Извлечение и объединение всех признаков

        Args:
            include_audio: Включать ли аудио признаки
            include_amplitude: Включать ли Amplitude признаки

        Returns:
            DataFrame с объединенными признаками
        """
        logger.info("Начинаем извлечение признаков...")

        all_features = []

        # Извлечение Amplitude признаков
        if include_amplitude:
            amplitude_features = self._extract_amplitude_features()
            if not amplitude_features.empty:
                all_features.append(amplitude_features)
                logger.info(f"Извлечено Amplitude признаков: {amplitude_features.shape}")

        # Извлечение аудио признаков
        if include_audio:
            audio_features = self._extract_audio_features()
            if not audio_features.empty:
                all_features.append(audio_features)
                logger.info(f"Извлечено аудио признаков: {audio_features.shape}")

        if not all_features:
            logger.warning("Не удалось извлечь признаки ни из одного источника")
            return pd.DataFrame()

        # Объединение признаков
        combined_features = self._combine_features(all_features)
        logger.info(f"Итоговые признаки: {combined_features.shape}")

        return combined_features

    def _extract_amplitude_features(self) -> pd.DataFrame:
        """
        Извлечение признаков из Amplitude данных

        Returns:
            DataFrame с Amplitude признаками
        """
        try:
            logger.info("Загрузка Amplitude данных...")
            amplitude_data = self.data_loader.load_amplitude_chunks()

            if amplitude_data.empty:
                logger.warning("Amplitude данные не найдены")
                return pd.DataFrame()

            logger.info("Извлечение Amplitude признаков...")
            features = self.amplitude_processor.extract_features(amplitude_data)

            # Валидация признаков
            features = self.amplitude_processor.validate_features(features)

            # Добавляем префикс для различения источника
            feature_cols = [col for col in features.columns if col != 'user_id']
            features = features.rename(columns={col: f'amplitude_{col}' for col in feature_cols})

            return features

        except Exception as e:
            logger.error(f"Ошибка при извлечении Amplitude признаков: {e}")
            return pd.DataFrame()

    def _extract_audio_features(self) -> pd.DataFrame:
        """
        Извлечение признаков из аудио данных

        Returns:
            DataFrame с аудио признаками
        """
        try:
            logger.info("Поиск аудио файлов...")
            audio_files = self.data_loader.get_audio_files_list()

            if not audio_files:
                logger.warning("Аудио файлы не найдены")
                return pd.DataFrame()

            logger.info(f"Найдено {len(audio_files)} аудио файлов")

            # Обработка аудио файлов
            features = self._process_audio_data(audio_files)

            if features.empty:
                return pd.DataFrame()

            # Добавляем префикс для различения источника
            feature_cols = [col for col in features.columns if col not in ['user_id', 'file_name']]
            features = features.rename(columns={col: f'audio_{col}' for col in feature_cols})

            return features

        except Exception as e:
            logger.error(f"Ошибка при извлечении аудио признаков: {e}")
            return pd.DataFrame()

    def _process_audio_data(self, audio_files: List) -> pd.DataFrame:
        """
        Обработка аудио данных и извлечение признаков

        Args:
            audio_files: Список путей к аудио файлам

        Returns:
            DataFrame с аудио признаками
        """
        try:
            # Фильтруем валидные файлы
            valid_files = []
            for audio_file in audio_files[:10]:  # Ограничиваем для MVP
                if self.audio_processor.validate_audio_file(str(audio_file)):
                    valid_files.append(str(audio_file))

            if not valid_files:
                logger.warning("Нет валидных аудио файлов")
                return pd.DataFrame()

            logger.info(f"Обработка {len(valid_files)} валидных аудио файлов...")

            # Извлечение признаков
            features = self.audio_processor.process_multiple_files(valid_files)

            # Группировка по пользователям (агрегация если несколько файлов на пользователя)
            if 'user_id' in features.columns:
                numeric_cols = features.select_dtypes(include=[np.number]).columns
                agg_dict = {col: ['mean', 'std', 'min', 'max'] for col in numeric_cols}

                # Добавляем подсчет файлов
                agg_dict['file_name'] = 'count'

                grouped_features = features.groupby('user_id').agg(agg_dict)

                # Упрощаем названия колонок
                grouped_features.columns = [f'{col[0]}_{col[1]}' if col[1] != '' else col[0]
                                          for col in grouped_features.columns]

                grouped_features = grouped_features.reset_index()

                return grouped_features

            return features

        except Exception as e:
            logger.error(f"Ошибка при обработке аудио данных: {e}")
            return pd.DataFrame()

    def _combine_features(self, feature_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Объединение признаков из разных источников

        Args:
            feature_list: Список DataFrame с признаками

        Returns:
            Объединенный DataFrame
        """
        if not feature_list:
            return pd.DataFrame()

        if len(feature_list) == 1:
            return feature_list[0]

        # Объединяем по user_id
        combined = feature_list[0]

        for features_df in feature_list[1:]:
            if 'user_id' in features_df.columns and 'user_id' in combined.columns:
                combined = pd.merge(combined, features_df, on='user_id', how='outer')
            else:
                logger.warning("Не найдена колонка user_id для объединения")

        # Заполняем пропущенные значения
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        combined[numeric_cols] = combined[numeric_cols].fillna(0)

        # Удаляем строки без user_id
        if 'user_id' in combined.columns:
            combined = combined.dropna(subset=['user_id'])

        logger.info(f"Объединено признаков: {combined.shape[1]} для {combined.shape[0]} пользователей")

        return combined

    def get_features_with_targets(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Получение признаков с целевыми метками

        Returns:
            Кортеж (признаки, целевые метки)
        """
        # Извлекаем признаки
        features_df = self.extract_features()

        if features_df.empty:
            return pd.DataFrame(), pd.Series()

        # Загружаем целевые метки
        targets_df = self.data_loader.load_target_data()

        if targets_df.empty:
            logger.warning("Целевые метки не найдены")
            return features_df, pd.Series()

        # Объединяем с целевыми метками
        if 'user_id' in features_df.columns and 'user_id' in targets_df.columns:
            merged = pd.merge(features_df, targets_df, on='user_id', how='inner')

            if merged.empty:
                logger.warning("Нет пересечений между признаками и целевыми метками")
                return features_df, pd.Series()

            # Разделяем признаки и цели
            target_col = 'is_fraud'
            if target_col in merged.columns:
                y = merged[target_col]
                X = merged.drop(columns=[target_col, 'user_id'])
            else:
                logger.error(f"Колонка {target_col} не найдена в целевых данных")
                return features_df, pd.Series()

            logger.info(f"Подготовлено {X.shape[0]} образцов с {X.shape[1]} признаками")
            logger.info(f"Распределение классов: {y.value_counts().to_dict()}")

            return X, y

        logger.error("Не удалось объединить признаки с целевыми метками")
        return features_df, pd.Series()

    def create_sample_features(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Создание примера признаков для тестирования

        Args:
            n_samples: Количество образцов

        Returns:
            DataFrame с примерами признаков
        """
        np.random.seed(42)

        features = {
            'user_id': [f'user_{i}' for i in range(n_samples)],

            # Amplitude признаки
            'amplitude_session_count': np.random.poisson(10, n_samples),
            'amplitude_avg_session_duration': np.random.normal(300, 100, n_samples),
            'amplitude_unique_event_types': np.random.randint(1, 10, n_samples),
            'amplitude_night_activity_ratio': np.random.beta(2, 8, n_samples),
            'amplitude_weekend_ratio': np.random.beta(3, 7, n_samples),

            # Аудио признаки
            'audio_mfcc_0_mean': np.random.normal(-5, 2, n_samples),
            'audio_mfcc_1_mean': np.random.normal(0, 1, n_samples),
            'audio_spectral_centroid_mean': np.random.normal(2000, 500, n_samples),
            'audio_rms_mean': np.random.exponential(0.1, n_samples),
            'audio_duration': np.random.normal(30, 10, n_samples)
        }

        return pd.DataFrame(features)

    def get_feature_statistics(self, features_df: pd.DataFrame) -> Dict:
        """
        Получение статистики по признакам

        Args:
            features_df: DataFrame с признаками

        Returns:
            Словарь со статистикой
        """
        if features_df.empty:
            return {}

        numeric_cols = features_df.select_dtypes(include=[np.number]).columns

        stats = {
            'total_features': len(features_df.columns),
            'numeric_features': len(numeric_cols),
            'samples_count': len(features_df),
            'missing_values': features_df.isnull().sum().to_dict(),
            'feature_types': {
                'amplitude': len([col for col in features_df.columns if col.startswith('amplitude_')]),
                'audio': len([col for col in features_df.columns if col.startswith('audio_')])
            }
        }

        if numeric_cols.any():
            stats['numeric_stats'] = features_df[numeric_cols].describe().to_dict()

        return stats

    def save_features(self, features_df: pd.DataFrame, filepath: str):
        """
        Сохранение признаков в файл

        Args:
            features_df: DataFrame с признаками
            filepath: Путь для сохранения
        """
        try:
            if filepath.endswith('.parquet'):
                features_df.to_parquet(filepath, index=False)
            else:
                features_df.to_csv(filepath, index=False)

            logger.info(f"Признаки сохранены: {filepath}")

        except Exception as e:
            logger.error(f"Ошибка при сохранении признаков: {e}")

    def load_features(self, filepath: str) -> pd.DataFrame:
        """
        Загрузка признаков из файла

        Args:
            filepath: Путь к файлу с признаками

        Returns:
            DataFrame с признаками
        """
        try:
            if filepath.endswith('.parquet'):
                features_df = pd.read_parquet(filepath)
            else:
                features_df = pd.read_csv(filepath)

            logger.info(f"Признаки загружены: {filepath}")
            return features_df

        except Exception as e:
            logger.error(f"Ошибка при загрузке признаков: {e}")
            return pd.DataFrame()
