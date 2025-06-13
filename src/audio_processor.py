"""
Модуль обработки аудио данных для извлечения признаков
"""

import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import warnings

# Подавляем предупреждения librosa
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Класс для обработки аудио данных и извлечения признаков
    """

    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13):
        """
        Инициализация процессора аудио

        Args:
            sample_rate: Частота дискретизации для загрузки аудио
            n_mfcc: Количество MFCC коэффициентов
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def extract_audio_features(self, audio_path: str) -> Dict:
        """
        Извлечение признаков из аудио файла

        Args:
            audio_path: Путь к аудио файлу

        Returns:
            Словарь с извлеченными признаками
        """
        try:
            # Загрузка аудио файла
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            if len(y) == 0:
                logger.warning(f"Пустой аудио файл: {audio_path}")
                return self._get_default_features()

            # Извлечение различных типов признаков
            features = {}

            # MFCC признаки
            mfcc_features = self._process_mfcc(y, sr)
            features.update(mfcc_features)

            # Спектральные признаки
            spectral_features = self._extract_spectral_features(y, sr)
            features.update(spectral_features)

            # Временные признаки
            temporal_features = self._extract_temporal_features(y, sr)
            features.update(temporal_features)

            # Добавляем метаинформацию
            features.update({
                'file_name': Path(audio_path).name,
                'duration': len(y) / sr,
                'sample_rate': sr
            })

            return features

        except Exception as e:
            logger.error(f"Ошибка при обработке {audio_path}: {e}")
            return self._get_default_features()

    def _process_mfcc(self, y: np.ndarray, sr: int) -> Dict:
        """
        Извлечение MFCC признаков

        Args:
            y: Аудио сигнал
            sr: Частота дискретизации

        Returns:
            Словарь с MFCC признаками
        """
        try:
            # Извлечение MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

            # Статистики по MFCC коэффициентам
            mfcc_features = {}
            for i in range(self.n_mfcc):
                mfcc_coeff = mfcc[i]
                mfcc_features.update({
                    f'mfcc_{i}_mean': np.mean(mfcc_coeff),
                    f'mfcc_{i}_std': np.std(mfcc_coeff),
                    f'mfcc_{i}_min': np.min(mfcc_coeff),
                    f'mfcc_{i}_max': np.max(mfcc_coeff)
                })

            return mfcc_features

        except Exception as e:
            logger.error(f"Ошибка при извлечении MFCC: {e}")
            return {f'mfcc_{i}_{stat}': 0.0
                   for i in range(self.n_mfcc)
                   for stat in ['mean', 'std', 'min', 'max']}

    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict:
        """
        Извлечение спектральных признаков

        Args:
            y: Аудио сигнал
            sr: Частота дискретизации

        Returns:
            Словарь со спектральными признаками
        """
        try:
            features = {}

            # Спектральный центроид
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features.update({
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_centroid_min': np.min(spectral_centroids),
                'spectral_centroid_max': np.max(spectral_centroids)
            })

            # Спектральная полоса пропускания
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features.update({
                'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                'spectral_bandwidth_std': np.std(spectral_bandwidth),
                'spectral_bandwidth_min': np.min(spectral_bandwidth),
                'spectral_bandwidth_max': np.max(spectral_bandwidth)
            })

            # Спектральный откат
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features.update({
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'spectral_rolloff_std': np.std(spectral_rolloff),
                'spectral_rolloff_min': np.min(spectral_rolloff),
                'spectral_rolloff_max': np.max(spectral_rolloff)
            })

            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.update({
                'zcr_mean': np.mean(zcr),
                'zcr_std': np.std(zcr),
                'zcr_min': np.min(zcr),
                'zcr_max': np.max(zcr)
            })

            return features

        except Exception as e:
            logger.error(f"Ошибка при извлечении спектральных признаков: {e}")
            return {f'{feature}_{stat}': 0.0
                   for feature in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zcr']
                   for stat in ['mean', 'std', 'min', 'max']}

    def _extract_temporal_features(self, y: np.ndarray, sr: int) -> Dict:
        """
        Извлечение временных признаков

        Args:
            y: Аудио сигнал
            sr: Частота дискретизации

        Returns:
            Словарь с временными признаками
        """
        try:
            features = {}

            # RMS энергия
            rms = librosa.feature.rms(y=y)[0]
            features.update({
                'rms_mean': np.mean(rms),
                'rms_std': np.std(rms),
                'rms_min': np.min(rms),
                'rms_max': np.max(rms)
            })

            # Основная частота (pitch)
            try:
                pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)

                if pitch_values:
                    features.update({
                        'pitch_mean': np.mean(pitch_values),
                        'pitch_std': np.std(pitch_values),
                        'pitch_min': np.min(pitch_values),
                        'pitch_max': np.max(pitch_values)
                    })
                else:
                    features.update({
                        'pitch_mean': 0.0,
                        'pitch_std': 0.0,
                        'pitch_min': 0.0,
                        'pitch_max': 0.0
                    })
            except:
                features.update({
                    'pitch_mean': 0.0,
                    'pitch_std': 0.0,
                    'pitch_min': 0.0,
                    'pitch_max': 0.0
                })

            return features

        except Exception as e:
            logger.error(f"Ошибка при извлечении временных признаков: {e}")
            return {f'{feature}_{stat}': 0.0
                   for feature in ['rms', 'pitch']
                   for stat in ['mean', 'std', 'min', 'max']}

    def _get_default_features(self) -> Dict:
        """
        Получение признаков по умолчанию при ошибке обработки

        Returns:
            Словарь с признаками по умолчанию
        """
        features = {}

        # MFCC признаки по умолчанию
        for i in range(self.n_mfcc):
            for stat in ['mean', 'std', 'min', 'max']:
                features[f'mfcc_{i}_{stat}'] = 0.0

        # Спектральные признаки по умолчанию
        spectral_features = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zcr']
        for feature in spectral_features:
            for stat in ['mean', 'std', 'min', 'max']:
                features[f'{feature}_{stat}'] = 0.0

        # Временные признаки по умолчанию
        temporal_features = ['rms', 'pitch']
        for feature in temporal_features:
            for stat in ['mean', 'std', 'min', 'max']:
                features[f'{feature}_{stat}'] = 0.0

        # Метаинформация по умолчанию
        features.update({
            'file_name': 'unknown',
            'duration': 0.0,
            'sample_rate': self.sample_rate
        })

        return features

    def process_multiple_files(self, audio_files: List[str]) -> pd.DataFrame:
        """
        Обработка нескольких аудио файлов

        Args:
            audio_files: Список путей к аудио файлам

        Returns:
            DataFrame с признаками для всех файлов
        """
        all_features = []

        logger.info(f"Обработка {len(audio_files)} аудио файлов...")

        for i, audio_file in enumerate(audio_files):
            if i > 0 and i % 10 == 0:
                logger.info(f"Обработано {i}/{len(audio_files)} файлов")

            features = self.extract_audio_features(audio_file)

            # Добавляем идентификатор пользователя из имени файла
            file_name = Path(audio_file).stem
            features['user_id'] = self._extract_user_id_from_filename(file_name)

            all_features.append(features)

        features_df = pd.DataFrame(all_features)
        logger.info(f"Обработка завершена. Извлечено {features_df.shape[1]} признаков")

        return features_df

    def _extract_user_id_from_filename(self, filename: str) -> str:
        """
        Извлечение ID пользователя из имени файла

        Args:
            filename: Имя файла

        Returns:
            ID пользователя
        """
        # Простая логика извлечения user_id из имени файла
        # Можно адаптировать под конкретный формат имен файлов
        if '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 2:
                return parts[0]  # Предполагаем, что user_id в начале

        # Если не удалось извлечь, возвращаем само имя файла
        return filename

    def get_feature_names(self) -> List[str]:
        """
        Получение списка названий аудио признаков

        Returns:
            Список названий признаков
        """
        feature_names = []

        # MFCC признаки
        for i in range(self.n_mfcc):
            for stat in ['mean', 'std', 'min', 'max']:
                feature_names.append(f'mfcc_{i}_{stat}')

        # Спектральные признаки
        spectral_features = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zcr']
        for feature in spectral_features:
            for stat in ['mean', 'std', 'min', 'max']:
                feature_names.append(f'{feature}_{stat}')

        # Временные признаки
        temporal_features = ['rms', 'pitch']
        for feature in temporal_features:
            for stat in ['mean', 'std', 'min', 'max']:
                feature_names.append(f'{feature}_{stat}')

        # Метаинформация
        feature_names.extend(['file_name', 'duration', 'sample_rate'])

        return feature_names

    def validate_audio_file(self, audio_path: str) -> bool:
        """
        Проверка валидности аудио файла

        Args:
            audio_path: Путь к аудио файлу

        Returns:
            True если файл валидный, False иначе
        """
        try:
            if not os.path.exists(audio_path):
                return False

            # Пробуем загрузить файл
            y, sr = librosa.load(audio_path, sr=None, duration=1.0)  # Загружаем только первую секунду
            return len(y) > 0

        except Exception as e:
            logger.error(f"Невалидный аудио файл {audio_path}: {e}")
            return False
