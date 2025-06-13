"""
Модуль обработки Amplitude данных для извлечения поведенческих признаков
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AmplitudeProcessor:
    """
    Класс для обработки поведенческих данных Amplitude
    и извлечения признаков для антифрод системы
    """

    def __init__(self, user_id_col: str = 'user_id', timestamp_col: str = 'event_time'):
        """
        Инициализация процессора Amplitude данных

        Args:
            user_id_col: Название колонки с ID пользователя
            timestamp_col: Название колонки с временной меткой
        """
        self.user_id_col = user_id_col
        self.timestamp_col = timestamp_col

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение всех поведенческих признаков

        Args:
            df: DataFrame с Amplitude данными

        Returns:
            DataFrame с извлеченными признаками по пользователям
        """
        if df.empty:
            logger.warning("Пустой DataFrame для обработки")
            return pd.DataFrame()

        logger.info(f"Обработка данных для {df[self.user_id_col].nunique()} пользователей")

        # Подготовка данных
        df = self._prepare_data(df)

        # Извлечение различных типов признаков
        temporal_features = self._extract_temporal_features(df)
        behavioral_features = self._extract_behavioral_features(df)
        activity_features = self._extract_activity_features(df)

        # Объединение всех признаков
        all_features = [temporal_features, behavioral_features, activity_features]
        features_df = pd.concat(all_features, axis=1)

        # Удаление дубликатов колонок
        features_df = features_df.loc[:, ~features_df.columns.duplicated()]

        logger.info(f"Извлечено {features_df.shape[1]} признаков для {features_df.shape[0]} пользователей")

        return features_df

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка данных перед извлечением признаков

        Args:
            df: Исходные данные

        Returns:
            Подготовленные данные
        """
        df = df.copy()

        # Преобразование временных меток
        if self.timestamp_col in df.columns:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])

        # Сортировка по пользователю и времени
        df = df.sort_values([self.user_id_col, self.timestamp_col])

        return df

    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение временных признаков

        Args:
            df: Подготовленные данные

        Returns:
            DataFrame с временными признаками
        """
        temporal_features = []

        for user_id, user_data in df.groupby(self.user_id_col):
            if self.timestamp_col not in user_data.columns:
                continue

            timestamps = user_data[self.timestamp_col].dropna()

            if len(timestamps) < 2:
                continue

            # Временные интервалы между событиями
            time_diffs = timestamps.diff().dt.total_seconds().dropna()

            features = {
                self.user_id_col: user_id,
                'session_count': len(user_data),
                'time_span_days': (timestamps.max() - timestamps.min()).days,
                'avg_time_between_events': time_diffs.mean() if len(time_diffs) > 0 else 0,
                'std_time_between_events': time_diffs.std() if len(time_diffs) > 0 else 0,
                'min_time_between_events': time_diffs.min() if len(time_diffs) > 0 else 0,
                'max_time_between_events': time_diffs.max() if len(time_diffs) > 0 else 0,
                'median_time_between_events': time_diffs.median() if len(time_diffs) > 0 else 0,
                'hour_of_day_mean': timestamps.dt.hour.mean(),
                'hour_of_day_std': timestamps.dt.hour.std(),
                'day_of_week_mean': timestamps.dt.dayofweek.mean(),
                'weekend_ratio': (timestamps.dt.dayofweek >= 5).mean(),
                'night_activity_ratio': ((timestamps.dt.hour >= 22) | (timestamps.dt.hour <= 6)).mean()
            }

            temporal_features.append(features)

        return pd.DataFrame(temporal_features)

    def _extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение поведенческих признаков

        Args:
            df: Подготовленные данные

        Returns:
            DataFrame с поведенческими признаками
        """
        behavioral_features = []

        for user_id, user_data in df.groupby(self.user_id_col):
            features = {self.user_id_col: user_id}

            # Признаки событий
            if 'event_type' in user_data.columns:
                event_counts = user_data['event_type'].value_counts()
                features.update({
                    'unique_event_types': len(event_counts),
                    'most_common_event_ratio': event_counts.iloc[0] / len(user_data) if len(event_counts) > 0 else 0,
                    'event_diversity': -sum(p * np.log(p) for p in event_counts / event_counts.sum()) if len(event_counts) > 1 else 0
                })

                # Добавляем топ-3 события как отдельные признаки
                for i, (event_type, count) in enumerate(event_counts.head(3).items()):
                    features[f'event_type_{i+1}'] = event_type
                    features[f'event_type_{i+1}_count'] = count
                    features[f'event_type_{i+1}_ratio'] = count / len(user_data)

            # Признаки сессий
            if 'session_duration' in user_data.columns:
                session_durations = user_data['session_duration'].dropna()
                if len(session_durations) > 0:
                    features.update({
                        'avg_session_duration': session_durations.mean(),
                        'std_session_duration': session_durations.std(),
                        'min_session_duration': session_durations.min(),
                        'max_session_duration': session_durations.max(),
                        'median_session_duration': session_durations.median(),
                        'short_sessions_ratio': (session_durations < 60).mean(),
                        'long_sessions_ratio': (session_durations > 1800).mean()
                    })

            # Признаки активности
            numeric_cols = user_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != self.user_id_col and user_data[col].notna().sum() > 0:
                    values = user_data[col].dropna()
                    features.update({
                        f'{col}_mean': values.mean(),
                        f'{col}_std': values.std(),
                        f'{col}_min': values.min(),
                        f'{col}_max': values.max(),
                        f'{col}_median': values.median(),
                        f'{col}_sum': values.sum(),
                        f'{col}_count': len(values)
                    })

            behavioral_features.append(features)

        return pd.DataFrame(behavioral_features)

    def _extract_activity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение признаков активности

        Args:
            df: Подготовленные данные

        Returns:
            DataFrame с признаками активности
        """
        activity_features = []

        for user_id, user_data in df.groupby(self.user_id_col):
            if self.timestamp_col not in user_data.columns:
                continue

            timestamps = pd.to_datetime(user_data[self.timestamp_col])
            daily_activity = timestamps.dt.date.value_counts()

            features = {
                self.user_id_col: user_id,
                'active_days_count': len(daily_activity),
                'avg_daily_activity': daily_activity.mean(),
                'std_daily_activity': daily_activity.std(),
                'max_daily_activity': daily_activity.max(),
                'min_daily_activity': daily_activity.min(),
                'days_with_high_activity': (daily_activity > daily_activity.mean() + daily_activity.std()).sum(),
                'days_with_low_activity': (daily_activity < daily_activity.mean() - daily_activity.std()).sum(),
                'activity_consistency': 1 - (daily_activity.std() / daily_activity.mean()) if daily_activity.mean() > 0 else 0
            }

            # Признаки по дням недели
            weekday_activity = timestamps.dt.dayofweek.value_counts()
            for day in range(7):
                features[f'weekday_{day}_activity'] = weekday_activity.get(day, 0)

            # Признаки по часам
            hourly_activity = timestamps.dt.hour.value_counts()
            peak_hours = hourly_activity.nlargest(3).index.tolist()
            features.update({
                'peak_hour_1': peak_hours[0] if len(peak_hours) > 0 else 0,
                'peak_hour_2': peak_hours[1] if len(peak_hours) > 1 else 0,
                'peak_hour_3': peak_hours[2] if len(peak_hours) > 2 else 0,
                'peak_hour_activity_ratio': hourly_activity.iloc[0] / len(user_data) if len(hourly_activity) > 0 else 0
            })

            activity_features.append(features)

        return pd.DataFrame(activity_features)

    def get_feature_names(self) -> List[str]:
        """
        Получение списка названий признаков

        Returns:
            Список названий признаков
        """
        feature_names = [
            # Временные признаки
            'session_count', 'time_span_days', 'avg_time_between_events',
            'std_time_between_events', 'min_time_between_events', 'max_time_between_events',
            'median_time_between_events', 'hour_of_day_mean', 'hour_of_day_std',
            'day_of_week_mean', 'weekend_ratio', 'night_activity_ratio',

            # Поведенческие признаки
            'unique_event_types', 'most_common_event_ratio', 'event_diversity',
            'avg_session_duration', 'std_session_duration', 'min_session_duration',
            'max_session_duration', 'median_session_duration', 'short_sessions_ratio',
            'long_sessions_ratio',

            # Признаки активности
            'active_days_count', 'avg_daily_activity', 'std_daily_activity',
            'max_daily_activity', 'min_daily_activity', 'days_with_high_activity',
            'days_with_low_activity', 'activity_consistency', 'peak_hour_activity_ratio'
        ]

        return feature_names

    def validate_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Валидация и очистка признаков

        Args:
            features_df: DataFrame с признаками

        Returns:
            Очищенный DataFrame
        """
        if features_df.empty:
            return features_df

        # Заполнение пустых значений
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_cols] = features_df[numeric_cols].fillna(0)

        # Удаление бесконечных значений
        features_df = features_df.replace([np.inf, -np.inf], 0)

        # Удаление колонок с одинаковыми значениями
        constant_cols = features_df.columns[features_df.nunique() <= 1]
        if len(constant_cols) > 0:
            logger.info(f"Удаление константных колонок: {list(constant_cols)}")
            features_df = features_df.drop(columns=constant_cols)

        return features_df
