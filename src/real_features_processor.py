"""
Процессор признаков для реальных данных антифрод системы
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import re
from tqdm import tqdm

from real_data_loader import RealDataLoader
from audio_processor import AudioProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealFeaturesProcessor:
    """
    Класс для извлечения признаков из реальных данных антифрод системы
    """

    def __init__(self, data_dir: str = "data"):
        """
        Инициализация процессора реальных признаков

        Args:
            data_dir: Директория с данными
        """
        self.data_dir = data_dir
        self.data_loader = RealDataLoader(data_dir)
        self.audio_processor = AudioProcessor()

    def extract_amplitude_features(self, amplitude_data: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение признаков из amplitude данных (технические параметры звонков)

        Args:
            amplitude_data: DataFrame с amplitude данными

        Returns:
            DataFrame с извлеченными признаками
        """
        logger.info("🔧 Извлечение amplitude признаков...")

        if amplitude_data.empty:
            return pd.DataFrame()

        features_list = []

        # Группируем по session_id или другому идентификатору
        group_col = self._find_group_column(amplitude_data)
        if not group_col:
            logger.warning("⚠️  Не найдена колонка для группировки amplitude данных")
            return pd.DataFrame()

        logger.info(f"📊 Группировка по колонке: {group_col}")

        for group_id, group_data in amplitude_data.groupby(group_col):
            try:
                features = self._extract_group_amplitude_features(group_id, group_data)
                features_list.append(features)

            except Exception as e:
                logger.error(f"Ошибка при обработке группы {group_id}: {e}")
                continue

        if not features_list:
            return pd.DataFrame()

        features_df = pd.DataFrame(features_list)
        logger.info(f"✅ Извлечено amplitude признаков: {features_df.shape}")

        return features_df

    def _find_group_column(self, data: pd.DataFrame) -> Optional[str]:
        """
        Поиск колонки для группировки данных
        """
        # Приоритетные колонки для группировки
        primary_cols = ['APPLICATIONID', 'applicationid']
        secondary_cols = ['session_id', 'ID', 'id', 'call_id', 'user_id', 'client_id']

        # Сначала ищем APPLICATIONID
        for col in primary_cols:
            if col in data.columns:
                return col

        # Затем ищем среди вторичных
        for col in secondary_cols:
            if col in data.columns:
                return col

        return None

    def _extract_group_amplitude_features(self, group_id: str, group_data: pd.DataFrame) -> Dict:
        """
        Извлечение признаков для одной группы amplitude данных
        """
        features = {
            'applicationid': group_id,
            'amplitude_records_count': len(group_data)
        }

        # Статистические признаки по числовым колонкам
        numeric_cols = group_data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if group_data[col].notna().sum() > 0:
                values = group_data[col].dropna()

                features.update({
                    f'amplitude_{col}_mean': values.mean(),
                    f'amplitude_{col}_std': values.std(),
                    f'amplitude_{col}_min': values.min(),
                    f'amplitude_{col}_max': values.max(),
                    f'amplitude_{col}_median': values.median(),
                    f'amplitude_{col}_q25': values.quantile(0.25),
                    f'amplitude_{col}_q75': values.quantile(0.75),
                    f'amplitude_{col}_skew': values.skew(),
                    f'amplitude_{col}_kurt': values.kurtosis()
                })

        # Временные признаки если есть временная колонка
        time_cols = [col for col in group_data.columns if 'time' in col.lower() or 'date' in col.lower()]

        for time_col in time_cols:
            if group_data[time_col].notna().sum() > 1:
                try:
                    times = pd.to_datetime(group_data[time_col]).dropna()
                    if len(times) > 1:
                        time_diffs = times.diff().dt.total_seconds().dropna()

                        features.update({
                            f'amplitude_{time_col}_span_seconds': (times.max() - times.min()).total_seconds(),
                            f'amplitude_{time_col}_avg_interval': time_diffs.mean(),
                            f'amplitude_{time_col}_std_interval': time_diffs.std()
                        })
                except:
                    pass

        # Категориальные признаки
        categorical_cols = group_data.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if group_data[col].notna().sum() > 0:
                value_counts = group_data[col].value_counts()

                features.update({
                    f'amplitude_{col}_unique_count': len(value_counts),
                    f'amplitude_{col}_most_common': value_counts.index[0] if len(value_counts) > 0 else 'unknown',
                    f'amplitude_{col}_most_common_ratio': value_counts.iloc[0] / len(group_data) if len(value_counts) > 0 else 0
                })

        return features

    def extract_app_features(self, app_data: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение признаков из данных заявок

        Args:
            app_data: DataFrame с данными заявок

        Returns:
            DataFrame с признаками заявок
        """
        logger.info("📱 Извлечение признаков заявок...")

        if app_data.empty:
            return pd.DataFrame()

        features_list = []
        group_col = self._find_group_column(app_data)

        if not group_col:
            # Если нет группировочной колонки, обрабатываем каждую строку отдельно
            with tqdm(total=len(app_data), desc="📱 Обработка заявок", unit="заявка") as pbar:
                for idx, row in app_data.iterrows():
                    features = self._extract_single_app_features(f"app_{idx}", row)
                    features_list.append(features)
                    pbar.update(1)
        else:
            # Группируем по идентификатору
            groups = list(app_data.groupby(group_col))
            with tqdm(total=len(groups), desc="📱 Обработка групп заявок", unit="группа") as pbar:
                for group_id, group_data in groups:
                    # Для заявок обычно одна запись на группу, но может быть несколько
                    features = self._extract_group_app_features(group_id, group_data)
                    features_list.append(features)
                    pbar.update(1)

        if not features_list:
            return pd.DataFrame()

        features_df = pd.DataFrame(features_list)
        logger.info(f"✅ Извлечено признаков заявок: {features_df.shape}")

        return features_df

    def _extract_single_app_features(self, row_id: str, row: pd.Series) -> Dict:
        """
        Извлечение признаков из одной записи заявки
        """
        features = {'applicationid': row_id}

        # Обрабатываем все доступные колонки
        for col, value in row.items():
            if pd.isna(value):
                continue

            if isinstance(value, (int, float)):
                features[f'app_{col}'] = value
            elif isinstance(value, str):
                # Длина строки
                features[f'app_{col}_length'] = len(value)

                # Количество цифр
                features[f'app_{col}_digit_count'] = sum(c.isdigit() for c in value)

                # Количество букв
                features[f'app_{col}_alpha_count'] = sum(c.isalpha() for c in value)

                # Есть ли спецсимволы
                features[f'app_{col}_has_special'] = int(bool(re.search(r'[^a-zA-Z0-9а-яА-Я\s]', value)))

        return features

    def _extract_group_app_features(self, group_id: str, group_data: pd.DataFrame) -> Dict:
        """
        Извлечение признаков для группы заявок
        """
        features = {
            'applicationid': group_id,
            'app_records_count': len(group_data)
        }

        # Агрегируем числовые признаки
        numeric_cols = group_data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if group_data[col].notna().sum() > 0:
                values = group_data[col].dropna()

                features.update({
                    f'app_{col}_mean': values.mean(),
                    f'app_{col}_sum': values.sum(),
                    f'app_{col}_max': values.max(),
                    f'app_{col}_min': values.min()
                })

        # Категориальные признаки
        categorical_cols = group_data.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if group_data[col].notna().sum() > 0:
                unique_values = group_data[col].nunique()
                features[f'app_{col}_unique_count'] = unique_values

        return features

    def extract_audio_features(self, audio_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение признаков из аудио данных

        Args:
            audio_metadata: DataFrame с метаданными аудиофайлов

        Returns:
            DataFrame с аудио признаками
        """
        logger.info("🎵 Извлечение аудио признаков...")

        if audio_metadata.empty:
            return pd.DataFrame()

        features_list = []

        for _, row in audio_metadata.iterrows():
            try:
                # Извлекаем признаки из метаданных
                metadata_features = self._extract_audio_metadata_features(row)

                # Если есть путь к файлу, извлекаем аудио признаки
                if 'file_path' in row and pd.notna(row['file_path']):
                    audio_features = self.audio_processor.extract_audio_features(row['file_path'])

                    # Объединяем метаданные и аудио признаки
                    combined_features = {**metadata_features, **audio_features}
                else:
                    combined_features = metadata_features

                features_list.append(combined_features)

            except Exception as e:
                logger.error(f"Ошибка при обработке аудио {row.get('session_id', 'unknown')}: {e}")
                # Добавляем базовые признаки даже при ошибке
                features_list.append(self._get_default_audio_features(row))

        if not features_list:
            return pd.DataFrame()

        features_df = pd.DataFrame(features_list)

        # Добавляем applicationid если его нет
        if 'applicationid' not in features_df.columns:
            features_df['applicationid'] = [f'audio_{i}' for i in range(len(features_df))]

        logger.info(f"✅ Извлечено аудио признаков: {features_df.shape}")

        return features_df

    def _extract_audio_metadata_features(self, audio_row: pd.Series) -> Dict:
        """
        Извлечение признаков из метаданных аудиофайла
        """
        features = {}

        # APPLICATIONID (основной ключ связи)
        if 'applicationid' in audio_row and pd.notna(audio_row['applicationid']):
            features['applicationid'] = audio_row['applicationid']
        elif 'call_id' in audio_row and pd.notna(audio_row['call_id']):
            # Fallback на call_id если нет applicationid
            features['applicationid'] = audio_row['call_id']

        # Временные признаки
        if 'datetime' in audio_row and pd.notna(audio_row['datetime']):
            call_time = audio_row['datetime']
            features.update({
                'audio_call_hour': call_time.hour,
                'audio_call_weekday': call_time.weekday(),
                'audio_call_is_weekend': int(call_time.weekday() >= 5),
                'audio_call_is_night': int(call_time.hour >= 22 or call_time.hour <= 6),
                'audio_call_is_business_hours': int(9 <= call_time.hour <= 17)
            })

        # Признаки телефона
        if 'phone' in audio_row and pd.notna(audio_row['phone']):
            phone = str(audio_row['phone'])
            features.update({
                'audio_phone_length': len(phone),
                'audio_phone_starts_with_7': int(phone.startswith('7')),
                'audio_phone_starts_with_8': int(phone.startswith('8')),
                'audio_phone_has_country_code': int(len(phone) > 10)
            })

        # Признаки кодов
        if 'codes' in audio_row and isinstance(audio_row['codes'], list):
            codes = audio_row['codes']
            features.update({
                'audio_codes_count': len(codes),
                'audio_has_codes': int(len(codes) > 0)
            })

            # Добавляем первые несколько кодов как отдельные признаки
            for i, code in enumerate(codes[:3]):
                features[f'audio_code_{i+1}'] = code

        # Размер файла
        if 'file_size' in audio_row and pd.notna(audio_row['file_size']):
            file_size = audio_row['file_size']
            features.update({
                'audio_file_size_bytes': file_size,
                'audio_file_size_mb': file_size / (1024 * 1024),
                'audio_estimated_duration': file_size / 32000  # Примерная оценка длительности
            })

        return features

    def _get_default_audio_features(self, audio_row: pd.Series) -> Dict:
        """
        Получение базовых признаков при ошибке обработки аудио
        """
        features = {
            'applicationid': audio_row.get('applicationid', audio_row.get('call_id', 'unknown')),
            'audio_processing_error': 1
        }

        # Добавляем нулевые значения для основных аудио признаков
        audio_feature_names = self.audio_processor.get_feature_names()
        for feature_name in audio_feature_names:
            if feature_name not in ['file_name', 'sample_rate']:
                features[feature_name] = 0.0

        return features

    def extract_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение дополнительных временных признаков

        Args:
            data: DataFrame с временными данными

        Returns:
            DataFrame с временными признаками
        """
        logger.info("⏰ Извлечение временных признаков...")

        if data.empty:
            return pd.DataFrame()

        features_list = []
        group_col = self._find_group_column(data)

        if not group_col:
            return pd.DataFrame()

        for group_id, group_data in data.groupby(group_col):
            features = {'applicationid': group_id}

            # Ищем временные колонки
            time_columns = []
            for col in group_data.columns:
                if any(keyword in col.lower() for keyword in ['time', 'date', 'datetime']):
                    time_columns.append(col)

            for time_col in time_columns:
                try:
                    times = pd.to_datetime(group_data[time_col]).dropna()

                    if len(times) > 0:
                        # Основные временные статистики
                        features.update({
                            f'temporal_{time_col}_count': len(times),
                            f'temporal_{time_col}_span_minutes': (times.max() - times.min()).total_seconds() / 60,
                            f'temporal_{time_col}_first_hour': times.min().hour,
                            f'temporal_{time_col}_last_hour': times.max().hour
                        })

                        # Распределение по часам
                        hour_counts = times.dt.hour.value_counts()
                        features.update({
                            f'temporal_{time_col}_unique_hours': len(hour_counts),
                            f'temporal_{time_col}_night_ratio': ((times.dt.hour >= 22) | (times.dt.hour <= 6)).mean(),
                            f'temporal_{time_col}_business_hours_ratio': ((times.dt.hour >= 9) & (times.dt.hour <= 17)).mean()
                        })

                except Exception as e:
                    logger.debug(f"Ошибка при обработке временной колонки {time_col}: {e}")
                    continue

            features_list.append(features)

        if not features_list:
            return pd.DataFrame()

        features_df = pd.DataFrame(features_list)
        logger.info(f"✅ Извлечено временных признаков: {features_df.shape}")

        return features_df

    def combine_all_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Объединение всех признаков в единый датасет

        Returns:
            Кортеж (признаки, целевые метки)
        """
        logger.info("🔗 Объединение всех признаков...")

        # Загружаем базовые данные
        app_data = self.data_loader.load_app_data()
        target_data = self.data_loader.load_target_data()

        logger.info("🎯 УПРОЩЕННАЯ СТРАТЕГИЯ: Используем только app_data + target_data")
        logger.info("📊 Причина: Разные типы ID в amplitude vs target данных")

        target_series = pd.Series(dtype=int)
        combined_features = pd.DataFrame()

        if not target_data.empty and not app_data.empty:
            # Извлекаем признаки из app_data
            logger.info("🔧 Извлечение признаков из app_data...")
            with tqdm(total=3, desc="🔧 Обработка app данных", unit="этап") as pbar:
                app_features = self.extract_app_features(app_data)
                pbar.update(1)

                if app_features.empty:
                    logger.error("❌ Не удалось извлечь app признаки")
                    return pd.DataFrame(), pd.Series()

                target_col = self._find_target_column(target_data)
                merge_col = self._find_group_column(target_data)
                pbar.update(1)

                logger.info(f"🔍 Целевая колонка: {target_col}")
                logger.info(f"🔍 Колонка для слияния: {merge_col}")
                logger.info(f"🔍 App признаки: {app_features.shape}")
                logger.info(f"🔍 Target данные: {target_data.shape}")

                if target_col and merge_col:
                    # Проверяем пересечение ключей
                    app_ids = set(app_features['applicationid'].dropna().astype(str))
                    target_ids = set(target_data[merge_col].dropna().astype(str))
                    intersection = app_ids.intersection(target_ids)

                    logger.info(f"🔍 Пересечение ключей: {len(intersection)} из {len(app_ids)} app и {len(target_ids)} target")

                    if len(intersection) > 0:
                        # Прямое слияние app_data с target_data
                        logger.info("🔗 Выполнение merge app_features + target_data...")
                        final_data = pd.merge(
                            app_features, target_data,
                            left_on='applicationid', right_on=merge_col,
                            how='inner'
                        )

                        logger.info(f"✅ Размер после слияния: {final_data.shape}")

                        if not final_data.empty:
                            target_series = final_data[target_col].copy()

                            # Удаляем служебные колонки И идентификаторы
                            cols_to_drop = [
                                target_col, merge_col, 'applicationid',  # Основные ID колонки
                                'APPLICATIONID', 'application_id', 'ApplicationID',  # Вариации названий
                                'user_id', 'session_id', 'device_id',  # Другие ID
                            ] + [col for col in final_data.columns if col.endswith('_y')]

                            # Также удаляем все строковые колонки которые не являются числовыми
                            string_cols = []
                            for col in final_data.columns:
                                if col not in cols_to_drop:
                                    try:
                                        pd.to_numeric(final_data[col], errors='raise')
                                    except (ValueError, TypeError):
                                        string_cols.append(col)
                                        logger.warning(f"⚠️  Удаляем нечисловую колонку: {col}")

                            cols_to_drop.extend(string_cols)
                            combined_features = final_data.drop(columns=cols_to_drop, errors='ignore')

                            logger.info(f"🎯 Итоговые признаки: {combined_features.shape}")
                            logger.info(f"🎯 Целевые метки: {len(target_series)}")
                            logger.info(f"📊 Распределение классов: {target_series.value_counts().to_dict()}")
                            logger.info(f"🧹 Удалено колонок: {len(cols_to_drop)}")
                        else:
                            logger.error("❌ После merge получен пустой датафрейм!")
                    else:
                        logger.error("❌ Нет пересечения между app и target ключами!")
                        logger.info(f"🔍 Примеры app ID: {list(app_ids)[:5]}")
                        logger.info(f"🔍 Примеры target ID: {list(target_ids)[:5]}")
                else:
                    logger.error(f"❌ Не найдены обязательные колонки: target_col={target_col}, merge_col={merge_col}")

                pbar.update(1)
        else:
            logger.error("❌ Отсутствуют базовые данные (app_data или target_data)")

        # Финальная очистка
        combined_features = self._clean_features(combined_features)

        logger.info(f"✅ Итоговые признаки: {combined_features.shape}")
        logger.info(f"🎯 Целевых меток: {len(target_series)}")

        return combined_features, target_series

    def _find_target_column(self, target_data: pd.DataFrame) -> Optional[str]:
        """
        Поиск колонки с целевыми метками
        """
        possible_target_cols = [
            'is_fraud', 'target', 'fraud', 'label', 'class',
            'мошенничество', 'метка', 'fraud_flag'
        ]

        for col in possible_target_cols:
            if col in target_data.columns:
                return col

        # Берем последнюю числовую колонку
        numeric_cols = target_data.select_dtypes(include=[np.number]).columns
        return numeric_cols[-1] if len(numeric_cols) > 0 else None

    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Финальная очистка признаков
        """
        if features_df.empty:
            return features_df

        logger.info(f"🧹 Начальная очистка: {features_df.shape}")

        # Удаляем дублирующиеся колонки
        features_df = features_df.loc[:, ~features_df.columns.duplicated()]

        # КРИТИЧНО: Удаляем ВСЕ нечисловые колонки для ML
        numeric_cols = []
        non_numeric_cols = []

        for col in features_df.columns:
            # Проверяем можно ли преобразовать в числа
            try:
                # Пробуем преобразовать несколько значений
                sample_values = features_df[col].dropna().head(100)
                if len(sample_values) > 0:
                    pd.to_numeric(sample_values, errors='raise')
                    numeric_cols.append(col)
                else:
                    # Пустая колонка - удаляем
                    non_numeric_cols.append(col)
            except (ValueError, TypeError):
                non_numeric_cols.append(col)

        logger.info(f"📊 Числовых колонок: {len(numeric_cols)}")
        logger.info(f"🗑️  Нечисловых колонок для удаления: {len(non_numeric_cols)}")

        if non_numeric_cols:
            logger.info(f"🗑️  Удаляемые колонки: {non_numeric_cols[:10]}{'...' if len(non_numeric_cols) > 10 else ''}")

        # Оставляем только числовые колонки
        features_df = features_df[numeric_cols]

        # Заполняем пропуски в числовых данных
        features_df = features_df.fillna(0)

        # Заменяем бесконечные значения
        features_df = features_df.replace([np.inf, -np.inf], 0)

        # Удаляем константные колонки
        constant_cols = features_df.columns[features_df.nunique() <= 1]
        if len(constant_cols) > 0:
            logger.info(f"🗑️  Удаление константных колонок: {len(constant_cols)}")
            features_df = features_df.drop(columns=constant_cols)

        logger.info(f"🧹 Признаки очищены: {features_df.shape}")

        return features_df

    def create_prediction_features(self, data_dir: str) -> pd.DataFrame:
        """
        Создание признаков для предсказания (без целевых меток)

        Args:
            data_dir: Директория с данными для предсказания

        Returns:
            DataFrame с признаками для предсказания
        """
        logger.info("🔮 Создание признаков для предсказания...")

        # Временно меняем директорию данных
        original_data_dir = self.data_dir
        self.data_loader = RealDataLoader(data_dir)

        try:
            # Загружаем данные (без target_data)
            amplitude_data = self.data_loader.load_amplitude_chunks()
            app_data = self.data_loader.load_app_data()
            audio_metadata = self.data_loader.get_audio_files_metadata()

            # Извлекаем признаки
            amplitude_features = self.extract_amplitude_features(amplitude_data)
            app_features = self.extract_app_features(app_data)
            audio_features = self.extract_audio_features(audio_metadata)
            temporal_features = self.extract_temporal_features(amplitude_data)

            # Объединяем признаки
            all_features = [amplitude_features, app_features, audio_features, temporal_features]
            non_empty_features = [df for df in all_features if not df.empty]

            if not non_empty_features:
                return pd.DataFrame()

            combined_features = non_empty_features[0]

            for features_df in non_empty_features[1:]:
                if 'applicationid' in features_df.columns and 'applicationid' in combined_features.columns:
                    combined_features = pd.merge(
                        combined_features, features_df,
                        on='applicationid',
                        how='outer',
                        suffixes=('', '_dup')
                    )

            # Очищаем признаки
            combined_features = self._clean_features(combined_features)

            logger.info(f"✅ Признаки для предсказания готовы: {combined_features.shape}")

            return combined_features

        finally:
            # Восстанавливаем исходную директорию
            self.data_loader = RealDataLoader(original_data_dir)
