"""
Загрузчик данных для реальной структуры антифрод системы
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealDataLoader:
    """
    Класс для загрузки реальных данных антифрод системы

    Структура данных:
    - train_amplitude_chunk_XX.parquet - технические данные звонков (чанки)
    - train_app_data.parquet - справочные данные по заявкам
    - train_target_data.parquet - целевые метки
    - audiofiles/*.wav - аудиофайлы со структурой имени YYYYMMDDHHMMSS_ID_Телефон,_Код1,_Код2.wav
    - svod.csv - сводная таблица связей
    """

    def __init__(self, data_dir: str = "data"):
        """
        Инициализация загрузчика реальных данных

        Args:
            data_dir: Путь к директории с данными
        """
        self.data_dir = Path(data_dir)

        # Основные директории
        self.amplitude_dir = self.data_dir / "amplitude"
        self.audio_dir = self.data_dir / "audiofiles"
        self.svod_dir = self.data_dir

        # Создаем директории если их нет
        self.amplitude_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def load_amplitude_chunks(self) -> pd.DataFrame:
        """
        Загрузка и объединение всех чанков amplitude данных

        Returns:
            DataFrame с объединенными техническими данными звонков
        """
        logger.info("🔄 Загрузка amplitude чанков...")

        # Ищем все файлы с паттерном train_amplitude_chunk_XX.parquet
        chunk_files = list(self.amplitude_dir.glob("train_amplitude_chunk_*.parquet"))

        if not chunk_files:
            logger.warning("⚠️  Amplitude чанки не найдены")
            return pd.DataFrame()

        logger.info(f"📊 Найдено {len(chunk_files)} amplitude чанков")

        all_chunks = []

        for chunk_file in sorted(chunk_files):
            try:
                logger.info(f"Загружаем: {chunk_file.name}")
                chunk_df = pd.read_parquet(chunk_file)

                # Добавляем информацию о чанке
                chunk_df['source_chunk'] = chunk_file.name

                all_chunks.append(chunk_df)
                logger.info(f"  Загружено записей: {len(chunk_df)}")

            except Exception as e:
                logger.error(f"❌ Ошибка при загрузке {chunk_file}: {e}")
                continue

        if not all_chunks:
            logger.error("❌ Не удалось загрузить ни одного чанка")
            return pd.DataFrame()

        # Объединяем все чанки
        combined_df = pd.concat(all_chunks, ignore_index=True)
        logger.info(f"✅ Объединено {len(combined_df)} записей из {len(all_chunks)} чанков")

        # Показываем базовую информацию
        logger.info(f"📋 Колонки amplitude данных: {list(combined_df.columns)}")

        return combined_df

    def load_app_data(self) -> pd.DataFrame:
        """
        Загрузка справочных данных по заявкам

        Returns:
            DataFrame с данными заявок
        """
        logger.info("📱 Загрузка данных заявок...")

        app_data_file = self.amplitude_dir / "train_app_data.parquet"

        if not app_data_file.exists():
            logger.warning(f"⚠️  Файл данных заявок не найден: {app_data_file}")
            return pd.DataFrame()

        try:
            app_df = pd.read_parquet(app_data_file)
            logger.info(f"✅ Загружено {len(app_df)} записей заявок")
            logger.info(f"📋 Колонки данных заявок: {list(app_df.columns)}")

            return app_df

        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке данных заявок: {e}")
            return pd.DataFrame()

    def load_target_data(self) -> pd.DataFrame:
        """
        Загрузка целевых меток (fraud/not fraud)

        Returns:
            DataFrame с целевыми метками
        """
        logger.info("🎯 Загрузка целевых меток...")

        target_file = self.amplitude_dir / "train_target_data.parquet"

        if not target_file.exists():
            logger.warning(f"⚠️  Файл целевых меток не найден: {target_file}")
            return pd.DataFrame()

        try:
            target_df = pd.read_parquet(target_file)
            logger.info(f"✅ Загружено {len(target_df)} целевых меток")
            logger.info(f"📋 Колонки целевых данных: {list(target_df.columns)}")

            # Анализируем распределение меток
            if 'is_fraud' in target_df.columns:
                fraud_counts = target_df['is_fraud'].value_counts()
                fraud_rate = target_df['is_fraud'].mean()
                logger.info(f"📈 Распределение меток: {fraud_counts.to_dict()}")
                logger.info(f"📊 Доля мошенничества: {fraud_rate:.2%}")
            elif 'target' in target_df.columns:
                fraud_counts = target_df['target'].value_counts()
                fraud_rate = target_df['target'].mean()
                logger.info(f"📈 Распределение меток: {fraud_counts.to_dict()}")
                logger.info(f"📊 Доля мошенничества: {fraud_rate:.2%}")

            return target_df

        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке целевых меток: {e}")
            return pd.DataFrame()

    def load_svod_data(self) -> pd.DataFrame:
        """
        Загрузка сводных данных (метаданные звонков)

        Returns:
            DataFrame со сводными данными
        """
        logger.info("📋 Загрузка сводных данных...")

        # Ищем сводные файлы
        svod_files = []
        for pattern in ["svod.csv", "свод.csv", "*свод*.csv"]:
            svod_files.extend(list(self.svod_dir.glob(pattern)))

        if not svod_files:
            logger.warning("⚠️  Сводные файлы не найдены")
            return pd.DataFrame()

        logger.info(f"📊 Найдено сводных файлов: {len(svod_files)}")

        all_svod = []

        for svod_file in svod_files:
            try:
                logger.info(f"Загружаем: {svod_file.name}")

                # Пробуем разные кодировки
                for encoding in ['utf-8', 'cp1251', 'windows-1251']:
                    try:
                        svod_df = pd.read_csv(svod_file, encoding=encoding)
                        logger.info(f"  Успешно загружено с кодировкой {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    logger.error(f"❌ Не удалось определить кодировку для {svod_file}")
                    continue

                svod_df['source_file'] = svod_file.name
                all_svod.append(svod_df)
                logger.info(f"  Загружено записей: {len(svod_df)}")

            except Exception as e:
                logger.error(f"❌ Ошибка при загрузке {svod_file}: {e}")
                continue

        if not all_svod:
            return pd.DataFrame()

        # Объединяем все сводные данные
        combined_svod = pd.concat(all_svod, ignore_index=True)
        logger.info(f"✅ Объединено {len(combined_svod)} сводных записей")
        logger.info(f"📋 Колонки сводных данных: {list(combined_svod.columns)}")

        return combined_svod

    def parse_audio_filename(self, filename: str) -> Dict:
        """
        Парсинг имени аудиофайла для извлечения метаданных

        Формат: YYYYMMDDHHMMSS_ID_Телефон,_Код1,_Код2.wav
        Пример: 20241130151507_503121_77070094034,_500209,_500214.wav

        Args:
            filename: Имя аудиофайла

        Returns:
            Словарь с извлеченными данными
        """
        try:
            # Убираем расширение
            name_without_ext = filename.replace('.wav', '').replace('.mp3', '')

            # Разбиваем по подчеркиваниям
            parts = name_without_ext.split('_')

            if len(parts) < 3:
                # Если формат не соответствует ожидаемому
                return {
                    'call_id': name_without_ext,
                    'datetime': None,
                    'phone': None,
                    'codes': [],
                    'original_filename': filename,
                    'applicationid': None  # Будет связан через сводные данные
                }

            # Извлекаем дату и время
            datetime_str = parts[0]
            try:
                call_datetime = datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
            except:
                call_datetime = None

            # ID звонка (НЕ applicationid!)
            call_id = parts[1]

            # Телефон и коды (могут содержать запятые)
            phone_and_codes = '_'.join(parts[2:])

            # Разбиваем по запятым
            phone_parts = phone_and_codes.split(',')
            phone = phone_parts[0] if phone_parts else None

            # Коды (убираем пустые и лишние символы)
            codes = [code.strip('_').strip() for code in phone_parts[1:] if code.strip('_').strip()]

            return {
                'call_id': call_id,
                'datetime': call_datetime,
                'phone': phone,
                'codes': codes,
                'original_filename': filename,
                'applicationid': None  # Будет заполнен из сводных данных
            }

        except Exception as e:
            logger.error(f"Ошибка при парсинге имени файла {filename}: {e}")
            return {
                'call_id': filename.replace('.wav', '').replace('.mp3', ''),
                'datetime': None,
                'phone': None,
                'codes': [],
                'original_filename': filename,
                'applicationid': None
            }

    def get_audio_files_metadata(self) -> pd.DataFrame:
        """
        Получение метаданных всех аудиофайлов

        Returns:
            DataFrame с метаданными аудиофайлов
        """
        logger.info("🎵 Анализ аудиофайлов...")

        if not self.audio_dir.exists():
            logger.warning(f"⚠️  Директория аудиофайлов не найдена: {self.audio_dir}")
            return pd.DataFrame()

        # Ищем аудиофайлы
        audio_extensions = ['*.wav', '*.mp3', '*.flac']
        audio_files = []

        for ext in audio_extensions:
            audio_files.extend(list(self.audio_dir.glob(ext)))

        if not audio_files:
            logger.warning("⚠️  Аудиофайлы не найдены")
            return pd.DataFrame()

        logger.info(f"🎵 Найдено {len(audio_files)} аудиофайлов")

        # Парсим метаданные
        metadata_list = []

        for audio_file in audio_files:
            metadata = self.parse_audio_filename(audio_file.name)
            metadata['file_path'] = str(audio_file)
            metadata['file_size'] = audio_file.stat().st_size
            metadata_list.append(metadata)

        metadata_df = pd.DataFrame(metadata_list)

        # Связываем аудиофайлы с APPLICATIONID через сводные данные
        metadata_df = self._link_audio_with_applicationid(metadata_df)

        logger.info(f"✅ Обработано метаданных: {len(metadata_df)}")
        logger.info(f"📊 Уникальных call_id: {metadata_df['call_id'].nunique()}")

        return metadata_df

    def merge_all_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Объединение всех данных в единый датасет

        Returns:
            Кортеж (признаки, целевые метки)
        """
        logger.info("🔗 Объединение всех данных...")

        # Загружаем все компоненты
        amplitude_data = self.load_amplitude_chunks()
        app_data = self.load_app_data()
        target_data = self.load_target_data()
        svod_data = self.load_svod_data()
        audio_metadata = self.get_audio_files_metadata()

        # Определяем ключевые колонки для объединения
        merge_keys = self._identify_merge_keys(amplitude_data, app_data, target_data, audio_metadata)

        logger.info(f"🔑 Ключи для объединения: {merge_keys}")

        # Начинаем с amplitude данных как основы
        merged_data = amplitude_data.copy()

        # Добавляем данные заявок
        if not app_data.empty and merge_keys['app']:
            logger.info("🔗 Объединение с данными заявок...")
            merged_data = pd.merge(
                merged_data, app_data,
                on=merge_keys['app'],
                how='left',
                suffixes=('', '_app')
            )
            logger.info(f"📊 После объединения с заявками: {len(merged_data)} записей")

        # Добавляем аудио метаданные
        if not audio_metadata.empty and merge_keys['audio']:
            logger.info("🔗 Объединение с аудио метаданными...")
            merged_data = pd.merge(
                merged_data, audio_metadata,
                on=merge_keys['audio'],
                how='left',
                suffixes=('', '_audio')
            )
            logger.info(f"📊 После объединения с аудио: {len(merged_data)} записей")

        # Добавляем сводные данные
        if not svod_data.empty and merge_keys['svod']:
            logger.info("🔗 Объединение со сводными данными...")
            merged_data = pd.merge(
                merged_data, svod_data,
                on=merge_keys['svod'],
                how='left',
                suffixes=('', '_svod')
            )
            logger.info(f"📊 После объединения со сводом: {len(merged_data)} записей")

        # Объединяем с целевыми метками
        target_series = pd.Series(dtype=int)

        if not target_data.empty and merge_keys['target']:
            logger.info("🎯 Объединение с целевыми метками...")

            # Объединяем с целевыми данными по APPLICATIONID
            final_data = pd.merge(
                merged_data, target_data,
                on=merge_keys['target'],
                how='inner',  # Только записи с метками
                suffixes=('', '_target')
            )

            # Извлекаем целевую переменную
            target_col = self._identify_target_column(target_data)
            if target_col:
                target_series = final_data[target_col]
                final_data = final_data.drop(columns=[target_col])

            merged_data = final_data
            logger.info(f"📊 Финальный датасет: {len(merged_data)} записей")
            logger.info(f"🎯 Целевых меток: {len(target_series)}")

            # Проверяем распределение меток
            if len(target_series) > 0:
                fraud_count = target_series.sum()
                fraud_rate = fraud_count / len(target_series)
                logger.info(f"📈 Распределение: мошенничество {fraud_count}, легитимные {len(target_series) - fraud_count}")
                logger.info(f"📊 Доля мошенничества: {fraud_rate:.2%}")

        # Очистка и подготовка данных
        merged_data = self._clean_merged_data(merged_data)

        return merged_data, target_series

    def _identify_merge_keys(self, amplitude_data: pd.DataFrame, app_data: pd.DataFrame,
                           target_data: pd.DataFrame, audio_metadata: pd.DataFrame) -> Dict:
        """
        Определение ключей для объединения данных
        """
        merge_keys = {}

        # Приоритетные ключевые колонки для связи
        primary_keys = ['APPLICATIONID', 'applicationid']
        secondary_keys = ['call_id', 'session_id', 'ID', 'id', 'user_id']

        # Для каждого датасета находим подходящие ключи
        for dataset_name, dataset in [
            ('app', app_data),
            ('target', target_data),
            ('audio', audio_metadata)
        ]:
            if dataset.empty:
                merge_keys[dataset_name] = None
                continue

            # Сначала ищем APPLICATIONID (приоритет)
            found_key = None

            for key in primary_keys:
                if key in amplitude_data.columns and key in dataset.columns:
                    found_key = key
                    break

            # Если не нашли, ищем среди вторичных ключей
            if not found_key:
                for key in secondary_keys:
                    if key in amplitude_data.columns and key in dataset.columns:
                        found_key = key
                        break

            merge_keys[dataset_name] = found_key

        # Для сводных данных связь идет через имя файла
        merge_keys['svod'] = None

        return merge_keys

    def _identify_target_column(self, target_data: pd.DataFrame) -> Optional[str]:
        """
        Определение колонки с целевыми метками
        """
        possible_target_cols = [
            'is_fraud', 'target', 'fraud', 'label', 'class',
            'мошенничество', 'метка'
        ]

        for col in possible_target_cols:
            if col in target_data.columns:
                return col

        # Если не нашли, берем последнюю числовую колонку
        numeric_cols = target_data.select_dtypes(include=[np.number]).columns
        return numeric_cols[-1] if len(numeric_cols) > 0 else None

    def _clean_merged_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Очистка объединенных данных
        """
        if data.empty:
            return data

        # Удаляем дублирующиеся колонки
        data = data.loc[:, ~data.columns.duplicated()]

        # Убираем служебные колонки
        cols_to_drop = [col for col in data.columns if
                       col.startswith('source_') or
                       col.endswith('_target') or
                       col in ['original_filename', 'file_path']]

        data = data.drop(columns=cols_to_drop, errors='ignore')

        # Заполняем пропуски
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(0)

        categorical_cols = data.select_dtypes(include=['object']).columns
        data[categorical_cols] = data[categorical_cols].fillna('unknown')

        logger.info(f"🧹 Данные очищены: {data.shape}")

        return data

    def get_data_summary(self) -> Dict:
        """
        Получение сводки по всем доступным данным

        Returns:
            Словарь с информацией о данных
        """
        summary = {}

        # Amplitude чанки
        amplitude_files = list(self.amplitude_dir.glob("train_amplitude_chunk_*.parquet"))
        summary['amplitude_chunks'] = len(amplitude_files)

        # Данные заявок
        app_file = self.amplitude_dir / "train_app_data.parquet"
        summary['app_data_available'] = app_file.exists()

        # Целевые данные
        target_file = self.amplitude_dir / "train_target_data.parquet"
        summary['target_data_available'] = target_file.exists()

        # Аудиофайлы
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(list(self.audio_dir.glob(ext)))
        summary['audio_files_count'] = len(audio_files)

        # Сводные файлы
        svod_files = []
        for pattern in ["svod.csv", "свод.csv", "*свод*.csv"]:
            svod_files.extend(list(self.svod_dir.glob(pattern)))
        summary['svod_files_count'] = len(svod_files)

        return summary

    def _link_audio_with_applicationid(self, audio_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Связывание аудиофайлов с APPLICATIONID через сводные данные

        Args:
            audio_metadata: DataFrame с метаданными аудиофайлов

        Returns:
            DataFrame с добавленными APPLICATIONID
        """
        logger.info("🔗 Связывание аудиофайлов с APPLICATIONID...")

        # Загружаем сводные данные
        svod_data = self.load_svod_data()

        if svod_data.empty:
            logger.warning("⚠️  Сводные данные не найдены, связка через имена файлов невозможна")
            return audio_metadata

        # Ищем колонки для связки в сводных данных
        filename_cols = []
        applicationid_cols = []

        for col in svod_data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['файл', 'file', 'name', 'аудио']):
                filename_cols.append(col)
            if any(keyword in col_lower for keyword in ['application', 'заявка', 'номер']):
                applicationid_cols.append(col)

        if not filename_cols or not applicationid_cols:
            logger.warning("⚠️  Не найдены нужные колонки в сводных данных для связки")
            logger.info(f"Доступные колонки: {list(svod_data.columns)}")
            return audio_metadata

        # Используем первые найденные колонки
        filename_col = filename_cols[0]
        applicationid_col = applicationid_cols[0]

        logger.info(f"🔑 Связываем через колонки: {filename_col} → {applicationid_col}")

        # Создаем маппинг имя_файла → APPLICATIONID
        file_to_app_mapping = {}

        for _, row in svod_data.iterrows():
            filename = row[filename_col]
            app_id = row[applicationid_col]

            if pd.notna(filename) and pd.notna(app_id):
                # Очищаем имя файла от пути и расширения
                clean_filename = str(filename).split('/')[-1].split('\\')[-1]
                file_to_app_mapping[clean_filename] = str(app_id)

        logger.info(f"📋 Создан маппинг для {len(file_to_app_mapping)} файлов")

        # Применяем маппинг к аудио метаданным
        audio_metadata['applicationid'] = audio_metadata['original_filename'].map(file_to_app_mapping)

        # Статистика связывания
        linked_count = audio_metadata['applicationid'].notna().sum()
        total_count = len(audio_metadata)

        logger.info(f"✅ Связано аудиофайлов: {linked_count}/{total_count} ({linked_count/total_count:.1%})")

        if linked_count == 0:
            logger.error("❌ Ни один аудиофайл не был связан с APPLICATIONID!")
            logger.info("💡 Проверьте соответствие имен файлов в сводных данных и директории audiofiles")

        return audio_metadata

    def create_test_prediction_data(self, output_dir: str):
        """
        Создание тестовых данных для предсказания (без целевых меток)

        Args:
            output_dir: Директория для сохранения тестовых данных
        """
        logger.info("🧪 Создание тестовых данных для предсказания...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Загружаем только признаки (без target_data)
        amplitude_data = self.load_amplitude_chunks()
        app_data = self.load_app_data()
        audio_metadata = self.get_audio_files_metadata()

        # Объединяем без целевых меток
        merged_data = amplitude_data.copy()

        if not app_data.empty:
            merge_key = self._find_common_column(amplitude_data, app_data)
            if merge_key:
                merged_data = pd.merge(merged_data, app_data, on=merge_key, how='left')

        if not audio_metadata.empty:
            merge_key = self._find_common_column(amplitude_data, audio_metadata)
            if merge_key:
                merged_data = pd.merge(merged_data, audio_metadata, on=merge_key, how='left')

        # Очищаем данные
        merged_data = self._clean_merged_data(merged_data)

        # Сохраняем для предсказания
        prediction_file = output_path / "prediction_data.parquet"
        merged_data.to_parquet(prediction_file)

        logger.info(f"✅ Тестовые данные сохранены: {prediction_file}")
        logger.info(f"📊 Размер данных для предсказания: {merged_data.shape}")

    def _find_common_column(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Optional[str]:
        """
        Поиск общей колонки между двумя DataFrame
        """
        common_cols = set(df1.columns) & set(df2.columns)

        # Приоритетные колонки для объединения
        priority_cols = ['session_id', 'ID', 'id', 'APPLICATIONID']

        for col in priority_cols:
            if col in common_cols:
                return col

        # Возвращаем первую найденную общую колонку
        return list(common_cols)[0] if common_cols else None
