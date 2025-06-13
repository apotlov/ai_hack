"""
Модуль обучения и предсказания модели машинного обучения
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Класс для обучения и предсказания антифрод модели
    """

    def __init__(self, model_dir: str = "models"):
        """
        Инициализация тренера модели

        Args:
            model_dir: Директория для сохранения моделей
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        # Инициализируем модель (Random Forest для MVP)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # Препроцессоры
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

        # Метаданные модели
        self.feature_names = []
        self.is_trained = False
        self.model_metrics = {}

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict:
        """
        Обучение модели

        Args:
            X: Признаки
            y: Целевая переменная
            test_size: Размер тестовой выборки

        Returns:
            Словарь с метриками обучения
        """
        logger.info("Начинаем обучение модели...")

        if X.empty or y.empty:
            raise ValueError("Пустые данные для обучения")

        if len(X) != len(y):
            raise ValueError("Размеры X и y не совпадают")

        # Сохраняем названия признаков
        self.feature_names = list(X.columns)

        # Предобработка данных
        X_processed = self._preprocess_features(X, fit=True)

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42, stratify=y
        )

        logger.info(f"Размер обучающей выборки: {X_train.shape}")
        logger.info(f"Размер тестовой выборки: {X_test.shape}")
        logger.info(f"Распределение классов в обучающей выборке: {y_train.value_counts().to_dict()}")

        # Обучение модели
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Предсказания
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        y_pred_proba_test = self.model.predict_proba(X_test)[:, 1]

        # Вычисление метрик
        metrics = self._calculate_metrics(
            y_train, y_pred_train, y_test, y_pred_test, y_pred_proba_test
        )

        # Кросс-валидация
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='roc_auc')
        metrics['cv_auc_mean'] = cv_scores.mean()
        metrics['cv_auc_std'] = cv_scores.std()

        self.model_metrics = metrics

        logger.info("Обучение завершено!")
        logger.info(f"Test AUC: {metrics['test_auc']:.4f}")
        logger.info(f"CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание классов

        Args:
            X: Признаки для предсказания

        Returns:
            Массив предсказанных классов
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите train()")

        if X.empty:
            return np.array([])

        # Проверяем соответствие признаков
        if list(X.columns) != self.feature_names:
            logger.warning("Признаки не совпадают с обученными")
            # Приводим к нужному формату
            X = self._align_features(X)

        # Предобработка
        X_processed = self._preprocess_features(X, fit=False)

        # Предсказание
        predictions = self.model.predict(X_processed)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание вероятностей классов

        Args:
            X: Признаки для предсказания

        Returns:
            Массив вероятностей для каждого класса
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите train()")

        if X.empty:
            return np.array([])

        # Проверяем соответствие признаков
        if list(X.columns) != self.feature_names:
            X = self._align_features(X)

        # Предобработка
        X_processed = self._preprocess_features(X, fit=False)

        # Предсказание вероятностей
        probabilities = self.model.predict_proba(X_processed)

        return probabilities

    def _preprocess_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Предобработка признаков

        Args:
            X: Признаки
            fit: Нужно ли обучать препроцессоры

        Returns:
            Обработанные признаки
        """
        # Работаем с копией
        X_copy = X.copy()

        # Заполнение пропусков
        if fit:
            X_processed = self.imputer.fit_transform(X_copy)
        else:
            X_processed = self.imputer.transform(X_copy)

        # Стандартизация
        if fit:
            X_processed = self.scaler.fit_transform(X_processed)
        else:
            X_processed = self.scaler.transform(X_processed)

        return X_processed

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Приведение признаков к обученному формату

        Args:
            X: Признаки для выравнивания

        Returns:
            Выровненные признаки
        """
        # Добавляем отсутствующие признаки
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0

        # Удаляем лишние признаки и сортируем
        X = X[self.feature_names]

        return X

    def _calculate_metrics(self, y_train: pd.Series, y_pred_train: np.ndarray,
                          y_test: pd.Series, y_pred_test: np.ndarray,
                          y_pred_proba_test: np.ndarray) -> Dict:
        """
        Вычисление метрик модели

        Args:
            y_train: Истинные метки обучающей выборки
            y_pred_train: Предсказания обучающей выборки
            y_test: Истинные метки тестовой выборки
            y_pred_test: Предсказания тестовой выборки
            y_pred_proba_test: Вероятности тестовой выборки

        Returns:
            Словарь с метриками
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            # Метрики обучающей выборки
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'train_precision': precision_score(y_train, y_pred_train, zero_division=0),
            'train_recall': recall_score(y_train, y_pred_train, zero_division=0),
            'train_f1': f1_score(y_train, y_pred_train, zero_division=0),

            # Метрики тестовой выборки
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
            'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
            'test_f1': f1_score(y_test, y_pred_test, zero_division=0),
            'test_auc': roc_auc_score(y_test, y_pred_proba_test),

            # Размеры выборок
            'train_size': len(y_train),
            'test_size': len(y_test),
            'fraud_rate': y_train.mean()
        }

        return metrics

    def save_model(self, model_name: str = "antifraud_model"):
        """
        Сохранение обученной модели

        Args:
            model_name: Имя файла модели
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")

        model_path = self.model_dir / f"{model_name}.joblib"

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'metrics': self.model_metrics,
            'is_trained': self.is_trained
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Модель сохранена: {model_path}")

    def load_model(self, model_name: str = "antifraud_model"):
        """
        Загрузка сохраненной модели

        Args:
            model_name: Имя файла модели
        """
        model_path = self.model_dir / f"{model_name}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        model_data = joblib.load(model_path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.imputer = model_data['imputer']
        self.feature_names = model_data['feature_names']
        self.model_metrics = model_data.get('metrics', {})
        self.is_trained = model_data.get('is_trained', True)

        logger.info(f"Модель загружена: {model_path}")

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Получение важности признаков

        Args:
            top_n: Количество топ признаков

        Returns:
            DataFrame с важностью признаков
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")

        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Модель не поддерживает важность признаков")
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })

        importance_df = importance_df.sort_values('importance', ascending=False)

        if top_n:
            importance_df = importance_df.head(top_n)

        return importance_df

    def get_model_info(self) -> Dict:
        """
        Получение информации о модели

        Returns:
            Словарь с информацией о модели
        """
        info = {
            'is_trained': self.is_trained,
            'model_type': type(self.model).__name__,
            'feature_count': len(self.feature_names),
            'model_params': self.model.get_params(),
            'metrics': self.model_metrics
        }

        return info

    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Оценка модели на новых данных

        Args:
            X: Признаки
            y: Истинные метки

        Returns:
            Словарь с метриками
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")

        # Предсказания
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]

        # Вычисление метрик
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_pred_proba),
            'samples_count': len(y),
            'fraud_rate': y.mean()
        }

        return metrics
