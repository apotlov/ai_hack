"""
Модуль интеграции локальных LLM моделей через Ollama для антифрод системы
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from datetime import datetime
from dataclasses import dataclass
import requests
import time
import re
import subprocess
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FraudExplanation:
    """
    Структура объяснения мошенничества
    """
    user_id: str
    fraud_probability: float
    risk_level: str
    explanation: str
    key_factors: List[str]
    recommendations: List[str]
    confidence: float


class LocalLLMEnhancer:
    """
    Класс для интеграции локальных LLM через Ollama
    """

    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        """
        Инициализация локального LLM енхансера

        Args:
            model: Название модели (llama3.2:3b, phi3.5, qwen2.5:7b)
            base_url: URL Ollama сервера
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

        # Проверяем доступность Ollama
        self._check_ollama_status()

        # Убеждаемся что модель загружена
        self._ensure_model_loaded()

        # Настройки для разных типов запросов
        self.explanation_prompts = {
            "high_risk": """Ты эксперт по банковской безопасности. Проанализируй данные и объясни, почему транзакция классифицирована как мошенническая.

Данные пользователя:
{user_data}

Признаки модели:
{features}

Вероятность мошенничества: {probability:.1%}

Дай структурированный ответ:
1. Основные подозрительные факторы (2-3 пункта)
2. Отклонения от нормального поведения
3. Рекомендуемые действия (конкретные)
4. Уровень уверенности

Ответ должен быть понятен банковскому аналитику. Максимум 150 слов.""",

            "pattern_analysis": """Проанализируй паттерны мошенничества в банковских данных.

Данные о мошенничестве:
{fraud_cases}

Задачи:
1. Выяви 3 основных паттерна мошенников
2. Определи временные тренды
3. Найди аномалии в поведении
4. Предложи 2-3 улучшения для модели

Формат: структурированный анализ, максимум 200 слов."""
        }

    def _check_ollama_status(self) -> bool:
        """Проверка статуса Ollama сервера"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Ollama сервер доступен")
                return True
            else:
                logger.warning(f"⚠️  Ollama сервер недоступен: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Не удается подключиться к Ollama: {e}")
            logger.info("💡 Запустите Ollama: ollama serve")
            return False

    def _ensure_model_loaded(self):
        """Убеждаемся что модель загружена"""
        try:
            # Проверяем список доступных моделей
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model['name'] for model in models_data.get('models', [])]

                if self.model not in available_models:
                    logger.warning(f"⚠️  Модель {self.model} не найдена")
                    logger.info(f"📋 Доступные модели: {available_models}")

                    # Предлагаем загрузить модель
                    logger.info(f"💡 Для загрузки модели выполните: ollama pull {self.model}")

                    # Пробуем автоматически загрузить
                    self._auto_pull_model()
                else:
                    logger.info(f"✅ Модель {self.model} доступна")

        except Exception as e:
            logger.error(f"Ошибка при проверке модели: {e}")

    def _auto_pull_model(self):
        """Автоматическая загрузка модели"""
        try:
            logger.info(f"🔄 Пытаемся загрузить модель {self.model}...")

            # Запускаем загрузку модели
            result = subprocess.run(['ollama', 'pull', self.model],
                                  capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                logger.info(f"✅ Модель {self.model} успешно загружена")
            else:
                logger.error(f"❌ Не удалось загрузить модель: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.warning("⏰ Загрузка модели заняла слишком много времени")
        except FileNotFoundError:
            logger.error("❌ Ollama CLI не найден. Установите Ollama: https://ollama.ai")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")

    def _call_local_llm(self, prompt: str, max_tokens: int = 300) -> str:
        """Вызов локальной LLM"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            }

            response = requests.post(self.api_url, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"Ошибка LLM API: {response.status_code}")
                return "Не удалось получить ответ от локальной модели"

        except requests.exceptions.Timeout:
            logger.error("Таймаут при запросе к LLM")
            return "Таймаут при обращении к модели"
        except Exception as e:
            logger.error(f"Ошибка при вызове локальной LLM: {e}")
            return "Ошибка при обращении к локальной модели"

    def explain_fraud_decision(self, user_data: Dict, features: Dict,
                             probability: float) -> FraudExplanation:
        """
        Генерация объяснения решения о мошенничестве

        Args:
            user_data: Данные пользователя
            features: Признаки модели
            probability: Вероятность мошенничества

        Returns:
            Объяснение решения
        """
        try:
            # Определяем уровень риска
            risk_level = self._determine_risk_level(probability)

            # Подготавливаем промпт
            prompt = self._prepare_explanation_prompt(user_data, features, probability)

            # Получаем объяснение от локальной LLM
            explanation_text = self._call_local_llm(prompt)

            # Если LLM недоступна, используем правила
            if "не удалось" in explanation_text.lower() or "ошибка" in explanation_text.lower():
                explanation_text = self._generate_rule_based_explanation(
                    user_data, features, probability
                )

            # Парсим ответ
            explanation = self._parse_explanation(explanation_text)

            # Извлекаем ключевые факторы
            key_factors = self._extract_key_factors(features, probability)

            # Генерируем рекомендации
            recommendations = self._generate_recommendations(risk_level, probability)

            return FraudExplanation(
                user_id=user_data.get('user_id', 'unknown'),
                fraud_probability=probability,
                risk_level=risk_level,
                explanation=explanation,
                key_factors=key_factors,
                recommendations=recommendations,
                confidence=self._calculate_confidence(features, probability)
            )

        except Exception as e:
            logger.error(f"Ошибка при генерации объяснения: {e}")
            return self._fallback_explanation(user_data, probability)

    def _determine_risk_level(self, probability: float) -> str:
        """Определение уровня риска"""
        if probability >= 0.7:
            return "Высокий"
        elif probability >= 0.3:
            return "Средний"
        else:
            return "Низкий"

    def _prepare_explanation_prompt(self, user_data: Dict, features: Dict,
                                  probability: float) -> str:
        """Подготовка промпта для LLM"""

        # Форматируем данные пользователя
        user_summary = self._format_user_data(user_data)

        # Форматируем топ признаки
        top_features = self._format_top_features(features)

        return self.explanation_prompts["high_risk"].format(
            user_data=user_summary,
            features=top_features,
            probability=probability
        )

    def _format_user_data(self, user_data: Dict) -> str:
        """Форматирование данных пользователя для промпта"""
        formatted = []

        if 'user_id' in user_data:
            formatted.append(f"ID пользователя: {user_data['user_id']}")

        # Добавляем доступную информацию
        for key, value in user_data.items():
            if key != 'user_id' and value is not None:
                formatted.append(f"{key}: {value}")

        return "\n".join(formatted) if formatted else "Базовая информация о пользователе"

    def _format_top_features(self, features: Dict, top_n: int = 8) -> str:
        """Форматирование топ признаков"""
        if not features:
            return "Признаки не доступны"

        # Сортируем признаки по абсолютному значению
        sorted_features = sorted(
            features.items(),
            key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
            reverse=True
        )

        formatted = []
        for feature, value in sorted_features[:top_n]:
            if isinstance(value, (int, float)):
                # Переводим технические названия в понятные
                readable_name = self._make_feature_readable(feature)
                formatted.append(f"{readable_name}: {value:.3f}")
            else:
                formatted.append(f"{feature}: {value}")

        return "\n".join(formatted)

    def _make_feature_readable(self, feature_name: str) -> str:
        """Перевод технических названий признаков в понятные"""
        translations = {
            'amplitude_night_activity_ratio': 'Активность ночью (%)',
            'amplitude_session_count': 'Количество сессий',
            'amplitude_avg_session_duration': 'Средняя длительность сессии (сек)',
            'amplitude_unique_event_types': 'Разнообразие действий',
            'amplitude_weekend_ratio': 'Активность в выходные (%)',
            'audio_mfcc_0_mean': 'Тон голоса (аудио)',
            'audio_spectral_centroid_mean': 'Качество речи (аудио)',
            'audio_rms_mean': 'Громкость голоса',
            'audio_duration': 'Длительность записи (сек)'
        }

        # Ищем точное совпадение
        if feature_name in translations:
            return translations[feature_name]

        # Ищем частичные совпадения
        for key, translation in translations.items():
            if key in feature_name:
                return translation

        # Упрощаем техническое название
        simplified = feature_name.replace('_', ' ').replace('amplitude', 'поведение').replace('audio', 'аудио')
        return simplified.title()

    def _parse_explanation(self, explanation_text: str) -> str:
        """Парсинг объяснения от LLM"""
        # Простая очистка текста
        cleaned = explanation_text.strip()

        # Убираем лишние символы
        cleaned = re.sub(r'\n+', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Ограничиваем длину
        if len(cleaned) > 500:
            cleaned = cleaned[:497] + "..."

        return cleaned

    def _extract_key_factors(self, features: Dict, probability: float) -> List[str]:
        """Извлечение ключевых факторов"""
        factors = []

        if not features:
            return [f"Общая модель уверенности: {probability:.1%}"]

        # Сортируем по важности
        sorted_features = sorted(
            features.items(),
            key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
            reverse=True
        )

        for feature, value in sorted_features[:5]:
            if isinstance(value, (int, float)) and abs(value) > 0.01:
                readable_name = self._make_feature_readable(feature)
                factors.append(f"{readable_name}: {value:.2f}")

        if not factors:
            factors.append(f"Общая оценка модели: {probability:.1%}")

        return factors

    def _generate_recommendations(self, risk_level: str, probability: float) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []

        if risk_level == "Высокий":
            recommendations.extend([
                "Немедленно заблокировать подозрительные операции",
                "Связаться с клиентом для подтверждения",
                "Провести дополнительную проверку документов",
                "Уведомить службу безопасности"
            ])
        elif risk_level == "Средний":
            recommendations.extend([
                "Усилить мониторинг операций пользователя",
                "Запросить дополнительную аутентификацию",
                "Проверить операции за последние 24 часа"
            ])
        else:
            recommendations.extend([
                "Продолжить стандартный мониторинг",
                "Периодически пересматривать статус"
            ])

        return recommendations

    def _calculate_confidence(self, features: Dict, probability: float) -> float:
        """Расчет уверенности в предсказании"""
        confidence = 0.6  # Базовая уверенность для локальных моделей

        # Увеличиваем уверенность при крайних значениях
        if probability > 0.8 or probability < 0.2:
            confidence += 0.2

        # Увеличиваем при наличии важных признаков
        if features:
            important_features = sum(
                1 for value in features.values()
                if isinstance(value, (int, float)) and abs(value) > 0.1
            )
            confidence += min(important_features * 0.03, 0.2)

        return min(confidence, 1.0)

    def _generate_rule_based_explanation(self, user_data: Dict, features: Dict,
                                       probability: float) -> str:
        """Генерация объяснения на основе правил (fallback)"""
        explanations = []

        # Анализируем ключевые признаки
        for feature, value in features.items():
            if not isinstance(value, (int, float)):
                continue

            if 'night_activity' in feature and value > 0.3:
                explanations.append(f"Высокая ночная активность ({value:.1%})")

            if 'session_count' in feature and value < 5:
                explanations.append(f"Малое количество сессий ({value:.0f})")

            if 'session_duration' in feature and value < 60:
                explanations.append(f"Короткие сессии ({value:.0f} сек)")

        if probability > 0.8:
            explanations.append("Крайне высокая вероятность мошенничества")
        elif probability > 0.5:
            explanations.append("Повышенная вероятность мошенничества")

        if not explanations:
            explanations.append("Модель выявила аномальные паттерны поведения")

        return ". ".join(explanations) + "."

    def _fallback_explanation(self, user_data: Dict, probability: float) -> FraudExplanation:
        """Fallback объяснение при ошибках"""
        return FraudExplanation(
            user_id=user_data.get('user_id', 'unknown'),
            fraud_probability=probability,
            risk_level=self._determine_risk_level(probability),
            explanation=f"Модель определила вероятность мошенничества {probability:.1%} на основе анализа поведенческих паттернов.",
            key_factors=[f"Общая оценка модели: {probability:.1%}"],
            recommendations=self._generate_recommendations(self._determine_risk_level(probability), probability),
            confidence=0.5
        )

    def generate_fraud_report(self, fraud_cases: List[Dict]) -> str:
        """
        Генерация отчета о мошенничестве с помощью локальной LLM

        Args:
            fraud_cases: Список случаев мошенничества

        Returns:
            Текстовый отчет
        """
        if not fraud_cases:
            return "Случаев мошенничества не обнаружено."

        try:
            # Анализируем паттерны
            patterns = self._analyze_fraud_patterns(fraud_cases)

            # Генерируем отчет с помощью локальной LLM
            prompt = self.explanation_prompts["pattern_analysis"].format(
                fraud_cases=json.dumps(patterns, ensure_ascii=False, indent=2)
            )

            llm_report = self._call_local_llm(prompt, max_tokens=400)

            # Если LLM недоступна, используем правила
            if "не удалось" in llm_report.lower() or "ошибка" in llm_report.lower():
                llm_report = self._generate_rule_based_report(patterns)

            return llm_report

        except Exception as e:
            logger.error(f"Ошибка при генерации отчета: {e}")
            return f"Обнаружено {len(fraud_cases)} случаев подозрительной активности. Требуется детальный анализ."

    def _analyze_fraud_patterns(self, fraud_cases: List[Dict]) -> Dict:
        """Анализ паттернов мошенничества"""
        patterns = {
            "total_cases": len(fraud_cases),
            "avg_probability": np.mean([case.get('probability', 0) for case in fraud_cases]),
            "high_risk_count": sum(1 for case in fraud_cases if case.get('probability', 0) > 0.7),
            "risk_distribution": {
                "high": sum(1 for case in fraud_cases if case.get('probability', 0) > 0.7),
                "medium": sum(1 for case in fraud_cases if 0.3 <= case.get('probability', 0) <= 0.7),
                "low": sum(1 for case in fraud_cases if case.get('probability', 0) < 0.3)
            }
        }

        return patterns

    def _generate_rule_based_report(self, patterns: Dict) -> str:
        """Генерация отчета на основе правил"""
        report_parts = []

        report_parts.append(f"ОТЧЕТ О МОШЕННИЧЕСТВЕ")
        report_parts.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report_parts.append("")

        report_parts.append(f"Общая статистика:")
        report_parts.append(f"- Всего случаев: {patterns['total_cases']}")
        report_parts.append(f"- Средняя вероятность: {patterns['avg_probability']:.1%}")
        report_parts.append(f"- Высокий риск: {patterns['high_risk_count']}")

        if 'risk_distribution' in patterns:
            dist = patterns['risk_distribution']
            report_parts.append(f"- Распределение: Высокий({dist['high']}) Средний({dist['medium']}) Низкий({dist['low']})")

        report_parts.append("")

        report_parts.append("Основные паттерны:")
        if patterns['high_risk_count'] > patterns['total_cases'] * 0.3:
            report_parts.append("- Высокая концентрация случаев высокого риска")
        report_parts.append("- Требуется анализ временных паттернов")
        report_parts.append("- Рекомендуется проверка географических аномалий")
        report_parts.append("")

        report_parts.append("Рекомендации:")
        if patterns['high_risk_count'] > 0:
            report_parts.append("- Немедленно проверить случаи высокого риска")
        report_parts.append("- Усилить мониторинг подозрительных операций")
        report_parts.append("- Обновить правила детекции")

        return "\n".join(report_parts)

    def get_model_info(self) -> Dict:
        """Получение информации о локальной модели"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                for model in models_data.get('models', []):
                    if model['name'] == self.model:
                        return {
                            'name': model['name'],
                            'size': model.get('size', 'unknown'),
                            'modified_at': model.get('modified_at', 'unknown'),
                            'status': 'available'
                        }
            return {'name': self.model, 'status': 'not_found'}
        except:
            return {'name': self.model, 'status': 'ollama_unavailable'}

    def test_connection(self) -> bool:
        """Тестирование соединения с локальной моделью"""
        try:
            test_prompt = "Скажи 'тест прошел успешно' если ты работаешь."
            response = self._call_local_llm(test_prompt, max_tokens=50)

            if response and "тест прошел" in response.lower():
                logger.info("✅ Локальная LLM работает корректно")
                return True
            else:
                logger.warning(f"⚠️  Получен неожиданный ответ: {response}")
                return False

        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании: {e}")
            return False


class OllamaInstaller:
    """
    Помощник для установки и настройки Ollama
    """

    @staticmethod
    def check_ollama_installed() -> bool:
        """Проверка установки Ollama"""
        try:
            result = subprocess.run(['ollama', '--version'],
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    @staticmethod
    def get_installation_instructions() -> str:
        """Инструкции по установке Ollama"""
        return """
🚀 УСТАНОВКА OLLAMA ДЛЯ ЛОКАЛЬНЫХ LLM

1. Установите Ollama:
   macOS/Linux: curl -fsSL https://ollama.ai/install.sh | sh
   Windows: Скачайте с https://ollama.ai/download

2. Запустите сервер:
   ollama serve

3. Загрузите рекомендуемую модель:
   ollama pull llama3.2:3b

4. Альтернативные модели:
   ollama pull phi3.5        # Быстрая модель (4GB)
   ollama pull qwen2.5:7b    # Качественная модель (8GB)

5. Проверьте статус:
   ollama list

Готово! Локальная LLM будет работать без интернета.
"""

    @staticmethod
    def recommend_model_by_resources() -> str:
        """Рекомендация модели по ресурсам"""
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024**3)

        if ram_gb >= 16:
            return "qwen2.5:7b (лучшее качество, 8GB)"
        elif ram_gb >= 8:
            return "llama3.2:3b (оптимальный выбор, 6GB)"
        else:
            return "phi3.5 (экономичный вариант, 4GB)"
