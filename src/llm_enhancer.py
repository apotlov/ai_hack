"""
Модуль интеграции LLM для улучшенной детекции мошенничества
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from datetime import datetime
import openai
from dataclasses import dataclass
import re

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


class LLMFraudEnhancer:
    """
    Класс для интеграции LLM в антифрод систему
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Инициализация LLM енхансера

        Args:
            api_key: OpenAI API ключ
            model: Модель для использования
        """
        self.model = model
        self.api_key = api_key

        if api_key:
            openai.api_key = api_key

        # Настройки для разных типов запросов
        self.explanation_prompts = {
            "high_risk": """
Проанализируй данные пользователя и объясни, почему транзакция классифицирована как высокий риск мошенничества.

Данные пользователя:
{user_data}

Признаки модели:
{features}

Вероятность мошенничества: {probability:.1%}

Предоставь структурированное объяснение:
1. Основные подозрительные факторы
2. Отклонения от нормального поведения
3. Рекомендуемые действия
4. Уровень уверенности в оценке

Ответ должен быть понятен банковскому аналитику.
""",
            "pattern_analysis": """
Проанализируй паттерны мошенничества в данных и выдели основные тренды.

Данные о мошенничестве:
{fraud_cases}

Задачи:
1. Выдели общие паттерны мошенников
2. Определи временные тренды
3. Найди географические аномалии
4. Предложи улучшения для модели

Формат ответа: структурированный анализ с конкретными инсайтами.
"""
        }

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

            # Получаем объяснение от LLM
            if self.api_key:
                explanation_text = self._call_llm(prompt)
            else:
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

        if 'session_count' in user_data:
            formatted.append(f"Количество сессий: {user_data['session_count']}")

        if 'avg_session_duration' in user_data:
            formatted.append(f"Средняя длительность сессии: {user_data['avg_session_duration']:.1f} сек")

        if 'night_activity_ratio' in user_data:
            formatted.append(f"Активность ночью: {user_data['night_activity_ratio']:.1%}")

        return "\n".join(formatted)

    def _format_top_features(self, features: Dict, top_n: int = 10) -> str:
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
                formatted.append(f"{feature}: {value:.3f}")
            else:
                formatted.append(f"{feature}: {value}")

        return "\n".join(formatted)

    def _call_llm(self, prompt: str) -> str:
        """Вызов LLM API"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты эксперт по банковской безопасности и анализу мошенничества."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Ошибка при вызове LLM: {e}")
            return "Не удалось получить объяснение от LLM"

    def _generate_rule_based_explanation(self, user_data: Dict, features: Dict,
                                       probability: float) -> str:
        """Генерация объяснения на основе правил (fallback)"""
        explanations = []

        # Анализируем ключевые признаки
        if 'night_activity_ratio' in features and features['night_activity_ratio'] > 0.3:
            explanations.append("Высокая активность в ночное время (подозрительно)")

        if 'session_count' in features and features['session_count'] < 5:
            explanations.append("Низкое количество сессий (нетипичное поведение)")

        if 'avg_session_duration' in features and features['avg_session_duration'] < 60:
            explanations.append("Очень короткие сессии (автоматизированное поведение)")

        if probability > 0.8:
            explanations.append("Крайне высокая вероятность мошенничества")
        elif probability > 0.5:
            explanations.append("Повышенная вероятность мошенничества")

        if not explanations:
            explanations.append("Модель выявила аномальные паттерны поведения")

        return ". ".join(explanations) + "."

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

        # Анализируем топ признаки
        if not features:
            return ["Недостаточно данных для анализа"]

        # Сортируем по важности
        sorted_features = sorted(
            features.items(),
            key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
            reverse=True
        )

        for feature, value in sorted_features[:5]:
            if isinstance(value, (int, float)):
                if abs(value) > 0.1:  # Порог значимости
                    factor_desc = self._describe_feature(feature, value)
                    if factor_desc:
                        factors.append(factor_desc)

        if not factors:
            factors.append(f"Общая модель уверенности: {probability:.1%}")

        return factors

    def _describe_feature(self, feature_name: str, value: float) -> Optional[str]:
        """Описание признака на человеческом языке"""
        descriptions = {
            'night_activity_ratio': f"Ночная активность: {value:.1%}",
            'session_count': f"Количество сессий: {value:.0f}",
            'avg_session_duration': f"Средняя длительность сессии: {value:.0f} сек",
            'unique_event_types': f"Разнообразие действий: {value:.0f}",
            'weekend_ratio': f"Активность в выходные: {value:.1%}",
            'mfcc_0_mean': f"Аудио признак (тон голоса): {value:.2f}",
            'spectral_centroid_mean': f"Аудио признак (качество речи): {value:.0f}",
        }

        # Ищем частичные совпадения
        for key, desc in descriptions.items():
            if key in feature_name:
                return desc

        # Общее описание
        if 'amplitude' in feature_name:
            return f"Поведенческий признак: {value:.2f}"
        elif 'audio' in feature_name:
            return f"Аудио признак: {value:.2f}"

        return None

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
        confidence = 0.5  # Базовая уверенность

        # Увеличиваем уверенность при крайних значениях
        if probability > 0.8 or probability < 0.2:
            confidence += 0.3

        # Увеличиваем при наличии важных признаков
        if features:
            important_features = sum(
                1 for value in features.values()
                if isinstance(value, (int, float)) and abs(value) > 0.1
            )
            confidence += min(important_features * 0.05, 0.2)

        return min(confidence, 1.0)

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
        Генерация отчета о мошенничестве

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

            # Генерируем отчет
            if self.api_key:
                prompt = self.explanation_prompts["pattern_analysis"].format(
                    fraud_cases=json.dumps(patterns, ensure_ascii=False, indent=2)
                )
                report = self._call_llm(prompt)
            else:
                report = self._generate_rule_based_report(patterns)

            return report

        except Exception as e:
            logger.error(f"Ошибка при генерации отчета: {e}")
            return f"Обнаружено {len(fraud_cases)} случаев подозрительной активности. Требуется детальный анализ."

    def _analyze_fraud_patterns(self, fraud_cases: List[Dict]) -> Dict:
        """Анализ паттернов мошенничества"""
        patterns = {
            "total_cases": len(fraud_cases),
            "avg_probability": np.mean([case.get('probability', 0) for case in fraud_cases]),
            "high_risk_count": sum(1 for case in fraud_cases if case.get('probability', 0) > 0.7),
            "time_patterns": {},
            "common_features": {}
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
        report_parts.append("")

        report_parts.append("Рекомендации:")
        if patterns['high_risk_count'] > 0:
            report_parts.append("- Немедленно проверить случаи высокого риска")
        report_parts.append("- Усилить мониторинг подозрительных операций")
        report_parts.append("- Обновить правила детекции")

        return "\n".join(report_parts)

    def enhance_predictions_with_explanations(self, predictions_df: pd.DataFrame,
                                           features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление объяснений к предсказаниям

        Args:
            predictions_df: DataFrame с предсказаниями
            features_df: DataFrame с признаками

        Returns:
            DataFrame с добавленными объяснениями
        """
        enhanced_df = predictions_df.copy()

        explanations = []
        key_factors = []
        recommendations = []

        for idx, row in predictions_df.iterrows():
            try:
                # Получаем данные пользователя
                user_data = {'user_id': row.get('user_id', f'user_{idx}')}

                # Получаем признаки пользователя
                if idx < len(features_df):
                    features = features_df.iloc[idx].to_dict()
                else:
                    features = {}

                # Генерируем объяснение
                explanation = self.explain_fraud_decision(
                    user_data, features, row['fraud_probability']
                )

                explanations.append(explanation.explanation)
                key_factors.append("; ".join(explanation.key_factors))
                recommendations.append("; ".join(explanation.recommendations))

            except Exception as e:
                logger.error(f"Ошибка при обработке строки {idx}: {e}")
                explanations.append("Не удалось сгенерировать объяснение")
                key_factors.append("Данные недоступны")
                recommendations.append("Стандартная проверка")

        # Добавляем новые колонки
        enhanced_df['explanation'] = explanations
        enhanced_df['key_factors'] = key_factors
        enhanced_df['recommendations'] = recommendations

        return enhanced_df


class MockLLMEnhancer(LLMFraudEnhancer):
    """
    Mock версия LLM енхансера для демонстрации без API ключа
    """

    def __init__(self):
        super().__init__(api_key=None)
        logger.info("Инициализирован Mock LLM енхансер (без реального API)")

    def _call_llm(self, prompt: str) -> str:
        """Mock версия вызова LLM"""
        return "Это демонстрационное объяснение. Для полноценной работы нужен API ключ OpenAI."
