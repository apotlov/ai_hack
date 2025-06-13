#!/usr/bin/env python3
"""
Расширенный скрипт предсказания с LLM объяснениями
"""

import sys
import os
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent / "src"))

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json

from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
from llm_enhancer import LLMFraudEnhancer, MockLLMEnhancer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_llm_enhancer(api_key: Optional[str] = None) -> LLMFraudEnhancer:
    """
    Загрузка LLM енхансера

    Args:
        api_key: OpenAI API ключ (опционально)

    Returns:
        Инициализированный LLM енхансер
    """
    if api_key:
        logger.info("🤖 Инициализация LLM с OpenAI API")
        return LLMFraudEnhancer(api_key=api_key)
    else:
        logger.info("🤖 Инициализация Mock LLM (демо режим)")
        return MockLLMEnhancer()


def enhance_predictions_with_explanations(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
    llm_enhancer: LLMFraudEnhancer
) -> pd.DataFrame:
    """
    Добавление LLM объяснений к предсказаниям

    Args:
        predictions_df: DataFrame с предсказаниями
        features_df: DataFrame с признаками
        llm_enhancer: LLM енхансер

    Returns:
        DataFrame с объяснениями
    """
    logger.info("🧠 Генерация объяснений с помощью LLM...")

    enhanced_df = predictions_df.copy()

    # Списки для новых колонок
    explanations = []
    key_factors_list = []
    recommendations_list = []
    confidence_scores = []

    # Обрабатываем каждую строку
    for idx, row in predictions_df.iterrows():
        try:
            # Подготавливаем данные пользователя
            user_data = {
                'user_id': row.get('user_id', f'user_{idx}'),
                'sample_id': row.get('sample_id', idx)
            }

            # Получаем признаки пользователя
            if idx < len(features_df):
                features = features_df.iloc[idx].to_dict()
                # Убираем NaN значения
                features = {k: v for k, v in features.items() if pd.notna(v)}
            else:
                features = {}

            # Генерируем объяснение
            fraud_explanation = llm_enhancer.explain_fraud_decision(
                user_data=user_data,
                features=features,
                probability=row['fraud_probability']
            )

            # Добавляем результаты
            explanations.append(fraud_explanation.explanation)
            key_factors_list.append(" | ".join(fraud_explanation.key_factors))
            recommendations_list.append(" | ".join(fraud_explanation.recommendations))
            confidence_scores.append(fraud_explanation.confidence)

            # Логируем прогресс
            if (idx + 1) % 10 == 0:
                logger.info(f"Обработано {idx + 1}/{len(predictions_df)} предсказаний")

        except Exception as e:
            logger.error(f"Ошибка при обработке строки {idx}: {e}")
            explanations.append("Не удалось сгенерировать объяснение")
            key_factors_list.append("Данные недоступны")
            recommendations_list.append("Требуется ручная проверка")
            confidence_scores.append(0.5)

    # Добавляем новые колонки
    enhanced_df['llm_explanation'] = explanations
    enhanced_df['key_factors'] = key_factors_list
    enhanced_df['recommendations'] = recommendations_list
    enhanced_df['explanation_confidence'] = confidence_scores

    logger.info("✅ Объяснения сгенерированы успешно")
    return enhanced_df


def generate_detailed_fraud_report(
    enhanced_predictions: pd.DataFrame,
    llm_enhancer: LLMFraudEnhancer
) -> str:
    """
    Генерация детального отчета о мошенничестве

    Args:
        enhanced_predictions: DataFrame с предсказаниями и объяснениями
        llm_enhancer: LLM енхансер

    Returns:
        Текстовый отчет
    """
    logger.info("📊 Генерация детального отчета...")

    # Подготавливаем данные для анализа
    fraud_cases = []
    for _, row in enhanced_predictions.iterrows():
        if row['fraud_probability'] > 0.3:  # Только подозрительные случаи
            fraud_cases.append({
                'user_id': row['user_id'],
                'probability': row['fraud_probability'],
                'risk_level': row['risk_level'],
                'explanation': row['llm_explanation'],
                'key_factors': row['key_factors']
            })

    # Генерируем отчет с помощью LLM
    llm_report = llm_enhancer.generate_fraud_report(fraud_cases)

    # Добавляем статистическую информацию
    total_cases = len(enhanced_predictions)
    high_risk_cases = len(enhanced_predictions[enhanced_predictions['fraud_probability'] >= 0.7])
    medium_risk_cases = len(enhanced_predictions[
        (enhanced_predictions['fraud_probability'] >= 0.3) &
        (enhanced_predictions['fraud_probability'] < 0.7)
    ])

    detailed_report = f"""
=== ДЕТАЛЬНЫЙ ОТЧЕТ АНТИФРОД СИСТЕМЫ С LLM ===
Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

СТАТИСТИКА:
- Всего проанализировано: {total_cases}
- Высокий риск (≥70%): {high_risk_cases}
- Средний риск (30-70%): {medium_risk_cases}
- Низкий риск (<30%): {total_cases - high_risk_cases - medium_risk_cases}

АНАЛИЗ ПАТТЕРНОВ (LLM):
{llm_report}

ТОП-5 САМЫХ ПОДОЗРИТЕЛЬНЫХ СЛУЧАЕВ:
"""

    # Добавляем топ случаи
    top_cases = enhanced_predictions.nlargest(5, 'fraud_probability')
    for i, (_, case) in enumerate(top_cases.iterrows(), 1):
        detailed_report += f"""
{i}. User {case['user_id']} - Риск: {case['fraud_probability']:.1%}
   Объяснение: {case['llm_explanation'][:200]}...
   Ключевые факторы: {case['key_factors'][:150]}...
   Рекомендации: {case['recommendations'][:150]}...
"""

    return detailed_report


def create_interactive_report(enhanced_predictions: pd.DataFrame, output_dir: Path):
    """
    Создание интерактивного отчета

    Args:
        enhanced_predictions: DataFrame с предсказаниями
        output_dir: Директория для сохранения
    """
    logger.info("📋 Создание интерактивного отчета...")

    # HTML отчет
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Антифрод Отчет с LLM</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}
        .risk-high {{ background-color: #ffebee; }}
        .risk-medium {{ background-color: #fff3e0; }}
        .risk-low {{ background-color: #e8f5e8; }}
        .case {{ margin: 10px 0; padding: 15px; border-radius: 5px; border: 1px solid #ddd; }}
        .explanation {{ font-style: italic; color: #555; }}
        .factors {{ color: #333; font-weight: bold; }}
        .recommendations {{ color: #0066cc; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Антифрод Отчет с LLM Объяснениями</h1>
        <p>Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Всего случаев: {len(enhanced_predictions)}</p>
    </div>

    <h2>📊 Статистика по уровням риска</h2>
    <ul>
        <li>Высокий риск: {len(enhanced_predictions[enhanced_predictions['fraud_probability'] >= 0.7])}</li>
        <li>Средний риск: {len(enhanced_predictions[(enhanced_predictions['fraud_probability'] >= 0.3) & (enhanced_predictions['fraud_probability'] < 0.7)])}</li>
        <li>Низкий риск: {len(enhanced_predictions[enhanced_predictions['fraud_probability'] < 0.3])}</li>
    </ul>

    <h2>🚨 Детальный анализ случаев</h2>
"""

    # Добавляем случаи
    for _, case in enhanced_predictions.head(20).iterrows():
        risk_class = "risk-high" if case['fraud_probability'] >= 0.7 else "risk-medium" if case['fraud_probability'] >= 0.3 else "risk-low"

        html_content += f"""
    <div class="case {risk_class}">
        <h3>👤 {case['user_id']} - Риск: {case['fraud_probability']:.1%} ({case['risk_level']})</h3>
        <p class="explanation"><strong>Объяснение:</strong> {case['llm_explanation']}</p>
        <p class="factors"><strong>Ключевые факторы:</strong> {case['key_factors']}</p>
        <p class="recommendations"><strong>Рекомендации:</strong> {case['recommendations']}</p>
        <p><small>Уверенность в объяснении: {case['explanation_confidence']:.1%}</small></p>
    </div>
"""

    html_content += """
</body>
</html>
"""

    # Сохраняем HTML
    html_file = output_dir / "fraud_report_with_llm.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"💾 Интерактивный отчет сохранен: {html_file}")


def main():
    """
    Основная функция предсказания с LLM
    """
    logger.info("🚀 Запуск расширенного предсказания с LLM")

    try:
        # Настройки
        data_dir = Path(__file__).parent.parent / "data"
        models_dir = Path(__file__).parent.parent / "models"
        output_dir = Path(__file__).parent.parent / "output"

        # Создаем выходную директорию
        output_dir.mkdir(exist_ok=True)

        # Проверяем наличие API ключа
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            logger.info("✅ OpenAI API ключ найден")
        else:
            logger.warning("⚠️  OpenAI API ключ не найден, используем демо режим")

        # Инициализируем LLM енхансер
        llm_enhancer = load_llm_enhancer(api_key)

        # Загружаем модель
        logger.info("📂 Загрузка модели...")
        model_trainer = ModelTrainer(str(models_dir))
        try:
            model_trainer.load_model("antifraud_model_v1")
        except FileNotFoundError:
            logger.error("❌ Модель не найдена. Сначала запустите обучение.")
            return False

        # Инициализируем извлекатель признаков
        logger.info("📊 Извлечение признаков...")
        feature_extractor = FeatureExtractor(str(data_dir))

        # Получаем признаки
        X_predict = feature_extractor.extract_features()

        if X_predict.empty:
            logger.warning("⚠️  Данные не найдены, создаем тестовые...")
            X_predict = feature_extractor.create_sample_features(n_samples=50)

            if 'user_id' in X_predict.columns:
                user_ids = X_predict['user_id'].copy()
                X_predict = X_predict.drop('user_id', axis=1)
            else:
                user_ids = pd.Series([f'test_user_{i}' for i in range(len(X_predict))])
        else:
            if 'user_id' in X_predict.columns:
                user_ids = X_predict['user_id'].copy()
                X_predict = X_predict.drop('user_id', axis=1)
            else:
                user_ids = pd.Series([f'user_{i}' for i in range(len(X_predict))])

        logger.info(f"📈 Данные для предсказания: {X_predict.shape}")

        # Делаем предсказания
        logger.info("🔮 Выполнение предсказаний...")
        predictions = model_trainer.predict(X_predict)
        probabilities = model_trainer.predict_proba(X_predict)
        fraud_probabilities = probabilities[:, 1]

        # Создаем базовый DataFrame с предсказаниями
        basic_predictions = pd.DataFrame({
            'user_id': user_ids.values,
            'sample_id': range(len(X_predict)),
            'prediction': predictions,
            'fraud_probability': fraud_probabilities,
            'risk_level': pd.cut(
                fraud_probabilities,
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Низкий', 'Средний', 'Высокий']
            )
        })

        # Добавляем LLM объяснения
        enhanced_predictions = enhance_predictions_with_explanations(
            basic_predictions, X_predict, llm_enhancer
        )

        # Сохраняем расширенные предсказания
        enhanced_file = output_dir / "fraud_predictions_with_llm.csv"
        enhanced_predictions.to_csv(enhanced_file, index=False, encoding='utf-8')
        logger.info(f"💾 Расширенные предсказания сохранены: {enhanced_file}")

        # Генерируем детальный отчет
        detailed_report = generate_detailed_fraud_report(enhanced_predictions, llm_enhancer)

        # Сохраняем отчет
        report_file = output_dir / "detailed_fraud_report_llm.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(detailed_report)
        logger.info(f"📄 Детальный отчет сохранен: {report_file}")

        # Создаем интерактивный отчет
        create_interactive_report(enhanced_predictions, output_dir)

        # Выводим статистику
        total_cases = len(enhanced_predictions)
        high_risk = len(enhanced_predictions[enhanced_predictions['fraud_probability'] >= 0.7])
        medium_risk = len(enhanced_predictions[
            (enhanced_predictions['fraud_probability'] >= 0.3) &
            (enhanced_predictions['fraud_probability'] < 0.7)
        ])

        logger.info("📊 Итоговая статистика:")
        logger.info(f"   Всего случаев: {total_cases}")
        logger.info(f"   Высокий риск: {high_risk} ({high_risk/total_cases:.1%})")
        logger.info(f"   Средний риск: {medium_risk} ({medium_risk/total_cases:.1%})")
        logger.info(f"   Низкий риск: {total_cases-high_risk-medium_risk} ({(total_cases-high_risk-medium_risk)/total_cases:.1%})")

        # Показываем примеры объяснений
        logger.info("\n🧠 Примеры LLM объяснений:")
        for i, (_, case) in enumerate(enhanced_predictions.head(3).iterrows()):
            logger.info(f"\n{i+1}. User {case['user_id']} (риск: {case['fraud_probability']:.1%}):")
            logger.info(f"   Объяснение: {case['llm_explanation'][:150]}...")
            logger.info(f"   Рекомендации: {case['recommendations'][:100]}...")

        logger.info("\n✅ Расширенное предсказание с LLM завершено!")
        logger.info(f"🎯 Файлы созданы:")
        logger.info(f"   - CSV с объяснениями: {enhanced_file}")
        logger.info(f"   - Детальный отчет: {report_file}")
        logger.info(f"   - HTML отчет: {output_dir / 'fraud_report_with_llm.html'}")

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка при выполнении предсказания с LLM: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
