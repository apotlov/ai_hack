#!/usr/bin/env python3
"""
Скрипт предсказания с локальными LLM через Ollama
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
from local_llm_enhancer import LocalLLMEnhancer, OllamaInstaller

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_ollama_setup() -> bool:
    """
    Проверка настройки Ollama

    Returns:
        True если Ollama готова к использованию
    """
    logger.info("🔍 Проверка настройки Ollama...")

    # Проверяем установку
    if not OllamaInstaller.check_ollama_installed():
        logger.error("❌ Ollama не установлена")
        print("\n" + OllamaInstaller.get_installation_instructions())
        return False

    logger.info("✅ Ollama установлена")

    # Рекомендуем модель по ресурсам
    recommended_model = OllamaInstaller.recommend_model_by_resources()
    logger.info(f"💡 Рекомендуемая модель для вашей системы: {recommended_model}")

    return True


def setup_local_llm(model_name: str = "llama3.2:3b") -> Optional[LocalLLMEnhancer]:
    """
    Настройка локальной LLM

    Args:
        model_name: Название модели

    Returns:
        Настроенный LocalLLMEnhancer или None при ошибке
    """
    try:
        logger.info(f"🤖 Инициализация локальной LLM: {model_name}")

        # Создаем энхансер
        llm_enhancer = LocalLLMEnhancer(model=model_name)

        # Тестируем соединение
        if llm_enhancer.test_connection():
            logger.info("✅ Локальная LLM готова к работе")

            # Показываем информацию о модели
            model_info = llm_enhancer.get_model_info()
            logger.info(f"📊 Информация о модели: {model_info}")

            return llm_enhancer
        else:
            logger.warning("⚠️  Тест соединения не прошел, используем fallback режим")
            return llm_enhancer

    except Exception as e:
        logger.error(f"❌ Ошибка при настройке локальной LLM: {e}")
        return None


def enhance_predictions_with_local_llm(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
    llm_enhancer: LocalLLMEnhancer
) -> pd.DataFrame:
    """
    Добавление объяснений от локальной LLM к предсказаниям

    Args:
        predictions_df: DataFrame с предсказаниями
        features_df: DataFrame с признаками
        llm_enhancer: Локальный LLM енхансер

    Returns:
        DataFrame с объяснениями
    """
    logger.info("🧠 Генерация объяснений с помощью локальной LLM...")

    enhanced_df = predictions_df.copy()

    # Списки для новых колонок
    explanations = []
    key_factors_list = []
    recommendations_list = []
    confidence_scores = []

    total_rows = len(predictions_df)

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

            # Генерируем объяснение с помощью локальной LLM
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
            if (idx + 1) % 5 == 0 or idx + 1 == total_rows:
                progress = (idx + 1) / total_rows * 100
                logger.info(f"Обработано {idx + 1}/{total_rows} ({progress:.1f}%)")

        except Exception as e:
            logger.error(f"Ошибка при обработке строки {idx}: {e}")
            explanations.append("Не удалось сгенерировать объяснение")
            key_factors_list.append("Данные недоступны")
            recommendations_list.append("Требуется ручная проверка")
            confidence_scores.append(0.5)

    # Добавляем новые колонки
    enhanced_df['local_llm_explanation'] = explanations
    enhanced_df['key_factors'] = key_factors_list
    enhanced_df['recommendations'] = recommendations_list
    enhanced_df['explanation_confidence'] = confidence_scores

    logger.info("✅ Объяснения от локальной LLM сгенерированы успешно")
    return enhanced_df


def generate_local_fraud_report(
    enhanced_predictions: pd.DataFrame,
    llm_enhancer: LocalLLMEnhancer
) -> str:
    """
    Генерация отчета о мошенничестве с помощью локальной LLM

    Args:
        enhanced_predictions: DataFrame с предсказаниями и объяснениями
        llm_enhancer: Локальный LLM енхансер

    Returns:
        Текстовый отчет
    """
    logger.info("📊 Генерация отчета с помощью локальной LLM...")

    # Подготавливаем данные для анализа
    fraud_cases = []
    for _, row in enhanced_predictions.iterrows():
        if row['fraud_probability'] > 0.3:  # Только подозрительные случаи
            fraud_cases.append({
                'user_id': row['user_id'],
                'probability': row['fraud_probability'],
                'risk_level': row['risk_level'],
                'explanation': row['local_llm_explanation'],
                'key_factors': row['key_factors']
            })

    # Генерируем отчет с помощью локальной LLM
    llm_report = llm_enhancer.generate_fraud_report(fraud_cases)

    # Добавляем статистическую информацию
    total_cases = len(enhanced_predictions)
    high_risk_cases = len(enhanced_predictions[enhanced_predictions['fraud_probability'] >= 0.7])
    medium_risk_cases = len(enhanced_predictions[
        (enhanced_predictions['fraud_probability'] >= 0.3) &
        (enhanced_predictions['fraud_probability'] < 0.7)
    ])

    detailed_report = f"""
=== ОТЧЕТ АНТИФРОД СИСТЕМЫ С ЛОКАЛЬНОЙ LLM ===
Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Модель: {llm_enhancer.model}

СТАТИСТИКА:
- Всего проанализировано: {total_cases}
- Высокий риск (≥70%): {high_risk_cases}
- Средний риск (30-70%): {medium_risk_cases}
- Низкий риск (<30%): {total_cases - high_risk_cases - medium_risk_cases}

АНАЛИЗ ПАТТЕРНОВ (Локальная LLM):
{llm_report}

ТОП-5 САМЫХ ПОДОЗРИТЕЛЬНЫХ СЛУЧАЕВ:
"""

    # Добавляем топ случаи
    top_cases = enhanced_predictions.nlargest(5, 'fraud_probability')
    for i, (_, case) in enumerate(top_cases.iterrows(), 1):
        detailed_report += f"""
{i}. User {case['user_id']} - Риск: {case['fraud_probability']:.1%}
   Объяснение: {case['local_llm_explanation'][:200]}...
   Ключевые факторы: {case['key_factors'][:150]}...
   Рекомендации: {case['recommendations'][:150]}...
"""

    return detailed_report


def create_local_llm_html_report(enhanced_predictions: pd.DataFrame, output_dir: Path):
    """
    Создание HTML отчета с результатами локальной LLM

    Args:
        enhanced_predictions: DataFrame с предсказаниями
        output_dir: Директория для сохранения
    """
    logger.info("📋 Создание HTML отчета...")

    # HTML отчет
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Антифрод Отчет с Локальной LLM</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; text-align: center;
                      box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .risk-high {{ background-color: #ffebee; border-left: 4px solid #f44336; }}
        .risk-medium {{ background-color: #fff3e0; border-left: 4px solid #ff9800; }}
        .risk-low {{ background-color: #e8f5e8; border-left: 4px solid #4caf50; }}
        .case {{ margin: 15px 0; padding: 20px; border-radius: 8px;
                 background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .user-id {{ font-size: 1.2em; font-weight: bold; color: #333; }}
        .probability {{ font-size: 1.1em; font-weight: bold; }}
        .explanation {{ font-style: italic; color: #555; margin: 10px 0;
                       background: #f8f9fa; padding: 15px; border-radius: 5px; }}
        .factors {{ color: #333; font-weight: 500; margin: 10px 0; }}
        .recommendations {{ color: #0066cc; margin: 10px 0;
                           background: #e3f2fd; padding: 10px; border-radius: 5px; }}
        .confidence {{ font-size: 0.9em; color: #666; }}
        .badge {{ padding: 5px 10px; border-radius: 15px; color: white; font-size: 0.9em; }}
        .badge-high {{ background-color: #f44336; }}
        .badge-medium {{ background-color: #ff9800; }}
        .badge-low {{ background-color: #4caf50; }}
        .footer {{ text-align: center; margin-top: 30px; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Антифрод Отчет с Локальной LLM</h1>
        <p>🤖 Модель: Локальная LLM через Ollama</p>
        <p>📅 Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-number">{len(enhanced_predictions)}</div>
            <div>Всего случаев</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" style="color: #f44336;">
                {len(enhanced_predictions[enhanced_predictions['fraud_probability'] >= 0.7])}
            </div>
            <div>Высокий риск</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" style="color: #ff9800;">
                {len(enhanced_predictions[(enhanced_predictions['fraud_probability'] >= 0.3) & (enhanced_predictions['fraud_probability'] < 0.7)])}
            </div>
            <div>Средний риск</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" style="color: #4caf50;">
                {len(enhanced_predictions[enhanced_predictions['fraud_probability'] < 0.3])}
            </div>
            <div>Низкий риск</div>
        </div>
    </div>

    <h2>🚨 Детальный анализ случаев</h2>
"""

    # Добавляем случаи (первые 20)
    for _, case in enhanced_predictions.head(20).iterrows():
        risk_class = "risk-high" if case['fraud_probability'] >= 0.7 else "risk-medium" if case['fraud_probability'] >= 0.3 else "risk-low"
        badge_class = "badge-high" if case['fraud_probability'] >= 0.7 else "badge-medium" if case['fraud_probability'] >= 0.3 else "badge-low"

        html_content += f"""
    <div class="case {risk_class}">
        <div class="user-id">👤 {case['user_id']}</div>
        <div class="probability">
            📊 Вероятность: {case['fraud_probability']:.1%}
            <span class="badge {badge_class}">{case['risk_level']}</span>
        </div>
        <div class="explanation">
            <strong>🧠 Объяснение локальной LLM:</strong><br>
            {case['local_llm_explanation']}
        </div>
        <div class="factors">
            <strong>🔍 Ключевые факторы:</strong><br>
            {case['key_factors'].replace(' | ', '<br>• ')}
        </div>
        <div class="recommendations">
            <strong>💡 Рекомендации:</strong><br>
            {case['recommendations'].replace(' | ', '<br>• ')}
        </div>
        <div class="confidence">
            🎯 Уверенность в объяснении: {case['explanation_confidence']:.1%}
        </div>
    </div>
"""

    html_content += """
    <div class="footer">
        <p>🤖 Отчет сгенерирован локальной LLM без использования внешних API</p>
        <p>🔒 Все данные обработаны локально для максимальной безопасности</p>
    </div>
</body>
</html>
"""

    # Сохраняем HTML
    html_file = output_dir / "fraud_report_local_llm.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"💾 HTML отчет сохранен: {html_file}")


def main():
    """
    Основная функция предсказания с локальной LLM
    """
    logger.info("🚀 Запуск предсказания с локальной LLM")

    try:
        # Проверяем настройку Ollama
        if not check_ollama_setup():
            return False

        # Настройки
        data_dir = Path(__file__).parent.parent / "data"
        models_dir = Path(__file__).parent.parent / "models"
        output_dir = Path(__file__).parent.parent / "output"

        # Создаем выходную директорию
        output_dir.mkdir(exist_ok=True)

        # Выбор модели (можно настроить)
        available_models = {
            "1": ("llama3.2:3b", "Llama 3.2 3B - Рекомендуется (6GB RAM)"),
            "2": ("phi3.5", "Phi-3.5 Mini - Быстрая (4GB RAM)"),
            "3": ("qwen2.5:7b", "Qwen 2.5 7B - Качественная (8GB RAM)")
        }

        logger.info("🤖 Доступные локальные модели:")
        for key, (model, desc) in available_models.items():
            logger.info(f"  {key}. {desc}")

        # Используем рекомендуемую модель по умолчанию
        selected_model = "llama3.2:3b"
        logger.info(f"📌 Выбрана модель: {selected_model}")

        # Инициализируем локальную LLM
        llm_enhancer = setup_local_llm(selected_model)
        if not llm_enhancer:
            logger.error("❌ Не удалось настроить локальную LLM")
            return False

        # Загружаем модель ML
        logger.info("📂 Загрузка ML модели...")
        model_trainer = ModelTrainer(str(models_dir))
        try:
            model_trainer.load_model("antifraud_model_v1")
        except FileNotFoundError:
            logger.error("❌ ML модель не найдена. Сначала запустите обучение:")
            logger.error("   python3 scripts/main.py --train")
            return False

        # Инициализируем извлекатель признаков
        logger.info("📊 Извлечение признаков...")
        feature_extractor = FeatureExtractor(str(data_dir))

        # Получаем признаки
        X_predict = feature_extractor.extract_features()

        if X_predict.empty:
            logger.warning("⚠️  Данные не найдены, создаем тестовые...")
            X_predict = feature_extractor.create_sample_features(n_samples=30)

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
        logger.info("🔮 Выполнение ML предсказаний...")
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

        # Добавляем объяснения от локальной LLM
        enhanced_predictions = enhance_predictions_with_local_llm(
            basic_predictions, X_predict, llm_enhancer
        )

        # Сохраняем расширенные предсказания
        enhanced_file = output_dir / "fraud_predictions_local_llm.csv"
        enhanced_predictions.to_csv(enhanced_file, index=False, encoding='utf-8')
        logger.info(f"💾 Предсказания с локальной LLM сохранены: {enhanced_file}")

        # Генерируем отчет
        detailed_report = generate_local_fraud_report(enhanced_predictions, llm_enhancer)

        # Сохраняем отчет
        report_file = output_dir / "fraud_report_local_llm.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(detailed_report)
        logger.info(f"📄 Отчет сохранен: {report_file}")

        # Создаем HTML отчет
        create_local_llm_html_report(enhanced_predictions, output_dir)

        # Выводим статистику
        total_cases = len(enhanced_predictions)
        high_risk = len(enhanced_predictions[enhanced_predictions['fraud_probability'] >= 0.7])
        medium_risk = len(enhanced_predictions[
            (enhanced_predictions['fraud_probability'] >= 0.3) &
            (enhanced_predictions['fraud_probability'] < 0.7)
        ])

        logger.info("\n📊 ИТОГОВАЯ СТАТИСТИКА:")
        logger.info(f"   Всего случаев: {total_cases}")
        logger.info(f"   🔴 Высокий риск: {high_risk} ({high_risk/total_cases:.1%})")
        logger.info(f"   🟡 Средний риск: {medium_risk} ({medium_risk/total_cases:.1%})")
        logger.info(f"   🟢 Низкий риск: {total_cases-high_risk-medium_risk} ({(total_cases-high_risk-medium_risk)/total_cases:.1%})")

        # Показываем примеры объяснений
        logger.info("\n🧠 ПРИМЕРЫ ОБЪЯСНЕНИЙ ОТ ЛОКАЛЬНОЙ LLM:")
        for i, (_, case) in enumerate(enhanced_predictions.head(3).iterrows()):
            logger.info(f"\n{i+1}. 👤 User {case['user_id']} (риск: {case['fraud_probability']:.1%}):")
            logger.info(f"   💭 Объяснение: {case['local_llm_explanation'][:120]}...")
            logger.info(f"   💡 Рекомендации: {case['recommendations'][:80]}...")

        logger.info("\n✅ ПРЕДСКАЗАНИЕ С ЛОКАЛЬНОЙ LLM ЗАВЕРШЕНО!")
        logger.info(f"🎯 Созданные файлы:")
        logger.info(f"   📊 CSV: {enhanced_file.name}")
        logger.info(f"   📄 Отчет: {report_file.name}")
        logger.info(f"   🌐 HTML: fraud_report_local_llm.html")

        logger.info(f"\n🤖 Использована локальная модель: {llm_enhancer.model}")
        logger.info(f"🔒 Все данные обработаны локально без внешних API")

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка при выполнении предсказания: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
