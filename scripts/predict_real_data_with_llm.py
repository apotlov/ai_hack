#!/usr/bin/env python3
"""
Скрипт предсказания для реальных данных с локальными LLM объяснениями
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

from real_features_processor import RealFeaturesProcessor
from model_trainer import ModelTrainer
from local_llm_enhancer import LocalLLMEnhancer, OllamaInstaller

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_local_llm_for_real_data(model_name: str = "llama3.2:3b") -> Optional[LocalLLMEnhancer]:
    """
    Настройка локальной LLM для работы с реальными данными

    Args:
        model_name: Название модели

    Returns:
        Настроенный LocalLLMEnhancer или None при ошибке
    """
    logger.info("🤖 Настройка локальной LLM для реальных данных...")

    # Проверяем установку Ollama
    if not OllamaInstaller.check_ollama_installed():
        logger.error("❌ Ollama не установлена")
        print("\n" + OllamaInstaller.get_installation_instructions())
        return None

    # Рекомендуем модель по ресурсам
    recommended_model = OllamaInstaller.recommend_model_by_resources()
    logger.info(f"💡 Рекомендуемая модель: {recommended_model}")

    try:
        # Инициализируем локальную LLM
        llm_enhancer = LocalLLMEnhancer(model=model_name)

        # Тестируем соединение
        if llm_enhancer.test_connection():
            logger.info("✅ Локальная LLM готова для анализа реальных данных")

            # Показываем информацию о модели
            model_info = llm_enhancer.get_model_info()
            logger.info(f"📊 Модель: {model_info}")

            return llm_enhancer
        else:
            logger.warning("⚠️  Тест соединения не прошел, используем fallback режим")
            return llm_enhancer

    except Exception as e:
        logger.error(f"❌ Ошибка при настройке локальной LLM: {e}")
        return None


def enhance_real_predictions_with_llm(predictions_df: pd.DataFrame,
                                     features_df: pd.DataFrame,
                                     llm_enhancer: LocalLLMEnhancer) -> pd.DataFrame:
    """
    Добавление объяснений от локальной LLM к реальным предсказаниям

    Args:
        predictions_df: DataFrame с предсказаниями
        features_df: DataFrame с признаками
        llm_enhancer: Локальный LLM енхансер

    Returns:
        DataFrame с объяснениями
    """
    logger.info("🧠 Генерация объяснений для реальных данных с локальной LLM...")

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
            # Подготавливаем данные пользователя (реальные)
            user_data = {
                'applicationid': row.get('applicationid', f'app_{idx}'),
                'sample_id': row.get('sample_id', idx),
                'risk_level': row.get('risk_level', 'Неизвестно'),
                'fraud_probability': row.get('fraud_probability', 0)
            }

            # Получаем признаки для данного образца
            if idx < len(features_df):
                features = features_df.iloc[idx].to_dict()
                # Убираем NaN значения и фильтруем важные признаки
                features = {k: v for k, v in features.items()
                           if pd.notna(v) and _is_important_feature(k, v)}
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
            if (idx + 1) % 10 == 0 or idx + 1 == total_rows:
                progress = (idx + 1) / total_rows * 100
                logger.info(f"Обработано {idx + 1}/{total_rows} ({progress:.1f}%)")

        except Exception as e:
            logger.error(f"Ошибка при обработке строки {idx}: {e}")
            explanations.append("Не удалось сгенерировать объяснение для реальных данных")
            key_factors_list.append("Данные недоступны")
            recommendations_list.append("Требуется ручная проверка специалистом")
            confidence_scores.append(0.5)

    # Добавляем новые колонки
    enhanced_df['llm_explanation'] = explanations
    enhanced_df['key_factors'] = key_factors_list
    enhanced_df['recommendations'] = recommendations_list
    enhanced_df['explanation_confidence'] = confidence_scores

    logger.info("✅ Объяснения для реальных данных сгенерированы успешно")
    return enhanced_df


def _is_important_feature(feature_name: str, value: Any) -> bool:
    """
    Проверка важности признака для объяснения
    """
    # Пропускаем технические и служебные признаки
    skip_patterns = ['chunk', 'source', 'file_path', '_dup', 'sample_id', 'applicationid']
    if any(pattern in feature_name.lower() for pattern in skip_patterns):
        return False

    # Пропускаем признаки с очень малыми значениями
    if isinstance(value, (int, float)) and abs(value) < 0.001:
        return False

    return True


def create_real_data_fraud_report(enhanced_predictions: pd.DataFrame,
                                llm_enhancer: LocalLLMEnhancer) -> str:
    """
    Генерация отчета о мошенничестве для реальных данных

    Args:
        enhanced_predictions: DataFrame с предсказаниями и объяснениями
        llm_enhancer: Локальный LLM енхансер

    Returns:
        Текстовый отчет
    """
    logger.info("📊 Генерация отчета по реальным данным...")

    # Подготавливаем данные для анализа (только подозрительные случаи)
    fraud_cases = []
    for _, row in enhanced_predictions.iterrows():
        if row['fraud_probability'] > 0.2:  # Снижаем порог для реальных данных
            fraud_cases.append({
                'applicationid': row.get('applicationid', 'unknown'),
                'probability': row['fraud_probability'],
                'risk_level': row['risk_level'],
                'explanation': row['llm_explanation'],
                'key_factors': row['key_factors'],
                'timestamp': pd.Timestamp.now().isoformat()
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
    suspicious_cases = len(enhanced_predictions[enhanced_predictions['fraud_probability'] >= 0.2])

    detailed_report = f"""
+=== ОТЧЕТ АНТИФРОД СИСТЕМЫ ПО РЕАЛЬНЫМ ДАННЫМ ===
+Дата анализа: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
+Локальная модель: {llm_enhancer.model}
+
+СТАТИСТИКА АНАЛИЗА:
+- Всего проанализировано сессий: {total_cases:,}
+- Подозрительных случаев (≥20%): {suspicious_cases:,} ({suspicious_cases/total_cases:.1%})
+- Высокий риск (≥70%): {high_risk_cases:,} ({high_risk_cases/total_cases:.1%})
+- Средний риск (30-70%): {medium_risk_cases:,} ({medium_risk_cases/total_cases:.1%})
+- Низкий риск (<30%): {total_cases - high_risk_cases - medium_risk_cases:,}
+
+АНАЛИЗ ПАТТЕРНОВ МОШЕННИЧЕСТВА (Локальная LLM):
+{llm_report}
+
+КРИТИЧЕСКИЕ СЛУЧАИ ДЛЯ НЕМЕДЛЕННОЙ ПРОВЕРКИ:
+"""
+
+    # Добавляем критические случаи (≥80%)
+    critical_cases = enhanced_predictions[enhanced_predictions['fraud_probability'] >= 0.8]
+    if len(critical_cases) > 0:
+        detailed_report += f"\n🚨 НАЙДЕНО {len(critical_cases)} КРИТИЧЕСКИХ СЛУЧАЕВ:\n"
+        for i, (_, case) in enumerate(critical_cases.head(10).iterrows(), 1):
+            session_info = case.get('session_id', f"sample_{case.get('sample_id', i)}")
+            detailed_report += f"""
+{i}. Сессия {session_info} - Риск: {case['fraud_probability']:.1%}
+   ⚠️  Объяснение: {case['llm_explanation'][:150]}...
+   🔍 Ключевые факторы: {case['key_factors'][:120]}...
+   💡 Действия: {case['recommendations'][:100]}...
+"""
+    else:
+        detailed_report += "\n✅ Критических случаев (≥80%) не обнаружено.\n"
+
+    # Добавляем топ подозрительных случаев
+    detailed_report += f"\n📈 ТОП-5 ПОДОЗРИТЕЛЬНЫХ СЕССИЙ:\n"
+    top_cases = enhanced_predictions.nlargest(5, 'fraud_probability')
for i, (_, case) in enumerate(top_cases.iterrows(), 1):
    app_info = case.get('applicationid', f"sample_{case.get('sample_id', i)}")
    detailed_report += f"""
{i}. Заявка {app_info}
📊 Вероятность мошенничества: {case['fraud_probability']:.1%}
🎯 Уровень риска: {case['risk_level']}
🤖 LLM объяснение: {case['llm_explanation'][:200]}...
⚡ Рекомендации: {case['recommendations'][:150]}...
🎲 Уверенность объяснения: {case['explanation_confidence']:.1%}
"""
+"""
+
+    detailed_report += f"""
+
+=== РЕКОМЕНДАЦИИ ПО РЕАЛЬНЫМ ДАННЫМ ===
+
+НЕМЕДЛЕННЫЕ ДЕЙСТВИЯ:
+"""
+    if high_risk_cases > 0:
+        detailed_report += f"""
+1. 🚨 СРОЧНО проверить {high_risk_cases} случаев высокого риска
+2. 🔒 Заблокировать подозрительные операции до выяснения
+3. 📞 Связаться с клиентами для подтверждения операций
+4. 👮 Уведомить службу безопасности о критических случаях
+5. 📋 Создать инциденты в системе мониторинга
+"""
+    else:
+        detailed_report += "\n✅ Случаев, требующих немедленного вмешательства, не обнаружено.\n"
+
+    detailed_report += f"""
+УСИЛЕННЫЙ МОНИТОРИНГ:
+"""
+    if medium_risk_cases > 0:
+        detailed_report += f"""
+1. 👁️  Усилить наблюдение за {medium_risk_cases} случаями среднего риска
+2. 🔔 Настроить дополнительные алерты на данные сессии
+3. 🔐 Потребовать дополнительную аутентификацию при необходимости
+4. 📊 Мониторить поведение в течение следующих 24-48 часов
+"""
+
+    detailed_report += f"""
+
+СИСТЕМНЫЕ РЕКОМЕНДАЦИИ:
+1. 📈 Регулярно переобучать модель на новых данных (раз в неделю)
+2. 🔍 Анализировать ложноположительные срабатывания
+3. ⚙️  Настраивать пороги в зависимости от текущих угроз
+4. 📝 Документировать все случаи для улучшения модели
+5. 🤖 Использовать объяснения LLM для обучения аналитиков
+
+=== КОНЕЦ ОТЧЕТА ===
+"""
+
+    return detailed_report


def create_real_data_html_report(enhanced_predictions: pd.DataFrame, output_dir: Path):
    """
    Создание HTML отчета для реальных данных
    """
    logger.info("🌐 Создание HTML отчета для реальных данных...")

    # Подсчитываем статистику
    total_cases = len(enhanced_predictions)
    high_risk = len(enhanced_predictions[enhanced_predictions['fraud_probability'] >= 0.7])
    medium_risk = len(enhanced_predictions[
        (enhanced_predictions['fraud_probability'] >= 0.3) &
        (enhanced_predictions['fraud_probability'] < 0.7)
    ])
    low_risk = total_cases - high_risk - medium_risk

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Антифрод Анализ Реальных Данных</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background-color: #f5f7fa; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   color: white; padding: 30px; border-radius: 15px; margin-bottom: 25px;
                   box-shadow: 0 10px 30px rgba(0,0,0,0.1); }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .header p {{ margin: 5px 0; opacity: 0.9; }}

        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                  gap: 20px; margin: 25px 0; }}
        .stat-card {{ background: white; padding: 25px; border-radius: 12px; text-align: center;
                      box-shadow: 0 5px 15px rgba(0,0,0,0.08); transition: transform 0.2s; }}
        .stat-card:hover {{ transform: translateY(-2px); }}
        .stat-number {{ font-size: 2.5em; font-weight: bold; margin-bottom: 10px; }}
        .stat-label {{ color: #666; font-size: 1.1em; }}

        .risk-critical {{ background: linear-gradient(135deg, #ff6b6b, #ee5a52); color: white; }}
        .risk-high {{ background-color: #ffe8e8; border-left: 5px solid #ff4757; }}
        .risk-medium {{ background-color: #fff4e6; border-left: 5px solid #ffa726; }}
        .risk-low {{ background-color: #e8f5e8; border-left: 5px solid #66bb6a; }}

        .case {{ margin: 20px 0; padding: 25px; border-radius: 12px;
                 background: white; box-shadow: 0 3px 10px rgba(0,0,0,0.1); }}
        .case-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
        .session-id {{ font-size: 1.3em; font-weight: bold; color: #2c3e50; }}
        .probability {{ font-size: 1.2em; font-weight: bold; }}
        .badge {{ padding: 8px 16px; border-radius: 20px; color: white; font-size: 0.9em; font-weight: bold; }}
        .badge-critical {{ background: #e74c3c; }}
        .badge-high {{ background: #e67e22; }}
        .badge-medium {{ background: #f39c12; }}
        .badge-low {{ background: #27ae60; }}

        .explanation {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0;
                       border-left: 4px solid #3498db; }}
        .factors {{ background: #ecf0f1; padding: 15px; border-radius: 8px; margin: 15px 0; }}
        .recommendations {{ background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 15px 0;
                           border-left: 4px solid #27ae60; }}
        .confidence {{ font-size: 0.9em; color: #7f8c8d; text-align: right; margin-top: 10px; }}

        .critical-alert {{ background: #c0392b; color: white; padding: 20px; border-radius: 10px;
                          margin: 20px 0; text-align: center; font-weight: bold; }}
        .footer {{ text-align: center; margin-top: 40px; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Антифрод Анализ Реальных Данных</h1>
        <p>🤖 Локальная LLM модель: Ollama</p>
        <p>📅 Время анализа: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>🔒 Все данные обработаны локально</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-number">{total_cases:,}</div>
            <div class="stat-label">Всего сессий</div>
        </div>
        <div class="stat-card">
            <div class="stat-number risk-critical">{high_risk:,}</div>
            <div class="stat-label">Высокий риск</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" style="color: #ffa726;">{medium_risk:,}</div>
            <div class="stat-label">Средний риск</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" style="color: #66bb6a;">{low_risk:,}</div>
            <div class="stat-label">Низкий риск</div>
        </div>
    </div>
"""

    # Добавляем критическое предупреждение если есть случаи высокого риска
    if high_risk > 0:
        html_content += f"""
    <div class="critical-alert">
        🚨 ВНИМАНИЕ: Обнаружено {high_risk} случаев высокого риска, требующих немедленной проверки!
    </div>
"""

    html_content += "<h2>📊 Детальный анализ случаев</h2>"

    # Добавляем случаи (сначала самые подозрительные)
    sorted_predictions = enhanced_predictions.sort_values('fraud_probability', ascending=False)

    for i, (_, case) in enumerate(sorted_predictions.head(20).iterrows()):
    app_id = case.get('applicationid', f"sample_{case.get('sample_id', i)}")
    probability = case['fraud_probability']

    # Определяем стиль в зависимости от риска
    if probability >= 0.8:
        risk_class = "risk-critical"
        badge_class = "badge-critical"
        risk_text = "КРИТИЧЕСКИЙ"
    elif probability >= 0.7:
        risk_class = "risk-high"
        badge_class = "badge-high"
        risk_text = case['risk_level']
    elif probability >= 0.3:
        risk_class = "risk-medium"
        badge_class = "badge-medium"
        risk_text = case['risk_level']
    else:
        risk_class = "risk-low"
        badge_class = "badge-low"
        risk_text = case['risk_level']

    html_content += f"""
<div class="case {risk_class}">
    <div class="case-header">
        <div class="session-id">🎯 Заявка: {app_id}</div>
        <div>
            <span class="probability">{probability:.1%}</span>
            <span class="badge {badge_class}">{risk_text}</span>
        </div>
    </div>

        <div class="explanation">
            <strong>🧠 Объяснение локальной LLM:</strong><br>
            {case['llm_explanation']}
        </div>

        <div class="factors">
            <strong>🔍 Ключевые факторы риска:</strong><br>
            {case['key_factors'].replace(' | ', '<br>• ')}
        </div>

        <div class="recommendations">
            <strong>💡 Рекомендуемые действия:</strong><br>
            {case['recommendations'].replace(' | ', '<br>• ')}
        </div>

        <div class="confidence">
            🎯 Уверенность в анализе: {case['explanation_confidence']:.1%}
        </div>
    </div>
"""

    html_content += """
    <div class="footer">
        <p>🤖 Отчет сгенерирован локальной LLM без использования внешних API</p>
        <p>🔒 Все персональные данные остались в безопасности на локальном сервере</p>
        <p>⚡ Для получения актуальных данных перезапустите анализ</p>
    </div>
</body>
</html>
"""

    # Сохраняем HTML
    html_file = output_dir / "real_data_fraud_analysis.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"🌐 HTML отчет сохранен: {html_file}")


def main():
    """
    Основная функция анализа реальных данных с локальной LLM
    """
    logger.info("🚀 Запуск анализа реальных данных с локальной LLM")

    try:
        # Настройки путей
        project_dir = Path(__file__).parent.parent
        models_dir = project_dir / "models"
        output_dir = project_dir / "output"
        data_dir = project_dir / "data"

        # Создаем выходную директорию
        output_dir.mkdir(exist_ok=True)

        # Настраиваем локальную LLM
        llm_enhancer = setup_local_llm_for_real_data()
        if not llm_enhancer:
            logger.error("❌ Не удалось настроить локальную LLM")
            return False

        # Загружаем обученную модель
        logger.info("📂 Загрузка обученной модели...")
        model_trainer = ModelTrainer(str(models_dir))

        model_names = ["real_antifraud_model", "antifraud_model_v1", "antifraud_model"]
        model_loaded = False

        for model_name in model_names:
            try:
                model_trainer.load_model(model_name)
                logger.info(f"✅ Модель загружена: {model_name}")
                model_loaded = True
                break
            except FileNotFoundError:
                continue

        if not model_loaded:
            logger.error("❌ Не найдена обученная модель. Запустите: python scripts/train_real_data.py")
            return False

        # Инициализируем процессор признаков
        logger.info("🔧 Инициализация процессора признаков...")
        features_processor = RealFeaturesProcessor(str(data_dir))

        # Подготавливаем признаки для предсказания
        logger.info("📊 Подготовка признаков из реальных данных...")
        features_df = features_processor.create_prediction_features(str(data_dir))

        if features_df.empty:
            logger.error("❌ Не удалось подготовить признаки из реальных данных")
            return False

        # Сохраняем applicationids
        application_ids = None
        if 'applicationid' in features_df.columns:
            application_ids = features_df['applicationid'].copy()
            features_df = features_df.drop('applicationid', axis=1)

        logger.info(f"📈 Подготовлено для анализа: {features_df.shape}")

        # Делаем предсказания
        logger.info("🔮 Выполнение предсказаний...")
        predictions = model_trainer.predict(features_df)
        probabilities = model_trainer.predict_proba(features_df)
        fraud_probabilities = probabilities[:, 1]

        # Создаем базовый DataFrame с предсказаниями
        basic_predictions = pd.DataFrame({
            'sample_id': range(len(features_df)),
            'prediction': predictions,
            'fraud_probability': fraud_probabilities,
            'risk_level': pd.cut(
                fraud_probabilities,
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Низкий', 'Средний', 'Высокий']
            )
        })

        # Добавляем applicationids если есть
        if application_ids is not None:
            basic_predictions.insert(0, 'applicationid', application_ids.values)

        # Добавляем объяснения от локальной LLM
        enhanced_predictions = enhance_real_predictions_with_llm(
            basic_predictions, features_df, llm_enhancer
        )

        # Сохраняем результаты
        results_file = output_dir / "real_data_predictions_with_llm.csv"
        enhanced_predictions.to_csv(results_file, index=False, encoding='utf-8')
        logger.info(f"💾 Результаты сохранены: {results_file}")

        # Генерируем детальный отчет
        detailed_report = create_real_data_fraud_report(enhanced_predictions, llm_enhancer)

        # Сохраняем отчет
        report_file = output_dir / "real_data_fraud_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(detailed_report)
        logger.info(f"📄 Отчет сохранен: {report_file}")

        # Создаем HTML отчет
        create_real_data_html_report(enhanced_predictions, output_dir)

        # Выводим статистику
        total_cases = len(enhanced_predictions)
        high_risk = len(enhanced_predictions[enhanced_predictions['fraud_probability'] >= 0.7])
        medium_risk = len(enhanced_predictions[
            (enhanced_predictions['fraud_probability'] >= 0.3) &
            (enhanced_predictions['fraud_probability'] < 0.7)
        ])
        critical_cases = len(enhanced_predictions[enhanced_predictions['fraud_probability'] >= 0.8])

        logger.info("\n" + "="*70)
        logger.info("✅ АНАЛИЗ РЕАЛЬНЫХ ДАННЫХ С LLM ЗАВЕРШЕН!")
        logger.info("="*70)
        logger.info(f"📊 Всего проанализировано заявок: {total_cases:,}")
        logger.info(f"🚨 Критический риск (≥80%): {critical_cases:,}")
        logger.info(f"🔴 Высокий риск (≥70%): {high_risk:,}")
        logger.info(f"🟡 Средний риск (30-70%): {medium_risk:,
