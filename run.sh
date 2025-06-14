#!/bin/bash

# =============================================================================
# 🚀 Автоматизированный скрипт запуска антифрод системы
# =============================================================================
#
# Структура данных:
# data_train/          - данные для обучения
# ├── amplitude/       - parquet файлы обучения
# ├── audiofiles/      - аудио для обучения
# └── svod.csv         - связки для обучения
#
# data/                - данные для предсказаний
# ├── amplitude/       - parquet файлы предсказаний
# │   └── svod.csv     - связки для предсказаний
# └── audiofiles/      - аудио для предсказаний
#
# =============================================================================

set -e  # Остановка при любой ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Функция логирования
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Функция проверки Python зависимостей
check_dependencies() {
    log "🔍 Проверка зависимостей..."

    # Проверяем Python
    if ! command -v python3 &> /dev/null; then
        error "Python3 не найден!"
        exit 1
    fi

    # Проверяем основные библиотеки
    python3 -c "
import pandas, numpy, sklearn, tqdm
print('✅ Основные библиотеки установлены')
" || {
        error "Не все зависимости установлены!"
        echo "Установите зависимости: pip install -r requirements.txt"
        exit 1
    }

    success "Зависимости проверены"
}

# Функция проверки структуры данных
check_data_structure() {
    log "📁 Проверка структуры данных..."

    # Проверяем папку обучения
    if [ ! -d "data_train" ]; then
        error "Папка data_train не найдена!"
        exit 1
    fi

    if [ ! -d "data_train/amplitude" ]; then
        error "Папка data_train/amplitude не найдена!"
        exit 1
    fi

    if [ ! -d "data_train/audiofiles" ]; then
        warning "Папка data_train/audiofiles не найдена!"
    fi

    if [ ! -f "data_train/svod.csv" ]; then
        warning "Файл data_train/svod.csv не найден!"
    fi

    # Проверяем папку предсказаний
    if [ ! -d "data" ]; then
        error "Папка data не найдена!"
        exit 1
    fi

    if [ ! -d "data/amplitude" ]; then
        error "Папка data/amplitude не найдена!"
        exit 1
    fi

    if [ ! -d "data/audiofiles" ]; then
        warning "Папка data/audiofiles не найдена!"
    fi

    if [ ! -f "data/amplitude/svod.csv" ]; then
        warning "Файл data/amplitude/svod.csv не найден!"
    fi

    # Проверяем наличие файлов обучения
    TRAIN_PARQUET_COUNT=$(find data_train/amplitude -name "*.parquet" 2>/dev/null | wc -l)
    if [ "$TRAIN_PARQUET_COUNT" -eq 0 ]; then
        error "Нет parquet файлов в data_train/amplitude!"
        exit 1
    fi

    # Проверяем наличие файлов предсказаний
    PRED_PARQUET_COUNT=$(find data/amplitude -name "*.parquet" 2>/dev/null | wc -l)
    if [ "$PRED_PARQUET_COUNT" -eq 0 ]; then
        error "Нет parquet файлов в data/amplitude!"
        exit 1
    fi

    success "Структура данных корректна"
    log "📊 Найдено файлов для обучения: $TRAIN_PARQUET_COUNT"
    log "📊 Найдено файлов для предсказаний: $PRED_PARQUET_COUNT"
}

# Функция проверки Ollama
check_ollama() {
    log "🤖 Проверка Ollama..."

    if ! command -v ollama &> /dev/null; then
        warning "Ollama не установлена!"
        echo ""
        echo "Для установки Ollama:"
        echo "curl -fsSL https://ollama.ai/install.sh | sh"
        echo ""
        echo "Затем запустите:"
        echo "ollama serve"
        echo "ollama pull llama3.2:3b"
        echo ""
        echo "Продолжаем без LLM объяснений..."
        return 1
    fi

    # Проверяем запущен ли сервер
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        warning "Ollama сервер не запущен!"
        echo "Запустите: ollama serve"
        echo "Продолжаем без LLM объяснений..."
        return 1
    fi

    # Проверяем наличие модели
    if ! ollama list | grep -q "llama3.2:3b"; then
        warning "Модель llama3.2:3b не найдена!"
        echo "Установите модель: ollama pull llama3.2:3b"
        echo "Продолжаем без LLM объяснений..."
        return 1
    fi

    success "Ollama готова к работе"
    return 0
}

# Функция подготовки окружения
prepare_environment() {
    log "🔧 Подготовка окружения..."

    # Создаем необходимые папки
    mkdir -p models
    mkdir -p output
    mkdir -p logs

    # Устанавливаем PYTHONPATH
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

    success "Окружение готово"
}

# Функция адаптации путей в скриптах
adapt_scripts() {
    log "🔄 Адаптация скриптов под структуру данных..."

    # Используем готовый скрипт для data_train
    if [ ! -f "scripts/train_with_data_train.py" ]; then
        error "Скрипт train_with_data_train.py не найден!"
        return 1
    fi

    success "Скрипты готовы к использованию"
}

# Функция запуска обучения
run_training() {
    log "🎓 Запуск обучения модели..."
    echo ""
    echo -e "${PURPLE}=== ЭТАП 1: ОБУЧЕНИЕ АНТИФРОД МОДЕЛИ ===${NC}"
    echo ""

    # Запускаем обучение с логированием
    if python3 scripts/train_with_data_train.py 2>&1 | tee logs/training.log; then
        success "Обучение завершено успешно!"

        # Проверяем что модель создана
        if [ -f "models/real_antifraud_model.joblib" ]; then
            success "Модель сохранена: models/real_antifraud_model.joblib"
        else
            error "Модель не найдена после обучения!"
            return 1
        fi
    else
        error "Ошибка при обучении модели!"
        echo "Проверьте логи: logs/training.log"
        return 1
    fi
}

# Функция запуска предсказаний
run_predictions() {
    log "🔮 Запуск предсказаний..."
    echo ""
    echo -e "${PURPLE}=== ЭТАП 2: ПРЕДСКАЗАНИЯ С LLM АНАЛИЗОМ ===${NC}"
    echo ""

    # Проверяем наличие обученной модели
    if [ ! -f "models/real_antifraud_model.joblib" ]; then
        error "Обученная модель не найдена!"
        echo "Сначала запустите обучение"
        return 1
    fi

    # Запускаем предсказания с логированием
    if python3 scripts/predict_real_data_with_llm.py 2>&1 | tee logs/predictions.log; then
        success "Предсказания завершены успешно!"

        # Проверяем созданные файлы
        if [ -f "output/real_data_predictions_with_llm.csv" ]; then
            success "Результаты сохранены: output/real_data_predictions_with_llm.csv"
        fi

        if [ -f "output/real_data_fraud_analysis.html" ]; then
            success "HTML отчет создан: output/real_data_fraud_analysis.html"
        fi

        if [ -f "output/real_data_fraud_report.txt" ]; then
            success "Текстовый отчет создан: output/real_data_fraud_report.txt"
        fi
    else
        error "Ошибка при выполнении предсказаний!"
        echo "Проверьте логи: logs/predictions.log"
        return 1
    fi
}

# Функция очистки временных файлов
cleanup() {
    log "🧹 Очистка временных файлов..."

    # Очищаем временные файлы если есть
    rm -f /tmp/antifraud_*.tmp

    success "Очистка завершена"
}

# Функция показа результатов
show_results() {
    echo ""
    echo -e "${GREEN}=== 🎉 АНТИФРОД СИСТЕМА УСПЕШНО ЗАПУЩЕНА! ===${NC}"
    echo ""
    echo -e "${CYAN}📊 Результаты работы:${NC}"
    echo ""

    if [ -f "models/real_antifraud_model.joblib" ]; then
        echo "✅ Модель: models/real_antifraud_model.joblib"
    fi

    if [ -f "output/real_data_predictions_with_llm.csv" ]; then
        echo "✅ Предсказания: output/real_data_predictions_with_llm.csv"
        PRED_COUNT=$(tail -n +2 output/real_data_predictions_with_llm.csv | wc -l)
        echo "   📈 Проанализировано заявок: $PRED_COUNT"
    fi

    if [ -f "output/real_data_fraud_analysis.html" ]; then
        echo "✅ HTML отчет: output/real_data_fraud_analysis.html"
        echo "   🌐 Откройте в браузере для интерактивного анализа"
    fi

    if [ -f "output/real_data_fraud_report.txt" ]; then
        echo "✅ Текстовый отчет: output/real_data_fraud_report.txt"
    fi

    if [ -f "logs/training.log" ]; then
        echo "📝 Логи обучения: logs/training.log"
    fi

    if [ -f "logs/predictions.log" ]; then
        echo "📝 Логи предсказаний: logs/predictions.log"
    fi

    echo ""
    echo -e "${YELLOW}💡 Рекомендации:${NC}"
    echo "1. Изучите HTML отчет для детального анализа"
    echo "2. Проверьте выявленные случаи высокого риска"
    echo "3. Используйте LLM объяснения для принятия решений"
    echo ""
}

# Функция обработки ошибок
handle_error() {
    error "Произошла ошибка в строке $1"
    cleanup
    exit 1
}

# Установка обработчика ошибок
trap 'handle_error $LINENO' ERR

# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

main() {
    echo ""
    echo -e "${BLUE}🛡️  АНТИФРОД СИСТЕМА ДЛЯ БАНКА${NC}"
    echo -e "${BLUE}=================================${NC}"
    echo ""
    echo "Автоматическое обучение и анализ данных с локальной LLM"
    echo ""

    # Этап 1: Проверки
    check_dependencies
    check_data_structure
    OLLAMA_AVAILABLE=0
    check_ollama && OLLAMA_AVAILABLE=1

    # Этап 2: Подготовка
    prepare_environment
    adapt_scripts

    # Этап 3: Обучение
    if ! run_training; then
        cleanup
        exit 1
    fi

    # Этап 4: Предсказания
    if [ $OLLAMA_AVAILABLE -eq 1 ]; then
        if ! run_predictions; then
            cleanup
            exit 1
        fi
    else
        warning "Пропускаем этап предсказаний (Ollama недоступна)"
        echo "Для получения LLM объяснений установите и запустите Ollama"
    fi

    # Этап 5: Завершение
    cleanup
    show_results

    echo -e "${GREEN}🚀 Система готова к работе!${NC}"
}

# Проверка аргументов командной строки
case "${1:-}" in
    --help|-h)
        echo "Использование: $0 [опции]"
        echo ""
        echo "Опции:"
        echo "  --help, -h     Показать эту справку"
        echo "  --train-only   Только обучение модели"
        echo "  --predict-only Только предсказания (требует обученной модели)"
        echo "  --check        Только проверка зависимостей и данных"
        echo ""
        echo "Структура данных:"
        echo "  data_train/    - данные для обучения"
        echo "  data/          - данные для предсказаний"
        echo ""
        exit 0
        ;;
    --train-only)
        check_dependencies
        check_data_structure
        prepare_environment
        adapt_scripts
        run_training
        cleanup
        ;;
    --predict-only)
        check_dependencies
        prepare_environment
        run_predictions
        ;;
    --check)
        check_dependencies
        check_data_structure
        check_ollama
        echo ""
        success "Все проверки пройдены!"
        ;;
    "")
        main
        ;;
    *)
        error "Неизвестная опция: $1"
        echo "Используйте --help для справки"
        exit 1
        ;;
esac
