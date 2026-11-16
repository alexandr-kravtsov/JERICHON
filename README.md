# 🏗️ Архитектура проекта JERICHON: Текущее состояние и Предлагаемая реорганизация

Этот документ описывает текущую структуру проекта JERICHON, выявляет ее ограничения и предлагает новую, отраслевую архитектуру (Industry Standard), оптимизированную для масштабирования исследований, надежного продакшена и успешной защиты ВКР.

---

## 📉 1. Текущая архитектура (As-Is)

Текущая структура сложилась эволюционно в процессе исследований. Основная проблема: **смешение данных, исследовательского кода и бизнес-логики**, что нарушает принцип Separation of Concerns.

```text
JERICHON/
├── app/                          # Streamlit UI (в целом ок)
├── config/                       # YAML конфигурации (в целом ок)
├── data/                         # ❌ ПРОБЛЕМА: Смешение данных и кода
│   ├── sample_data/              # ✅ Примеры данных (хорошо)
│   ├── for_reading_reports/      # Данные для фундаментального анализа
│   └── for_ml/                   # ❌ ГЛАВНАЯ ПРОБЛЕМА: Код заперт в папке "данных"
│       ├── storage/              # Локальное хранилище (нет в Git, но путь жестко задан)
│       ├── providers/            # Провайдеры данных (Algopack)
│       ├── builders/             # Построение баров и фичей
│       ├── labeling/             # CUSUM, Triple Barrier
│       └── models/               # ❌ Jupyter Notebook с бэктестингом внутри папки "модели"
├── live/                         # Модуль лайв-торговли (BCS API)
├── models/                       # Сохраненные веса моделей и реестр
├── src/                          # ❌ ПРОБЛЕМА: Почти пустая или используется не по назначению
│   └── agents/                   # Только для фундаментального анализа
├── .gitignore
├── README.md
└── requirements.txt


JERICHON/
├── .github/                      # ✅ CI/CD: Автотесты и линтинг при каждом push
├── config/                       # Все гиперпараметры и настройки (YAML)
│   ├── data_pipeline.yaml        # Настройки Algopack, пороги Dollar Bars
│   ├── features.yaml             # Конфигурация генерации признаков (73 → 475)
│   ├── risk.yaml                 # Лимиты риска, параметры Ralph Vince (optimal f)
│   └── strategies.yaml           # Пороги переключения рыночных режимов
│
├── data/                         # ✅ ТОЛЬКО ДАННЫЕ. Содержимое игнорируется в Git.
│   ├── sample_data/              # ✅ В Git! Минимальные срезы для Unit-тестов
│   ├── raw/                      # 🚫 .gitkeep (Сырые тики/стаканы MOEX)
│   ├── interim/                  # 🚫 .gitkeep (Dollar Bars, Super Candles)
│   └── processed/                # 🚫 .gitkeep (Готовые X, y с фичами и лейблами)
│
├── src/                          # ✅ ЯДРО: Весь переиспользуемый Python-код
│   ├── __init__.py
│   ├── pipeline/                 # Бывший data/for_ml/ (теперь на своем месте)
│   │   ├── providers.py          # Клиенты данных (Algopack, BCS)
│   │   ├── builders.py           # Построение баров и Super Candles
│   │   └── labeling.py           # CUSUM, Triple Barrier, sample weights
│   ├── features/                 # Инжиниринг признаков
│   │   ├── microstructure.py     # Order flow, дисбаланс стакана, спреды
│   │   └── regime_features.py    # Фичи для классификации рыночных режимов
│   ├── models/                   # ML-логика
│   │   ├── training.py           # Обучение с Purged K-Fold (без утечек!)
│   │   ├── registry.py           # Загрузка/сохранение моделей
│   │   └── regime_detector.py    # Модель определения текущего рыночного режима
│   ├── backtest/                 # ✅ ВЫНЕСЕНО ИЗ DATA!
│   │   ├── engine.py             # Симуляция с комиссиями, проскальзыванием, SL/TP
│   │   └── metrics.py            # Sharpe, Max DD, Profit Factor, Ralph Vince metrics
│   ├── strategy/                 # ✅ Торговые логики по направлениям
│   │   ├── afml/                 # Базовая реализация идей López de Prado
│   │   ├── nowcasting/           # Прогнозирование экстремумов
│   │   ├── super_candles/        # Логика анализа супер-свечей
│   │   └── order_flow/           # 🚀 Основной фокус: логика на основе потока ордеров
│   ├── risk/                     # ✅ Управление капиталом (новый модуль)
│   │   ├── capital_management.py # Реализация Ralph Vince optimal f, Kelly
│   │   └── position_sizing.py    # Динамическое определение размера позиции
│   └── utils/                    # Логирование, визуализация, хелперы
│
├── research/                     # ✅ НАУКА: Jupyter Notebooks только для экспериментов
│   ├── afml/                     # Эксперименты с Purged CV, feature importance
│   ├── nowcasting/               # Визуализация и проверка гипотез об экстремумах
│   ├── super_candles/            # Анализ статистик supercandles
│   └── order_flow/               # Глубокий EDA и прототипирование (основной фокус)
│
├── live/                         # ✅ ПРОДАКШЕН: Модуль лайв-торговли
│   ├── engine.py                 # Главный цикл live-торговли
│   ├── bcs_client.py             # Интеграция с BCS API
│   └── reconciliation.py         # Сверка позиций и обработка разрывов
│
├── app/                          # Streamlit веб-приложение (Dashboard)
│   ├── dashboard.py
│   └── pages/
│
├── tests/                        # ✅ КРИТИЧНО ДЛЯ ВКР И РЕЗЮМЕ: Unit и интеграционные тесты
│   ├── test_pipeline.py          # Проверка корректности построения Dollar Bars
│   ├── test_labeling.py          # Проверка Triple Barrier на sample_data
│   └── test_backtest.py          # Проверка корректного вычета комиссий
│
├── docs/                         # ✅ ДЛЯ ВКР: Документация
│   ├── architecture.md           # Описание архитектуры системы
│   ├── methodology.md            # Математическое обоснование (AFML, Ralph Vince)
│   └── results.md                # Таблицы с метриками бэктестов
│
├── models/                       # Артефакты обученных моделей (или .gitkeep для DVC)
├── .gitignore                    # Игнорируем data/raw, data/interim, data/processed, .env
├── .env.example                  # Шаблон для API ключей
├── pyproject.toml                # ✅ Современный стандарт управления зависимостями Python
├── requirements.txt              # Для обратной совместимости
└── README.md                     # Главный файл с описанием проекта
