"""
==========================================================
JERICHON Quant Research Platform
Configuration
==========================================================

Единая конфигурация проекта.

Здесь НЕ должно быть никакой бизнес-логики.

Только:

- пути
- API
- список инструментов
- параметры хранения
- параметры ML
- параметры Dollar Bars
- параметры CUSUM
- параметры Triple Barrier
- параметры обучения

Все остальные модули импортируют настройки отсюда.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(r"D:\JERICHON")

DATA_DIR = PROJECT_ROOT / "data"

ML_DIR = DATA_DIR / "for_ml"

PROVIDERS_DIR = ML_DIR / "providers"

STORAGE_DIR = ML_DIR / "storage"

HISTORY_DIR = ML_DIR / "history"

MODELS_DIR = ML_DIR / "models"

LOGS_DIR = ML_DIR / "logs"

DATABASE_PATH = ML_DIR / "results.db"

API_ENV_PATH = PROJECT_ROOT / "API.env"

# создаем отсутствующие папки автоматически
for folder in [
    ML_DIR,
    PROVIDERS_DIR,
    STORAGE_DIR,
    HISTORY_DIR,
    MODELS_DIR,
    LOGS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# ==========================================================
# API
# ==========================================================

load_dotenv(API_ENV_PATH)

ALGOPACK_API_KEY = os.getenv("ALGOPACK_API_KEY")

BCS_TOKEN = os.getenv("BCS_TOKEN")

if ALGOPACK_API_KEY is None:
    raise ValueError(
        "ALGOPACK_API_KEY не найден в API.env"
    )

# ==========================================================
# TICKERS
# ==========================================================

# Пока работаем только с тремя.
# Позже список расширится до полного Universe.

DEFAULT_TICKERS = [

    "ROSN",

    "RUAL",

    "MAGN",

]

# Полный Universe фонда

ALL_TICKERS = [

    "SBER",
    "GAZP",
    "LKOH",
    "ROSN",
    "GMKN",
    "NVTK",
    "CHMF",
    "MAGN",
    "RUAL",
    "ALRS",
    "MTLR",
    "TATN",
    "VTBR",
    "MOEX",
    "PLZL",
    "SNGS"

]

# ==========================================================
# AlgoPack
# ==========================================================

ALGOPACK_BASE_URL = "https://apim.moex.com"

REQUEST_TIMEOUT = 60

REQUEST_RETRIES = 5

REQUEST_SLEEP = 0.5

PAGE_LIMIT = 5000

# ==========================================================
# STORAGE
# ==========================================================

PARQUET_COMPRESSION = "snappy"

SAVE_RAW_DATA = True

SAVE_MINUTE_SNAPSHOT = True

# ==========================================================
# MARKET DATA
# ==========================================================

CANDLE_INTERVAL = 1

BOARD = "TQBR"

ENGINE = "stock"

MARKET = "shares"

# ==========================================================
# DOLLAR BARS
# ==========================================================

DEFAULT_DOLLAR_THRESHOLD = 10_000_000

# ==========================================================
# CUSUM
# ==========================================================

CUSUM_THRESHOLD = 0.002

# ==========================================================
# TRIPLE BARRIER
# ==========================================================

TB_HORIZON = 20

TB_ATR_PERIOD = 21

TB_ATR_MULT = 1.5

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

LOOKBACK = 20

EMA_PERIODS = [

    5,

    10,

    20

]

# ==========================================================
# MODEL TRAINING
# ==========================================================

TEST_SIZE = 0.25

RANDOM_STATE = 13

N_OPTUNA_TRIALS = 15

CV_SPLITS = 5

# ==========================================================
# BACKTEST
# ==========================================================

COMMISSION = 0.0006

MAX_LEVERAGE = 3

SLIPPAGE = 0.0001

# ==========================================================
# LOGGING
# ==========================================================

LOG_LEVEL = "INFO"

# ==========================================================
# DATABASE
# ==========================================================

EXPERIMENT_TABLE = "experiments"

MODEL_TABLE = "models"

BACKTEST_TABLE = "backtests"

# ==========================================================
# VERSION
# ==========================================================

PROJECT_NAME = "JERICHON"

VERSION = "3.0"

AUTHOR = "Alex"
