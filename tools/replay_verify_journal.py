"""
JERICHON Replay + Verify + Journal
===================================
Единый инструмент для:
1. Replay исторических данных через LiveStrategy (боевой режим на истории)
2. Сверка с research-пайплайном (бары + CUSUM + сигналы)
3. Журнал всех событий и сделок
4. Агрегация портфеля по нескольким тикерам

Использование:
    python tools/replay_verify_journal.py --config portfolio.json --date 2026-08-31
    python tools/replay_verify_journal.py --ticker GAZP --capital 100000 --date 2026-08-31

Формат portfolio.json:
    {"GAZP": 100000, "SMLT": 100000, "YDEX": 100000}

Результаты в data/journal/{date}/:
    - events_{TICKER}.parquet    — все CUSUM события и сигналы
    - trades_{TICKER}.parquet    — все закрытые сделки с PnL
    - verify_{TICKER}.json       — результаты сверки с research
    - portfolio.json             — сводка по портфелю
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Пути проекта
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data.for_ml.providers.algopack_provider import AlgoPackProvider
from data.for_ml.builders.dollar_bars import InformationDrivenBars
from data.for_ml.labeling.cusum import cusum_filter
from live.live_pipeline import LiveStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jerichon.replay_journal")


# =============================================================================
# 1. ЗАГРУЗКА КОНФИГА И АРТЕФАКТОВ
# =============================================================================

def load_config(path: Path) -> Dict[str, float]:
    """Загружает конфиг портфеля {ticker: capital}."""
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Конфиг должен быть словарём {ticker: capital}")
    return {str(k).upper(): float(v) for k, v in data.items()}


def find_model_artifacts(ticker: str) -> Tuple[Path, Path, Path]:
    """
    Ищет артефакты модели в models/params/ и models/.
    Возвращает (model_path, params_path, features_path).
    """
    search_dirs = [
        ROOT / "models" / "params",
        ROOT / "models",
    ]
    model_path = params_path = features_path = None
    for d in search_dirs:
        if not d.exists():
            continue
        if model_path is None:
            candidates = [
                p for p in d.glob(f"{ticker}_*.pkl")
                if "_features" not in p.stem and "_params" not in p.stem
            ]
            if candidates:
                model_path = sorted(candidates)[-1]
        if params_path is None:
            p = d / f"{ticker}_params.json"
            if p.exists():
                params_path = p
        if features_path is None:
            p = d / f"{ticker}_features.pkl"
            if p.exists():
                features_path = p
    if not all([model_path, params_path, features_path]):
        missing = []
        if not model_path: missing.append("model .pkl")
        if not params_path: missing.append("params .json")
        if not features_path: missing.append("features .pkl")
        raise FileNotFoundError(
            f"Артефакты для {ticker} не найдены. Отсутствует: {', '.join(missing)}"
        )
    return model_path, params_path, features_path


# =============================================================================
# 2. ЗАГРУЗКА ДАННЫХ
# =============================================================================

def get_available_dates(ticker: str) -> List[str]:
    """Возвращает отсортированный список дат с данными для тикера."""
    base = ROOT / "data" / "for_ml" / "storage" / "history" / "realtime" / ticker
    if not base.exists():
        return []
    return sorted([d.name for d in base.iterdir() if d.is_dir()])


def load_trades_for_ticker(
    ticker: str,
    target_date: str,
    history_days: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает trades для warmup (history_days до target_date) и replay (target_date).
    Возвращает (warmup_df, replay_df).
    """
    all_dates = get_available_dates(ticker)
    if not all_dates:
        raise ValueError(f"Нет данных для тикера {ticker}")
    if target_date not in all_dates:
        logger.warning(
            f"Дата {target_date} недоступна для {ticker}. "
            f"Доступные: {all_dates[-5:]}. Использую {all_dates[-1]}."
        )
        target_date = all_dates[-1]
    idx = all_dates.index(target_date)
    warmup_dates = all_dates[max(0, idx - history_days):idx]
    replay_dates = [target_date]
    base = ROOT / "data" / "for_ml" / "storage" / "history" / "realtime" / ticker

    def load_dates(dates: List[str]) -> pd.DataFrame:
        frames = []
        for d in dates:
            path = base / d / "trades.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                if not df.empty:
                    frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    warmup_df = load_dates(warmup_dates)
    replay_df = load_dates(replay_dates)
    logger.info(
        f"{ticker}: warmup={len(warmup_df):,} trades ({len(warmup_dates)} дней), "
        f"replay={len(replay_df):,} trades ({target_date})"
    )
    return warmup_df, replay_df

# =============================================================================
# 3. RESEARCH ПАЙПЛАЙН (для сверки)
# =============================================================================

def build_research_pipeline(
    trades_df: pd.DataFrame,
    dollar_threshold: float,
    cusum_threshold: float,
) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Строит research-бары и CUSUM события на тех же данных, что и live.
    Возвращает (research_bars, research_events).
    """
    if trades_df.empty:
        return pd.DataFrame(), pd.DatetimeIndex([])

    df = trades_df.copy()
    
    # Приводим названия колонок к нижнему регистру (как в исследовательском ноутбуке)
    col_map = {}
    for col in df.columns:
        col_upper = col.upper()
        if col_upper == "PRICE":
            col_map[col] = "price"
        elif col_upper == "VALUE":
            col_map[col] = "value"
        elif col_upper == "QUANTITY":
            col_map[col] = "quantity"
    if col_map:
        df = df.rename(columns=col_map)
    
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("Europe/Moscow")
    else:
        df["ts"] = df["ts"].dt.tz_convert("Europe/Moscow")

    # quantity_for_bars = VALUE / PRICE (учёт лотов, как в research)
    if "value" in df.columns and "price" in df.columns:
        df["quantity_for_bars"] = df["value"] / df["price"]
    else:
        return pd.DataFrame(), pd.DatetimeIndex([])

    trades_for_bars = df[["ts", "price", "quantity_for_bars"]].dropna()
    trades_for_bars = trades_for_bars.rename(columns={"quantity_for_bars": "quantity"})
    trades_for_bars = trades_for_bars[
        (trades_for_bars["price"] > 0) & (trades_for_bars["quantity"] > 0)
    ].copy()

    # Research-бары
    bar_builder = InformationDrivenBars()
    research_bars = bar_builder.build_dollar(trades_for_bars, threshold=dollar_threshold)
    if research_bars.empty:
        return pd.DataFrame(), pd.DatetimeIndex([])

    research_bars["ts"] = pd.to_datetime(research_bars["close_time"])
    if research_bars["ts"].dt.tz is None:
        research_bars["ts"] = research_bars["ts"].dt.tz_localize("Europe/Moscow")
    else:
        research_bars["ts"] = research_bars["ts"].dt.tz_convert("Europe/Moscow")
    research_bars = research_bars.set_index("ts").sort_index()

    # Research CUSUM
    close_series = research_bars["close"]
    research_events = cusum_filter(close_series, threshold=cusum_threshold)

    return research_bars, research_events

# =============================================================================
# 4. REPLAY ЧЕРЕЗ LiveStrategy
# =============================================================================

def run_replay_for_ticker(
    ticker: str,
    capital: float,
    target_date: str,
    history_days: int = 5,
) -> Dict[str, Any]:
    """
    Запускает replay для одного тикера.
    Возвращает словарь с результатами: events, trades, bars, signals, position, verify.
    """
    logger.info(f"{'='*60}")
    logger.info(f"REPLAY: {ticker} | capital={capital:,.0f} | date={target_date}")
    logger.info(f"{'='*60}")

    # 1. Загружаем артефакты
    model_path, params_path, features_path = find_model_artifacts(ticker)
    logger.info(f"Модель: {model_path.name}")

    # 2. Загружаем данные
    warmup_df, replay_df = load_trades_for_ticker(ticker, target_date, history_days)
    if warmup_df.empty and replay_df.empty:
        raise ValueError(f"Нет данных для {ticker}")

    # 3. Нормализуем через провайдер
    provider = AlgoPackProvider(mock=True)

    # 4. Создаём стратегию
    strategy = LiveStrategy(
        ticker=ticker,
        model_path=model_path,
        params_path=params_path,
        features_path=features_path,
        initial_capital=capital,
    )

    # 5. Warmup
    if not warmup_df.empty:
        warmup_trades = provider._normalize_trades(warmup_df.copy(), ticker)
        warm_result = strategy.warmup(warmup_trades)
        logger.info(
            f"Warmup: bars={warm_result['bars']}, ready={warm_result['ready']}"
        )
        if not warm_result['ready']:
            logger.warning(f"Warmup не завершён для {ticker}!")

    # 6. Replay текущего дня + сбор событий
    events_log: List[Dict[str, Any]] = []
    last_price = None
    last_ts = None
    last_signal_value = 0

    if not replay_df.empty:
        replay_trades = provider._normalize_trades(replay_df.copy(), ticker)

        for _, row in replay_trades.iterrows():
            trade_dict = row.to_dict()
            last_price = float(trade_dict.get('price', trade_dict.get('PRICE', 0)))
            last_ts = trade_dict.get('ts')

            bars_before = len(strategy.dollar_bars)
            events_before = len(strategy.cusum_events)
            strategy.process_trade(trade_dict)
            bars_after = len(strategy.dollar_bars)
            events_after = len(strategy.cusum_events)

            # Записываем новое CUSUM событие
            if events_after > events_before:
                last_event = strategy.cusum_events[-1]
                events_log.append({
                    'ts': last_event,
                    'event_type': 'cusum_event',
                    'bar_number': bars_after - 1,
                    'price': last_price,
                    'signal': strategy.last_signal,
                })

            # Записываем сигнал модели (если изменился)
            if strategy.last_signal != 0 and strategy.last_signal != last_signal_value:
                events_log.append({
                    'ts': strategy.last_signal_time or last_ts,
                    'event_type': 'signal',
                    'bar_number': bars_after - 1,
                    'price': last_price,
                    'signal': strategy.last_signal,
                })
                last_signal_value = strategy.last_signal

    # 7. Принудительное закрытие позиции в конце дня
    if strategy.position is not None and last_price is not None and last_ts is not None:
        strategy._close_position(last_price, last_ts, reason="end_of_day")
        logger.info(f"FORCE CLOSE: закрыли позицию в конце дня по {last_price:.4f}")

    # 8. Собираем результаты
    events_df = pd.DataFrame(events_log) if events_log else pd.DataFrame(
        columns=['ts', 'event_type', 'bar_number', 'price', 'signal']
    )
    trades_df = pd.DataFrame(strategy.trades) if strategy.trades else pd.DataFrame(
        columns=['entry_time', 'exit_time', 'side', 'entry_price', 'exit_price',
                 'lots', 'pnl', 'pnl_pct', 'exit_reason']
    )
    if not trades_df.empty:
        trades_df['ticker'] = ticker

    # 9. Сверка с research
    verify_result = verify_parity(
        ticker=ticker,
        strategy=strategy,
        warmup_df=warmup_df,
        replay_df=replay_df,
        provider=provider,
    )

    # 10. Метрики
    metrics = compute_metrics(trades_df, capital)

    logger.info(
        f"Результат: events={len(events_df)}, trades={len(trades_df)}, "
        f"pnl={metrics['pnl']:+.2f} ({metrics['pnl_pct']:+.2f}%)"
    )
    logger.info(
        f"Сверка: bars_live={verify_result['live_bars_count']}, "
        f"bars_research={verify_result['research_bars_count']}, "
        f"bars_match={verify_result['bars_match']}, "
        f"events_live={verify_result['live_events_count']}, "
        f"events_research={verify_result['research_events_count']}, "
        f"events_match={verify_result['events_match']}"
    )

    return {
        'ticker': ticker,
        'initial_capital': capital,
        'final_capital': strategy.capital,
        'events': events_df,
        'trades': trades_df,
        'bars': pd.DataFrame(strategy.dollar_bars) if strategy.dollar_bars else pd.DataFrame(),
        'cusum_events': strategy.cusum_events,
        'position': strategy.position,
        'metrics': metrics,
        'warmup_done': strategy.warmup_done,
        'verify': verify_result,
    }


# =============================================================================
# 5. СВЕРКА С RESEARCH
# =============================================================================

def verify_parity(
    ticker: str,
    strategy: LiveStrategy,
    warmup_df: pd.DataFrame,
    replay_df: pd.DataFrame,
    provider: AlgoPackProvider,
) -> Dict[str, Any]:
    """
    Сравнивает live-бары и CUSUM с research-пайплайном на тех же данных.
    """
    # Объединяем warmup + replay для research
    all_frames = []
    if not warmup_df.empty:
        all_frames.append(warmup_df)
    if not replay_df.empty:
        all_frames.append(replay_df)
    if not all_frames:
        return {
            'bars_match': False, 'events_match': False,
            'live_bars_count': 0, 'research_bars_count': 0,
            'live_events_count': 0, 'research_events_count': 0,
            'error': 'no data',
        }

    all_df = pd.concat(all_frames, ignore_index=True)

    # Research-бары и CUSUM
    research_bars, research_events = build_research_pipeline(
        all_df,
        dollar_threshold=strategy.dollar_threshold,
        cusum_threshold=strategy.cusum_threshold,
    )

    # Live-бары и CUSUM
    live_bars_count = len(strategy.dollar_bars)
    live_events_count = len(strategy.cusum_events)
    research_bars_count = len(research_bars)
    research_events_count = len(research_events)

    # Сверка количества (допускаем расхождение в 1-2 бара из-за incomplete bar)
    bars_match = abs(live_bars_count - research_bars_count) <= 2
    events_match = abs(live_events_count - research_events_count) <= 2

    # Сверка timestamps событий (если есть)
    events_timestamps_match = False
    if live_events_count > 0 and research_events_count > 0:
        live_ts = sorted([pd.Timestamp(t) for t in strategy.cusum_events])
        research_ts = sorted([pd.Timestamp(t) for t in research_events])
        # Сравниваем последние N событий (первые могут отличаться из-за warmup)
        n_compare = min(20, len(live_ts), len(research_ts))
        live_tail = live_ts[-n_compare:]
        research_tail = research_ts[-n_compare:]
        # Допускаем расхождение в 1-2 события
        common = set(live_tail) & set(research_tail)
        events_timestamps_match = len(common) >= n_compare - 2

    return {
        'bars_match': bars_match,
        'events_match': events_match,
        'events_timestamps_match': events_timestamps_match,
        'live_bars_count': live_bars_count,
        'research_bars_count': research_bars_count,
        'live_events_count': live_events_count,
        'research_events_count': research_events_count,
        'common_events': len(set(pd.Timestamp(t) for t in strategy.cusum_events) &
                             set(pd.Timestamp(t) for t in research_events)) if (live_events_count > 0 and research_events_count > 0) else 0,
    }


# =============================================================================
# 6. МЕТРИКИ
# =============================================================================

def compute_metrics(trades_df: pd.DataFrame, initial_capital: float) -> Dict[str, float]:
    """Вычисляет метрики по сделкам."""
    if trades_df.empty:
        return {
            'pnl': 0.0, 'pnl_pct': 0.0, 'trades_count': 0,
            'win_rate': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0,
            'profit_factor': 0.0,
        }
    pnl = trades_df['pnl'].sum()
    pnl_pct = (pnl / initial_capital) * 100
    trades_count = len(trades_df)
    win_rate = (trades_df['pnl'] > 0).mean() * 100
    returns = trades_df['pnl_pct'] / 100
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    cum_pnl = trades_df['pnl'].cumsum()
    peak = cum_pnl.cummax()
    drawdown = cum_pnl - peak
    max_dd = (drawdown.min() / initial_capital) * 100 if initial_capital > 0 else 0.0
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    return {
        'pnl': float(pnl), 'pnl_pct': float(pnl_pct),
        'trades_count': int(trades_count), 'win_rate': float(win_rate),
        'sharpe': float(sharpe), 'max_drawdown': float(max_dd),
        'profit_factor': float(profit_factor),
    }


# =============================================================================
# 7. ЗАПИСЬ ЖУРНАЛА
# =============================================================================

def write_journal(
    ticker: str,
    date: str,
    events_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    verify_result: Dict[str, Any],
    outdir: Path,
) -> None:
    """Записывает журнал событий, сделок и сверки."""
    journal_dir = outdir / date
    journal_dir.mkdir(parents=True, exist_ok=True)

    if not events_df.empty:
        events_path = journal_dir / f"events_{ticker}.parquet"
        events_df.to_parquet(events_path, index=False)
        logger.info(f"События записаны: {events_path} ({len(events_df)} записей)")

    if not trades_df.empty:
        trades_path = journal_dir / f"trades_{ticker}.parquet"
        trades_df.to_parquet(trades_path, index=False)
        logger.info(f"Сделки записаны: {trades_path} ({len(trades_df)} записей)")

    verify_path = journal_dir / f"verify_{ticker}.json"
    with open(verify_path, "w", encoding="utf-8") as f:
        json.dump(verify_result, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Сверка записана: {verify_path}")


def write_portfolio(
    date: str,
    results: Dict[str, Dict[str, Any]],
    outdir: Path,
) -> None:
    """Записывает сводку по портфелю в JSON."""
    journal_dir = outdir / date
    journal_dir.mkdir(parents=True, exist_ok=True)

    portfolio = {
        'date': date,
        'tickers': {},
        'portfolio': {
            'total_initial': 0.0, 'total_final': 0.0,
            'total_pnl': 0.0, 'total_pnl_pct': 0.0,
            'total_trades': 0,
        },
        'verification': {
            'all_bars_match': True,
            'all_events_match': True,
        },
    }

    for ticker, res in results.items():
        metrics = res['metrics']
        verify = res['verify']
        portfolio['tickers'][ticker] = {
            'initial_capital': res['initial_capital'],
            'final_capital': res['final_capital'],
            'pnl': metrics['pnl'], 'pnl_pct': metrics['pnl_pct'],
            'trades_count': metrics['trades_count'],
            'win_rate': metrics['win_rate'],
            'sharpe': metrics['sharpe'],
            'max_drawdown': metrics['max_drawdown'],
            'profit_factor': metrics['profit_factor'],
            'warmup_done': res['warmup_done'],
            'open_position': res['position'] is not None,
            'verify': {
                'bars_match': verify['bars_match'],
                'events_match': verify['events_match'],
                'live_bars': verify['live_bars_count'],
                'research_bars': verify['research_bars_count'],
                'live_events': verify['live_events_count'],
                'research_events': verify['research_events_count'],
            },
        }
        portfolio['portfolio']['total_initial'] += res['initial_capital']
        portfolio['portfolio']['total_final'] += res['final_capital']
        portfolio['portfolio']['total_pnl'] += metrics['pnl']
        portfolio['portfolio']['total_trades'] += metrics['trades_count']
        if not verify['bars_match']:
            portfolio['verification']['all_bars_match'] = False
        if not verify['events_match']:
            portfolio['verification']['all_events_match'] = False

    if portfolio['portfolio']['total_initial'] > 0:
        portfolio['portfolio']['total_pnl_pct'] = (
            portfolio['portfolio']['total_pnl'] / portfolio['portfolio']['total_initial']
        ) * 100

    portfolio_path = journal_dir / "portfolio.json"
    with open(portfolio_path, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Портфель записан: {portfolio_path}")

    # Вывод сводки
    print("\n" + "="*90)
    print(f"ПОРТФЕЛЬ | {date}")
    print("="*90)
    print(f"{'Тикер':<8} {'Капитал':>12} {'PnL':>12} {'PnL%':>8} {'Сделок':>8} {'Win%':>8} {'Sharpe':>8} {'Bars✓':>6} {'Events✓':>8}")
    print("-"*90)
    for ticker, data in portfolio['tickers'].items():
        v = data['verify']
        print(
            f"{ticker:<8} "
            f"{data['final_capital']:>12,.2f} "
            f"{data['pnl']:>+12,.2f} "
            f"{data['pnl_pct']:>+7.2f}% "
            f"{data['trades_count']:>8d} "
            f"{data['win_rate']:>7.1f}% "
            f"{data['sharpe']:>8.2f} "
            f"{'✓' if v['bars_match'] else '✗':>6} "
            f"{'✓' if v['events_match'] else '✗':>8}"
        )
    print("-"*90)
    p = portfolio['portfolio']
    print(
        f"{'ИТОГО':<8} "
        f"{p['total_final']:>12,.2f} "
        f"{p['total_pnl']:>+12,.2f} "
        f"{p['total_pnl_pct']:>+7.2f}% "
        f"{p['total_trades']:>8d}"
    )
    print("="*90)
    v = portfolio['verification']
    print(f"Сверка с research: Bars={'✓ PASS' if v['all_bars_match'] else '✗ FAIL'}, "
          f"Events={'✓ PASS' if v['all_events_match'] else '✗ FAIL'}")
    print("="*90 + "\n")


# =============================================================================
# 8. MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="JERICHON Replay + Verify + Journal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="Путь к portfolio.json")
    parser.add_argument("--ticker", type=str, help="Тикер для одиночного запуска")
    parser.add_argument("--capital", type=float, default=100_000.0,
                        help="Капитал для одиночного тикера (по умолчанию 100000)")
    parser.add_argument("--date", type=str, required=True, help="Дата для replay (YYYY-MM-DD)")
    parser.add_argument("--history-days", type=int, default=5,
                        help="Сколько дней истории для warmup (по умолчанию 5)")
    parser.add_argument("--outdir", type=Path, default=ROOT / "data" / "journal",
                        help="Директория для записи журнала")

    args = parser.parse_args()

    if args.config:
        tickers_config = load_config(args.config)
    elif args.ticker:
        tickers_config = {args.ticker.upper(): args.capital}
    else:
        logger.error("Укажите --config или --ticker")
        return 2

    logger.info(f"Дата replay: {args.date}")
    logger.info(f"Тикеры: {list(tickers_config.keys())}")
    logger.info(f"History days: {args.history_days}")
    logger.info(f"Outdir: {args.outdir}")

    results = {}
    for ticker, capital in tickers_config.items():
        try:
            result = run_replay_for_ticker(
                ticker=ticker, capital=capital,
                target_date=args.date, history_days=args.history_days,
            )
            results[ticker] = result
            write_journal(
                ticker=ticker, date=args.date,
                events_df=result['events'], trades_df=result['trades'],
                verify_result=result['verify'], outdir=args.outdir,
            )
        except Exception as e:
            logger.error(f"Ошибка для {ticker}: {e}", exc_info=True)
            continue

    if not results:
        logger.error("Ни один тикер не обработан успешно")
        return 1

    write_portfolio(args.date, results, args.outdir)
    logger.info("Replay завершён успешно")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())