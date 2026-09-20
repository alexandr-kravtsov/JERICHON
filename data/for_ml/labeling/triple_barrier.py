"""
JERICHON Triple Barrier Labeling
=================================

Реализация метода тройного барьера из "Advances in Financial Machine Learning".
Превращает события CUSUM в метки (-1, 0, +1) для обучения моделей.

Алгоритм:
Для каждого события (t0):
    1. Определяется цена входа (close[t0]).
    2. Рассчитываются верхний и нижний барьеры на основе волатильности.
    3. Устанавливается вертикальный барьер (t1 = t0 + horizon баров).
    4. Отслеживается, какой из трёх барьеров был достигнут первым:
       - верхний → метка +1 (длинная позиция сработала)
       - нижний → метка -1 (короткая позиция сработала)
       - вертикальный → метка 0 (время истекло)

Вход:
    bars : pd.DataFrame с колонками HIGH, LOW, CLOSE
    events : pd.DatetimeIndex — моменты событий (из CUSUM)
    pt_sl : tuple (profit_taking, stop_loss) — множители для барьеров
    vertical_barrier : int — максимальная длина события в барах
    target_type : str — 'ewm' или 'rolling' для расчёта волатильности

Выход:
    pd.DataFrame с колонками:
        - event_time : время события
        - t1 : время вертикального барьера
        - target : волатильность на момент события
        - label : -1, 0, +1
        - ret : доходность до выхода
        - exit_time : время выхода
        - exit_price : цена выхода
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("jerichon.triple_barrier")
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Функция расчёта волатильности (target)
# ---------------------------------------------------------------------------
def _compute_target(
    close: pd.Series,
    window: int = 50,
    method: str = "ewm",
) -> pd.Series:
    """
    Вычисляет локальную волатильность лог-доходностей.

    Параметры
    ---------
    close : pd.Series
        Цены закрытия.
    window : int
        Окно для расчёта (span для ewm, размер окна для rolling).
    method : str
        'ewm' — экспоненциально взвешенное стандартное отклонение.
        'rolling' — скользящее стандартное отклонение.

    Возвращает
    -------
    pd.Series с волатильностью.
    """
    log_ret = np.log(close).diff().dropna()
    if method == "ewm":
        target = log_ret.ewm(span=window).std()
    elif method == "rolling":
        target = log_ret.rolling(window=window).std()
    else:
        raise ValueError(f"Неизвестный метод расчёта target: {method}")
    return target


# ---------------------------------------------------------------------------
# Основная функция Triple Barrier
# ---------------------------------------------------------------------------
def triple_barrier_labeling(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    pt_sl: Tuple[float, float] = (1.0, 1.0),
    vertical_barrier: int = 20,
    target_window: int = 50,
    target_method: str = "ewm",
) -> pd.DataFrame:
    """
    Разметка событий методом тройного барьера.

    Параметры
    ---------
    bars : pd.DataFrame
        Бары с колонками HIGH, LOW, CLOSE. Индекс — DatetimeIndex.
    events : pd.DatetimeIndex
        Моменты событий (из CUSUM фильтра).
    pt_sl : tuple (profit_taking, stop_loss)
        Множители для верхнего и нижнего барьеров.
        pt — profit taking, sl — stop loss.
    vertical_barrier : int
        Максимальное количество баров до принудительного закрытия.
    target_window : int
        Окно для расчёта волатильности (target).
    target_method : str
        'ewm' или 'rolling'.

    Возвращает
    -------
    pd.DataFrame с колонками:
        - event_time
        - t1 (вертикальный барьер)
        - target (волатильность)
        - upper (цена верхнего барьера)
        - lower (цена нижнего барьера)
        - label (-1, 0, +1)
        - ret (доходность)
        - exit_time
        - exit_price
        - exit_type ('upper', 'lower', 'vertical')
    """
    # Проверка наличия нужных колонок
    required_cols = {"HIGH", "LOW", "CLOSE"}
    missing = required_cols - set(bars.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки в bars: {missing}")

    # Расчёт волатильности
    target_series = _compute_target(bars["CLOSE"], window=target_window, method=target_method)

    # Фильтруем события — оставляем только те, что есть в bars.index
    events = events[events.isin(bars.index)]
    if len(events) == 0:
        logger.warning("Нет событий CUSUM в индексе баров")
        return pd.DataFrame()

    # Сортируем события по времени
    events = events.sort_values()

    # Получаем позиции событий в индексе баров
    bar_index = bars.index
    event_positions = bar_index.get_indexer(events)

    results = []
    n_bars = len(bars)

    for pos, event_time in zip(event_positions, events):
        if pos == -1:
            continue  # событие не найдено в индексе (маловероятно)

        # Цена входа
        entry_price = bars["CLOSE"].iloc[pos]

        # Волатильность на момент события
        target_val = target_series.loc[event_time] if event_time in target_series.index else np.nan
        if pd.isna(target_val) or target_val <= 0:
            logger.debug(f"Пропущено событие {event_time}: target = {target_val}")
            continue

        # Барьеры (в абсолютных ценах)
        pt, sl = pt_sl
        upper = entry_price * (1 + pt * target_val)
        lower = entry_price * (1 - sl * target_val)

        # Вертикальный барьер
        t1_pos = min(pos + vertical_barrier, n_bars - 1)
        t1_time = bar_index[t1_pos]

        # Срез баров от события до вертикального барьера
        slice_bars = bars.iloc[pos + 1 : t1_pos + 1]
        if len(slice_bars) == 0:
            continue

        # Определяем, какой барьер достигнут первым
        exit_type = "vertical"
        exit_time = t1_time
        exit_price = bars["CLOSE"].iloc[t1_pos]

        # Проверяем верхний барьер
        upper_reached = slice_bars["HIGH"] >= upper
        if upper_reached.any():
            first_idx = upper_reached.idxmax()  # первое True
            exit_type = "upper"
            exit_time = first_idx
            exit_price = upper

        # Проверяем нижний барьер (только если не достигнут верхний)
        if exit_type == "vertical":
            lower_reached = slice_bars["LOW"] <= lower
            if lower_reached.any():
                first_idx = lower_reached.idxmax()
                exit_type = "lower"
                exit_time = first_idx
                exit_price = lower

        # Доходность (логарифмическая)
        ret = np.log(exit_price / entry_price)

        # Метка
        if exit_type == "upper":
            label = 1
        elif exit_type == "lower":
            label = -1
        else:
            label = 0

        results.append({
            "event_time": event_time,
            "t1": t1_time,
            "target": target_val,
            "upper": upper,
            "lower": lower,
            "label": label,
            "ret": ret,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_type": exit_type,
        })

    if not results:
        logger.warning("Не удалось обработать ни одного события")
        return pd.DataFrame()

    events_df = pd.DataFrame(results)
    events_df.set_index("event_time", inplace=True)
    events_df.sort_index(inplace=True)

    # Логируем распределение меток
    label_counts = events_df["label"].value_counts()
    total = len(events_df)
    logger.info(
        f"Triple Barrier завершён: всего событий {total}. "
        f"+1: {label_counts.get(1, 0)} ({100*label_counts.get(1,0)/total:.1f}%), "
        f"-1: {label_counts.get(-1, 0)} ({100*label_counts.get(-1,0)/total:.1f}%), "
        f"0: {label_counts.get(0, 0)} ({100*label_counts.get(0,0)/total:.1f}%)"
    )

    return events_df