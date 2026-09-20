"""Independent live trading engine.

The engine owns market-data polling, strategy processing and BCS execution.
Streamlit is only a monitor/controller and does not sit on the critical path.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Optional

import pandas as pd

from .bcs_executor import BCSExecutor
from .bcs_models import BCSOrder

logger = logging.getLogger("jerichon.live_engine")


class LiveEngine:
    def __init__(self, provider, strategies: Dict[str, object],
                 normalizer, tail_size: int = 10_000,
                 poll_seconds: float = 2.0, candle_loader=None):
        self.provider = provider
        self.strategies = strategies
        self.normalize_trades = normalizer
        self.tail_size = int(tail_size)
        self.poll_seconds = float(poll_seconds)
        self.candle_loader = candle_loader
        self.executor = BCSExecutor()
        self.automated_trading = False
        self.paused_tickers = set()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        self.last_update: Optional[datetime] = None
        self.poll_count = 0
        self.errors = []
        self.meta = {t: {} for t in strategies}
        # Execution intent state.  A failed LIVE submission must remain retryable;
        # a successful submission is marked exactly once.  clientOrderId is
        # stable for the lifetime of an intent, so a retry after a timeout does
        # not manufacture a new logical JERICHON order id.
        self._executed_intents = set()
        self._pending_client_ids = {}
        self._pending_close_intents = {}
        self._pending_open_intents = {}

    @property
    def execution_mode(self) -> str:
        return self.executor.mode

    def set_paused(self, ticker: str, paused: bool) -> None:
        with self.lock:
            if paused:
                self.paused_tickers.add(ticker)
            else:
                self.paused_tickers.discard(ticker)

    def set_automated_trading(self, enabled: bool) -> None:
        with self.lock:
            self.automated_trading = bool(enabled)
            logger.warning("AUTOMATED TRADING %s", "ON" if enabled else "OFF")

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "automated_trading": self.automated_trading,
                "execution_mode": self.execution_mode,
                "last_update": self.last_update,
                "poll_count": self.poll_count,
                "errors": list(self.errors[-50:]),
                "meta": {k: dict(v) for k, v in self.meta.items()},
            }

    def start(self) -> None:
        with self.lock:
            if self.running:
                return
            self.running = True
            self.thread = threading.Thread(target=self._run, name="jerichon-live-engine", daemon=True)
            self.thread.start()

    def stop(self) -> None:
        with self.lock:
            self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)

    def _run(self) -> None:
        while True:
            with self.lock:
                if not self.running:
                    break
            started = time.monotonic()
            try:
                self._poll_once()
            except Exception as exc:
                logger.exception("live engine poll failed")
                with self.lock:
                    self.errors.append(f"engine: {exc}")
                    self.errors = self.errors[-50:]
            elapsed = time.monotonic() - started
            time.sleep(max(0.05, self.poll_seconds - elapsed))

    def _fetch_one(self, ticker: str):
        today = datetime.now(ZoneInfo("Europe/Moscow")).date().isoformat()
        raw = self.provider.fetch_recent_trades(ticker, date=today, tail_size=self.tail_size)
        return ticker, self.normalize_trades(raw)

    def _poll_once(self) -> None:
        tickers = list(self.strategies)
        # Network calls are parallelized; strategy state is still mutated under lock.
        fetched = {}
        with ThreadPoolExecutor(max_workers=max(1, min(6, len(tickers)))) as pool:
            futures = {pool.submit(self._fetch_one, t): t for t in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    _, df = future.result()
                    fetched[ticker] = df
                except Exception as exc:
                    with self.lock:
                        self.errors.append(f"{ticker}: trades: {exc}")

        with self.lock:
            for ticker in tickers:
                df = fetched.get(ticker, pd.DataFrame())
                strategy = self.strategies[ticker]
                if ticker in self.paused_tickers:
                    self.meta[ticker] = {"trades_received": len(df), "trades_processed": 0, "last_trade_ts": df["ts"].iloc[-1] if not df.empty else None, "error": None, "paused": True}
                    continue
                before_closed = len(strategy.trades)
                before_position = dict(strategy.position) if strategy.position else None
                accepted = 0
                if not df.empty:
                    accepted = strategy.process_trades(df.to_dict("records"))
                self._handle_execution_events(ticker, strategy, before_closed, before_position)
                self.meta[ticker] = {
                    "trades_received": len(df),
                    "trades_processed": int(accepted),
                    "last_trade_ts": df["ts"].iloc[-1] if not df.empty else None,
                    "error": None,
                }
            self.poll_count += 1
            self.last_update = datetime.now(ZoneInfo("Europe/Moscow"))

    def _handle_execution_events(self, ticker, strategy, before_closed, before_position):
        # Register newly closed-trade intents.  Keep failed submissions in the
        # pending map so the next engine cycle can retry them.
        for trade in strategy.trades[before_closed:]:
            key = (
                ticker,
                "close",
                str(trade.get("entry_time")),
                str(trade.get("exit_time")),
                trade.get("side"),
                float(trade.get("exit_price", 0.0)),
                trade.get("exit_reason"),
            )
            self._pending_close_intents.setdefault(key, trade)

        # Register an open intent whenever a local position exists and its
        # corresponding intent has not been successfully submitted yet.  This
        # deliberately does not depend on before_position, because a failed
        # BCS request leaves the local strategy position open and must be
        # retried.
        current = strategy.position
        if current is not None:
            key = (
                ticker,
                "open",
                str(current.get("entry_time")),
                current.get("side"),
                float(current.get("entry_price", 0.0)),
            )
            self._pending_open_intents.setdefault(key, current)

        if not self.automated_trading:
            return

        # Submit pending opens first.
        for key, current in list(self._pending_open_intents.items()):
            if key in self._executed_intents:
                continue
            if key[0] != ticker:
                continue
            ok = self._submit(ticker,
                              "buy" if current["side"] == "long" else "sell",
                              current,
                              is_open=True,
                              intent_key=key)
            if ok:
                self._executed_intents.add(key)
                self._pending_open_intents.pop(key, None)

        # Submit pending closes.
        for key, trade in list(self._pending_close_intents.items()):
            if key in self._executed_intents:
                continue
            if key[0] != ticker:
                continue
            ok = self._submit(ticker,
                              "sell" if trade["side"] == "long" else "buy",
                              trade,
                              is_open=False,
                              intent_key=key)
            if ok:
                self._executed_intents.add(key)
                self._pending_close_intents.pop(key, None)

    def _submit(self, ticker, side, obj, is_open: bool, intent_key=None) -> bool:
        if is_open:
            shares = int(obj.get("shares", 0))
            lots = int(obj.get("lots", 0))
            price = None
            reason = "strategy_open"
        else:
            shares = int(obj.get("shares", obj.get("lots", 0)))
            # Older trade records may not contain shares; strategy lot_size is used below.
            if shares <= 0:
                strategy = self.strategies[ticker]
                shares = int(obj.get("lots", 0) * strategy.lot_size)
            lots = int(obj.get("lots", 0))
            price = float(obj.get("exit_price"))
            reason = str(obj.get("exit_reason", "strategy_close"))
        if shares <= 0:
            return
        client_order_id = None
        if intent_key is not None:
            client_order_id = self._pending_client_ids.setdefault(
                intent_key, str(uuid.uuid4())
            )

        order = BCSOrder(
            ticker=ticker,
            side=side,
            lots=lots,
            shares=shares,
            price=price,
            order_type="market",
            client_order_id=client_order_id,
            reason=reason,
            class_code=str(getattr(self.executor, "default_class_code", "TQBR")),
            metadata={
                "source": "JERICHON",
                "automated": True,
                "intent_key": repr(intent_key),
            },
        )
        result = self.executor.submit(order)
        logger.warning(
            "BCS execution %s %s %s ok=%s status=%s id=%s error=%s raw=%s",
            ticker, side.upper(), reason, result.ok, result.status,
            result.order_id, result.error, result.raw,
        )
        return bool(result.ok)
