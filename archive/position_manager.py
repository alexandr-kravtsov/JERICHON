# live/position_manager.py
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import sys
from pathlib import Path
# Добавляем путь к папке data, чтобы import for_ml работал
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data"))

class PositionManager:
    """
    Управление позициями: открытие, закрытие, стоп-лосс, тейк-профит, переворот.
    """
    
    def __init__(self, initial_capital: float = 100000.0, risk_per_trade: float = 0.01):
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade  # доля капитала, риск на сделку
        self.position = None  # {'type': 'long'/'short', 'entry': price, 'lots': int, 'sl': price, 'tp': price, 'time': timestamp}
        self.trades = []
        self.equity_history = []
        self._record_equity()
    
    def _record_equity(self):
        """Записывает текущий капитал в историю."""
        self.equity_history.append({'timestamp': pd.Timestamp.now(), 'equity': self.capital})
    
    def open_position(self, signal: int, price: float, timestamp, sl_pct: float = 0.02, tp_pct: float = 0.06):
        """Открывает позицию: signal = 1 (long) или -1 (short)."""
        if self.position is not None:
            raise ValueError("Позиция уже открыта")
        
        # Рассчитываем размер позиции на основе риска
        risk_amount = self.capital * self.risk_per_trade
        if signal == 1:  # long
            stop_loss = price * (1 - sl_pct)
            take_profit = price * (1 + tp_pct)
            loss_per_share = price - stop_loss
        else:  # short
            stop_loss = price * (1 + sl_pct)
            take_profit = price * (1 - tp_pct)
            loss_per_share = stop_loss - price
        
        if loss_per_share <= 0:
            raise ValueError("Неверный стоп-лосс")
        
        lots = int(risk_amount / loss_per_share)
        if lots == 0:
            lots = 1  # минимум 1 лот
        
        self.position = {
            'type': 'long' if signal == 1 else 'short',
            'entry': price,
            'lots': lots,
            'sl': stop_loss,
            'tp': take_profit,
            'time': timestamp
        }
    
    def close_position(self, price: float, timestamp, reason: str = 'manual'):
        """Закрывает текущую позицию по цене."""
        if self.position is None:
            return
        pos = self.position
        if pos['type'] == 'long':
            pnl_per_share = price - pos['entry']
        else:
            pnl_per_share = pos['entry'] - price
        total_pnl = pnl_per_share * pos['lots']
        self.capital += total_pnl
        self.trades.append({
            'entry_time': pos['time'],
            'exit_time': timestamp,
            'type': pos['type'],
            'entry_price': pos['entry'],
            'exit_price': price,
            'lots': pos['lots'],
            'pnl': total_pnl,
            'exit_reason': reason
        })
        self.position = None
        self._record_equity()
        return total_pnl
    
    def update(self, price: float, timestamp):
        """Обновляет состояние позиции: проверяет SL/TP."""
        if self.position is None:
            return None
        pos = self.position
        if pos['type'] == 'long':
            if price <= pos['sl']:
                return self.close_position(pos['sl'], timestamp, 'stop_loss')
            elif price >= pos['tp']:
                return self.close_position(pos['tp'], timestamp, 'take_profit')
        else:  # short
            if price >= pos['sl']:
                return self.close_position(pos['sl'], timestamp, 'stop_loss')
            elif price <= pos['tp']:
                return self.close_position(pos['tp'], timestamp, 'take_profit')
        return None
    
    def get_status(self, current_price: float):
        """Возвращает статус позиции (включая плавающий PnL)."""
        if self.position is None:
            return {'has_position': False}
        pos = self.position
        if pos['type'] == 'long':
            pnl = (current_price - pos['entry']) * pos['lots']
        else:
            pnl = (pos['entry'] - current_price) * pos['lots']
        return {
            'has_position': True,
            'type': pos['type'],
            'entry': pos['entry'],
            'lots': pos['lots'],
            'sl': pos['sl'],
            'tp': pos['tp'],
            'current_price': current_price,
            'pnl': pnl,
            'pnl_pct': pnl / (pos['entry'] * pos['lots']) * 100 if pos['lots'] > 0 else 0
        }