# live/risk_manager.py
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import sys
from pathlib import Path
# Добавляем путь к папке data, чтобы import for_ml работал
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data"))

class RiskManager:
    """
    Управление рисками на уровне портфеля и сделки.
    """
    
    def __init__(self, max_portfolio_risk: float = 0.05, max_position_risk: float = 0.02):
        self.max_portfolio_risk = max_portfolio_risk  # максимальный дневной убыток % от капитала
        self.max_position_risk = max_position_risk    # риск на сделку % от капитала
        self.daily_loss = 0.0
        self.initial_capital = None
    
    def check_daily_loss_limit(self, current_pnl: float) -> bool:
        """Проверяет, не превышен ли дневной лимит убытка."""
        if self.initial_capital is None:
            return True
        if -current_pnl / self.initial_capital > self.max_portfolio_risk:
            return False
        return True
    
    def calculate_position_size(self, capital: float, stop_loss_distance: float) -> int:
        """
        Рассчитывает размер позиции на основе риска и расстояния до стоп-лосса.
        """
        risk_amount = capital * self.max_position_risk
        if stop_loss_distance <= 0:
            return 1
        lots = int(risk_amount / stop_loss_distance)
        return max(1, lots)
    
    def update_daily_pnl(self, pnl: float):
        """Обновляет дневной PnL для контроля лимитов."""
        self.daily_loss = min(self.daily_loss, pnl)