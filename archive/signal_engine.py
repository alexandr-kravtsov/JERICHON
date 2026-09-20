# live/signal_engine.py
import pandas as pd
from typing import Dict, Any, Optional
import sys
from pathlib import Path
# Добавляем путь к папке data, чтобы import for_ml работал
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data"))

class SignalEngine:
    """
    Преобразует предсказания модели в торговые сигналы с учётом правил.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            "threshold": 0.65,          # минимальная вероятность для сигнала
            "min_bars_between_signals": 3,  # минимальное число баров между сигналами
            "allow_reversal": True,     # разрешать переворот позиции
        }
        self.last_signal_time = None
        self.current_signal = 0
    
    def process(self, df: pd.DataFrame, predictions: pd.Series, probabilities: pd.Series) -> pd.DataFrame:
        """
        Применяет правила к предсказаниям и возвращает DataFrame с колонками signal, probability, action.
        """
        df = df.copy()
        df['pred'] = predictions
        df['prob'] = probabilities
        
        # Применяем порог
        df['signal_raw'] = np.where(df['prob'] >= self.config['threshold'], df['pred'], 0)
        
        # Фильтр на минимальное количество баров между сигналами
        df['signal'] = 0
        last_idx = None
        for i, row in df.iterrows():
            if row['signal_raw'] != 0:
                if last_idx is None or (i - last_idx) >= self.config['min_bars_between_signals']:
                    df.loc[i, 'signal'] = row['signal_raw']
                    last_idx = i
                    if not self.config['allow_reversal']:
                        # Если запрещены перевороты, игнорируем противоположные сигналы
                        # (можно реализовать сложнее)
                        pass
        
        # Определяем действия
        df['action'] = 'HOLD'
        df.loc[df['signal'] == 1, 'action'] = 'BUY'
        df.loc[df['signal'] == -1, 'action'] = 'SELL'
        
        # Обновляем состояние
        if not df.empty and df['signal'].iloc[-1] != 0:
            self.current_signal = df['signal'].iloc[-1]
            self.last_signal_time = df.index[-1]
        
        return df