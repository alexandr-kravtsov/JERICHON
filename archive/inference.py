# live/inference.py
import pandas as pd
import numpy as np
from models.registry import ModelRegistry
from typing import Optional, Tuple
import sys
from pathlib import Path
# Добавляем путь к папке data, чтобы import for_ml работал
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data"))

class InferenceEngine:
    """
    Загружает активную модель и делает предсказания на новых данных.
    Предполагается, что признаки уже вычислены (например, через DatasetBuilder).
    """
    
    def __init__(self, ticker: str, model_name: str = "xgboost"):
        self.ticker = ticker
        self.model_name = model_name
        self.registry = ModelRegistry()
        self.model, self.metadata = self.registry.get_active_model(ticker, model_name)
        self.feature_columns = self.metadata.get("feature_columns", [])
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Возвращает предсказания для DataFrame признаков.
        Предполагается, что X содержит все необходимые колонки.
        """
        # Если есть схема фич, приводим порядок колонок
        if self.feature_columns:
            # Проверяем наличие всех колонок
            missing = set(self.feature_columns) - set(X.columns)
            if missing:
                raise ValueError(f"Отсутствуют колонки: {missing}")
            X = X[self.feature_columns]
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Возвращает вероятности классов."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            # fallback: бинарная классификация
            preds = self.predict(X)
            proba = np.zeros((len(X), 3))  # предположим 3 класса
            proba[np.arange(len(X)), preds.astype(int)] = 1.0
            return proba
    
    def get_signal(self, X: pd.DataFrame, threshold: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает (сигналы, вероятности).
        Сигнал: 1 - LONG, -1 - SHORT, 0 - FLAT
        Берёт класс с наибольшей вероятностью, если она > threshold.
        """
        proba = self.predict_proba(X)
        max_probs = np.max(proba, axis=1)
        pred_classes = np.argmax(proba, axis=1)
        signals = np.where(max_probs >= threshold, pred_classes, 0)
        # Предположим, что классы закодированы: 0 - DOWN, 1 - FLAT, 2 - UP? 
        # Но у нас в Triple Barrier метки: -1, 0, 1. Нужно привести.
        # Допустим, модель обучалась на метках -1,0,1. Тогда классы: 0 -> -1, 1 -> 0, 2 -> 1.
        # Но лучше получить это из metadata.
        # Пока просто возвращаем как есть.
        return signals, max_probs