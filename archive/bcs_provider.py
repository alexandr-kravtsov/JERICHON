import json
import threading
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Callable, List, Optional
from datetime import datetime, timedelta
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class BCSDataProvider:
    def __init__(self, refresh_token: str, error_list: List[str] = None):
        self.refresh_token = refresh_token
        self.access_token = None
        self.token_expiry = None
        self.headers = {}
        self.running = False
        self.thread = None
        self.on_tick_callback = None
        self.error_list = error_list if error_list is not None else []
        self.tickers = []
        self.last_prices = {}
        
        # Сессия с повторными попытками
        self.session = requests.Session()
        self.session.verify = False
        
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def set_tick_callback(self, callback: Callable[[str, float, int], None]):
        self.on_tick_callback = callback

    def _add_error(self, msg: str):
        if self.error_list is not None:
            self.error_list.append(msg)

    def _get_access_token(self) -> bool:
        try:
            url = "https://be.broker.ru/trade-api-keycloak/realms/tradeapi/protocol/openid-connect/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = {
                "client_id": "trade-api-read",
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token
            }
            response = self.session.post(url, headers=headers, data=data, timeout=30)
            if response.status_code == 200:
                response_data = response.json()
                self.access_token = response_data.get('access_token')
                expires_in = response_data.get('expires_in', 3600)
                self.token_expiry = time.time() + expires_in - 60
                self.headers = {
                    "Authorization": f"Bearer {self.access_token}",
                    "Accept": "application/json"
                }
                return True
            else:
                self._add_error(f"Ошибка получения access-токена: {response.status_code}")
                return False
        except Exception as e:
            self._add_error(f"Исключение при получении access-токена: {e}")
            return False

    def _ensure_token(self):
        if self.access_token is None or self.token_expiry is None or time.time() >= self.token_expiry:
            self._get_access_token()

    def get_historical_candles(self, ticker: str, interval: str, count: int = 200) -> pd.DataFrame:
        self._ensure_token()
        try:
            url = "https://be.broker.ru/trade-api-market-data-connector/api/v1/candles-chart"
            end_date = datetime.now()
            
            if interval == 'M1':
                start_date = end_date - timedelta(minutes=count)
            elif interval == 'M5':
                start_date = end_date - timedelta(minutes=count*5)
            elif interval == 'M15':
                start_date = end_date - timedelta(minutes=count*15)
            elif interval == 'M30':
                start_date = end_date - timedelta(minutes=count*30)
            elif interval == 'H1':
                start_date = end_date - timedelta(hours=count)
            elif interval == 'H4':
                start_date = end_date - timedelta(hours=count*4)
            elif interval == 'D':
                start_date = end_date - timedelta(days=count)
            else:
                start_date = end_date - timedelta(minutes=count)

            params = {
                "classCode": "TQBR",
                "ticker": ticker,
                "startDate": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endDate": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "timeFrame": interval
            }
            response = self.session.get(url, headers=self.headers, params=params, timeout=30)
            if response.status_code == 401:
                self._get_access_token()
                return self.get_historical_candles(ticker, interval, count)
            if response.status_code == 200:
                data = response.json()
                if 'bars' in data and data['bars']:
                    bars = data['bars']
                    df = pd.DataFrame(bars)
                    if 'time' in df.columns:
                        df['datetime'] = pd.to_datetime(df['time'])
                        df.set_index('datetime', inplace=True)
                    rename_map = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
                    df = df.rename(columns=rename_map)
                    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                    
                    # ========== КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ==========
                    # Сортируем по возрастанию (от старых к новым) — как в MOEX API
                    df = df.sort_index()
                    
                    return df
                else:
                    return pd.DataFrame()
            else:
                self._add_error(f"Ошибка загрузки истории: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            self._add_error(f"Исключение при загрузке истории: {e}")
            return pd.DataFrame()

    def _fetch_quotes_rest(self):
        if not self.on_tick_callback:
            return
        self._ensure_token()
        try:
            url = "https://be.broker.ru/trade-api-market-data-connector/api/v1/quotes"
            payload = {
                "instruments": [
                    {"classCode": "TQBR", "ticker": ticker}
                    for ticker in self.tickers
                ]
            }
            response = self.session.post(url, headers=self.headers, json=payload, timeout=30)
            if response.status_code == 401:
                self._get_access_token()
                self._fetch_quotes_rest()
                return
            if response.status_code == 200:
                data = response.json()
                records = data.get('records', [])
                if records:
                    for item in records:
                        ticker = item.get('ticker')
                        price = item.get('last')
                        if ticker and price is not None:
                            price = float(price)
                            timestamp = int(time.time() * 1000)
                            self.on_tick_callback(ticker, price, timestamp)
                            self.last_prices[ticker] = price
            else:
                self._add_error(f"REST ошибка: {response.status_code}")
        except Exception as e:
            self._add_error(f"REST исключение: {e}")

    def _run_rest_loop(self):
        while self.running:
            self._fetch_quotes_rest()
            time.sleep(2)

    def start_stream(self, tickers: List[str]):
        self.tickers = tickers
        if self.running:
            return
        if not self._get_access_token():
            self._add_error("Не удалось получить access-токен")
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_rest_loop, daemon=True)
        self.thread.start()

    def stop_stream(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)